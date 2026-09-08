#!/usr/bin/env python3
"""Verify and exactly-once seal a standalone FINAL-v13 report bundle.

This begins as an intentionally *unfrozen scaffold*.  The sealed FINAL-v12.1
receipt is the immediate parent and its exact 142-source ledger is inherited
without mutation.  The exact 87 extension IDs and eleven semantic Aim-3 IDs are
hard-coded here.  Only FINAL-v13 document hashes remain explicit ``UNFROZEN``
sentinels through Phase 1; a hash-pinned deterministic renderer regenerates and
byte-compares those documents before their strict ``reconciliation_pins.json``
digests are accepted.  That metadata remains outside the scientific source
ledger to avoid a verifier -> Audit document -> verifier self-hash cycle, and
its exact bytes are instead bound into the final receipt.

The implementation delegates generic strict-JSON, no-symlink, stable hashing,
and source-roster validation to the authenticated FINAL-v12.1 verifier.  This
file adds only the FINAL-v13 contract: the six-priority report topology, the
main-model external-data firewall, isolated legacy/target-internal evidence,
the source-bound TCGA+SurGen-primary UNI-v1 Aim-3 result, six-worker scheduler
evidence, and exact ``reviews/k32`` provenance.

Running without an action is read-only.  Only ``--seal`` may create
``report_bundle_receipt.json``.  Publication is deterministic, atomic, and
exactly once; no existing file or symlink is ever overwritten.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import final_v12_1_bundle_receipt as parent  # noqa: E402

BundleVerificationError = parent.BundleVerificationError
sha256_file = parent.sha256_file

FINAL_V13 = REPO / "reports" / "final_v13"
VERIFIER_CODE = Path(__file__).resolve()
VERIFIER_TEST = REPO / "tests" / "test_final_v13_bundle_receipt.py"
PARENT_DIR = REPO / "reports" / "final_v12_1"
PARENT_VERIFIER = REPO / "tools" / "final_v12_1_bundle_receipt.py"
PARENT_TEST = REPO / "tests" / "test_final_v12_1_bundle_receipt.py"

REPORT_DOCUMENTS = parent.REPORT_DOCUMENTS
SOURCE_MANIFEST_NAME = parent.SOURCE_MANIFEST_NAME
FINAL_RECEIPT_NAME = parent.FINAL_RECEIPT_NAME
RECONCILIATION_PINS_NAME = "reconciliation_pins.json"
FULL_RECONCILER_CODE = REPO / "tools" / "final_v13_full_reconciler.py"
PHASE1_CANDIDATE_HELPER = REPO / "tools" / "final_v13_phase1_candidate.py"

PARENT_SEALED_STATUS = "SEALED_FINAL_V12_1_INTEGRATED_PAPER_SELECTION"
PARENT_MANIFEST_STATUS = "candidate_ready_for_final_v12_1_verification"
FINAL_MANIFEST_STATUS = "candidate_ready_for_final_v13_verification"
PHASE1_MANIFEST_STATUS = "candidate_phase1_fine_external_complete_unsealed_final_v13"
INCREMENTAL_MANIFEST_STATUS = (
    "candidate_incremental_two_fixed_and_source_anchored_fewshot_complete_unsealed_final_v13"
)
FINAL_SEALED_STATUS = "SEALED_FINAL_V13_FIREWALLED_INTEGRATED_REPORT"
RECONCILIATION_PINS_STATUS = "FROZEN_FINAL_V13_PRE_SEAL_RECONCILIATION_PINS"
EXPECTED_PARENT_SOURCE_COUNT = 142

# The parent is already sealed.  These pins authenticate the immediate parent
# before its own verifier is replayed recursively.
EXPECTED_PARENT_RECEIPT_SHA256 = "66b56f4b85e7216c31e2c26c295b50cac5f5cae2cd1558bb30e855946cc57fd5"
EXPECTED_PARENT_MANIFEST_SHA256 = "9a3a46336cd54b983c8143ad8d61fdbe776f6987fe304f62f5d8d51f464e678e"
EXPECTED_PARENT_VERIFIER_SHA256 = "9881cf1495c0dea25445f0497747b833672b47bb5e3332ea3dc878ef706b1947"
EXPECTED_PARENT_TEST_SHA256 = "3eb5767f2881e763ef27239b93d295c26537a17e7221db7b23804e2f01d2b6ce"
EXPECTED_PARENT_DOCUMENT_SHA256 = {
    "Experimental_Setup.md": "3b9218a581f9277323a7e7cbd15280b8cb3831fa9073de0b0e0e235ca11e09f9",
    "Results.md": "9120c1f3ad64d3ff3279eb05f24dfd9362842ca06e8cfb48193a8e083170ddb7",
    "Audit.md": "eaea6be69e0cf8ff00c165cc1589f973cafd16622222a9be94165d75e677d8d3",
}

# The conventional external-primary headline is replayed directly from this
# inherited, hash-pinned patient-level artifact.  CPTAC-primary and RIH-primary
# are concatenated at patient level before computing one AUROC; cohort AUROCs
# are never averaged.  Orion is intentionally excluded because it is a
# separate retrospective cross-protocol processing sensitivity.
EXPECTED_AIM2_PRIMARY_POOL_SOURCE_ID = (
    "aim1-tcga-surgen-two-encoder-downstream-v3-patient-native-logits"
)
EXPECTED_AIM2_PRIMARY_POOL_SOURCE_ROLE = "governed_patient_native_logit_table"
EXPECTED_AIM2_PRIMARY_POOL_SOURCE_PATH = (
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim1_tcga_surgen_two_encoder_v1_20260824/downstream_v2/continuation_v3/"
    "analysis/patient_native_logits.parquet"
)
EXPECTED_AIM2_PRIMARY_POOL_SOURCE_SIZE = 311_419
EXPECTED_AIM2_PRIMARY_POOL_SOURCE_SHA256 = (
    "02f4a5bf0c2cd909c5ecb3f9edc7c05ae725649199a015cf0a1d0cb544df8d09"
)
EXPECTED_AIM2_PRIMARY_POOL_DATASETS = ("cptac_primary", "rih_primary")
EXPECTED_AIM2_PRIMARY_POOL_DATASET_CENSUS = {
    "cptac_primary": (94, 33, 61),
    "rih_primary": (153, 70, 83),
}
EXPECTED_AIM2_PRIMARY_POOL_CENSUS = (247, 103, 144)
EXPECTED_AIM2_PRIMARY_POOL_AUROC = 0.7445051240560949
EXPECTED_AIM2_PRIMARY_POOL_ANALYSIS_FAMILY = "target_refit_zero_shot_or_sensitivity"
EXPECTED_AIM2_PRIMARY_POOL_ENCODER = "univ1"
EXPECTED_AIM2_PRIMARY_POOL_SEED_COLUMNS = tuple(
    f"logit_seed{seed}" for seed in (42, 43, 44, 45, 46)
)

# Document digests cannot be embedded here because Audit.md includes this
# verifier's digest.  The trusted, hash-pinned reconciliation generator renders
# and byte-compares those documents before the pins are accepted.  Scientific
# membership and semantic source IDs have no such cycle and are hard-coded
# below: reconciliation_pins.json may repeat them, but can never choose them.
UNFROZEN = "UNFROZEN_FINAL_V13_RECONCILIATION_REQUIRED"
EXPECTED_FINAL_DOCUMENT_SHA256 = {
    "Experimental_Setup.md": UNFROZEN,
    "Results.md": UNFROZEN,
    "Audit.md": UNFROZEN,
}
EXPECTED_PHASE1_BASE_EXTENSION_SOURCE_IDS = (
    "aim2-e2c-direct-analysis-audit",
    "aim2-e2c-direct-lineage-completion",
    "aim2-e2c-direct-results",
    "aim3-source-primary-campaign-contract",
    "aim3-source-primary-campaign-controller",
    "aim3-source-primary-campaign-test",
    "aim3-source-primary-controls-scheduler",
    "aim3-source-primary-fine-external-analysis-completion",
    "aim3-source-primary-fine-external-campaign-contract",
    "aim3-source-primary-fine-external-campaign-controller",
    "aim3-source-primary-fine-external-campaign-test",
    "aim3-source-primary-fine-external-inference-seal",
    "aim3-source-primary-fine-external-preflight",
    "aim3-source-primary-fine-external-results",
    "aim3-source-primary-fine-external-scoring-completion",
    "aim3-source-primary-fine-results",
    "aim3-source-primary-fine-scheduler",
    "aim3-source-primary-fine-training-completion",
    "aim3-source-primary-preflight",
    "aim3-source-primary-results",
    "aim3-source-primary-scheduler",
    "aim3-source-primary-source-contract",
    "aim3-source-primary-source-integrity",
    "aim3-source-primary-source-manifest",
    "aim3-source-primary-source-splits",
    "aim3-source-primary-source-summary",
    "aim3-source-primary-source-training-seal",
    "aim3-source-primary-training-completion",
    "aim4-k32-human-read-receipt",
    "aim4-k32-source-01",
    "aim4-k32-source-02",
    "aim4-k32-source-03",
    "aim4-k32-source-04",
    "aim4-k32-source-05",
    "aim4-k32-source-06",
    "aim4-k32-source-07",
    "aim4-k32-source-08",
    "aim4-k32-source-09",
    "aim4-k32-source-10",
    "aim4-k32-source-11",
    "aim4-k32-source-12",
    "aim4-k32-source-13",
    "aim4-k32-source-14",
    "aim4-k32-source-15",
    "aim4-k32-source-16",
    "final-v13-bundle-verifier",
    "final-v13-bundle-verifier-test",
)
EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS = (
    "aim2-source-anchored-pure-ridge-analysis-completion",
    "aim2-source-anchored-pure-ridge-contract",
    "aim2-source-anchored-pure-ridge-controller",
    "aim2-source-anchored-pure-ridge-controller-test",
    "aim2-source-anchored-pure-ridge-deep-preflight",
    "aim2-source-anchored-pure-ridge-inference-environment",
    "aim2-source-anchored-pure-ridge-inference-seal",
    "aim2-source-anchored-pure-ridge-label-blind-rih-manifest",
    "aim2-source-anchored-pure-ridge-label-blind-surgen-manifest",
    "aim2-source-anchored-pure-ridge-labeled-target-manifest",
    "aim2-source-anchored-pure-ridge-oof-predictions",
    "aim2-source-anchored-pure-ridge-results",
    "aim2-source-anchored-pure-ridge-scheduler",
    "aim2-source-anchored-pure-ridge-source-embedding-scheduler",
    "aim2-source-anchored-pure-ridge-target-internal-open",
    "aim2-source-anchored-residual-analysis-completion",
    "aim2-source-anchored-residual-contract",
    "aim2-source-anchored-residual-controller",
    "aim2-source-anchored-residual-controller-test",
    "aim2-source-anchored-residual-oof-predictions",
    "aim2-source-anchored-residual-preflight",
    "aim2-source-anchored-residual-results",
    "aim2-source-anchored-residual-scheduler",
    "aim3-source-primary-two-fixed-analysis-completion",
    "aim3-source-primary-two-fixed-bootstrap-distributions",
    "aim3-source-primary-two-fixed-contract",
    "aim3-source-primary-two-fixed-control-manifest-codon",
    "aim3-source-primary-two-fixed-control-manifest-g12d-broad",
    "aim3-source-primary-two-fixed-control-split-integrity-codon",
    "aim3-source-primary-two-fixed-control-split-integrity-g12d-broad",
    "aim3-source-primary-two-fixed-control-splits-codon",
    "aim3-source-primary-two-fixed-control-splits-g12d-broad",
    "aim3-source-primary-two-fixed-controller",
    "aim3-source-primary-two-fixed-controller-test",
    "aim3-source-primary-two-fixed-job-plan",
    "aim3-source-primary-two-fixed-patient-native-logits",
    "aim3-source-primary-two-fixed-preflight",
    "aim3-source-primary-two-fixed-results",
    "aim3-source-primary-two-fixed-scheduler",
    "aim3-source-primary-two-fixed-training-completion",
)
EXPECTED_FULL_EXTENSION_SOURCE_IDS = tuple(
    sorted(
        (
            *EXPECTED_PHASE1_BASE_EXTENSION_SOURCE_IDS,
            *EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS,
        )
    )
)
# The incremental candidate and eventual full release intentionally share the
# same 87-ID scientific extension authority.  The only Phase-2 transition is
# promotion of the four explicitly pending Aim-3 records to material records.
EXPECTED_INCREMENTAL_EXTENSION_SOURCE_IDS = EXPECTED_FULL_EXTENSION_SOURCE_IDS
EXPECTED_EXTENSION_SOURCE_IDS = EXPECTED_FULL_EXTENSION_SOURCE_IDS
if (
    len(EXPECTED_PHASE1_BASE_EXTENSION_SOURCE_IDS) != 47
    or len(EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS) != 40
    or len(EXPECTED_INCREMENTAL_EXTENSION_SOURCE_IDS) != 87
    or len(set(EXPECTED_INCREMENTAL_EXTENSION_SOURCE_IDS)) != 87
    or set(EXPECTED_PHASE1_BASE_EXTENSION_SOURCE_IDS).intersection(
        EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS
    )
):
    raise RuntimeError("FINAL-v13 incremental source roster constant drift")
EXPECTED_AIM3_RESULTS_SOURCE_ID = "aim3-source-primary-results"
EXPECTED_AIM3_SCHEDULER_SOURCE_ID = "aim3-source-primary-scheduler"
EXPECTED_AIM3_TRAINING_SOURCE_ID = "aim3-source-primary-training-completion"
EXPECTED_AIM3_FINE_RESULTS_SOURCE_ID = "aim3-source-primary-fine-results"
EXPECTED_AIM3_FINE_SCHEDULER_SOURCE_ID = "aim3-source-primary-fine-scheduler"
EXPECTED_AIM3_FINE_TRAINING_SOURCE_ID = "aim3-source-primary-fine-training-completion"
EXPECTED_AIM3_CONTROLS_SCHEDULER_SOURCE_ID = "aim3-source-primary-controls-scheduler"
EXPECTED_AIM3_FINE_EXTERNAL_RESULTS_SOURCE_ID = "aim3-source-primary-fine-external-results"
EXPECTED_AIM3_FINE_EXTERNAL_COMPLETION_SOURCE_ID = (
    "aim3-source-primary-fine-external-analysis-completion"
)
EXPECTED_AIM3_FINE_EXTERNAL_INFERENCE_SEAL_SOURCE_ID = (
    "aim3-source-primary-fine-external-inference-seal"
)
EXPECTED_AIM3_FINE_EXTERNAL_SCORING_SOURCE_ID = (
    "aim3-source-primary-fine-external-scoring-completion"
)
EXPECTED_AIM3_FINE_EXTERNAL_CONTRACT_SOURCE_ID = (
    "aim3-source-primary-fine-external-campaign-contract"
)
EXPECTED_AIM3_FINE_EXTERNAL_CONTROLLER_SOURCE_ID = (
    "aim3-source-primary-fine-external-campaign-controller"
)
EXPECTED_AIM3_FINE_EXTERNAL_TEST_SOURCE_ID = "aim3-source-primary-fine-external-campaign-test"
EXPECTED_AIM3_FINE_EXTERNAL_PREFLIGHT_SOURCE_ID = "aim3-source-primary-fine-external-preflight"

EXPECTED_EXTENSION_SOURCE_COUNT = len(EXPECTED_EXTENSION_SOURCE_IDS)
EXPECTED_FINAL_SOURCE_COUNT = EXPECTED_PARENT_SOURCE_COUNT + EXPECTED_EXTENSION_SOURCE_COUNT

# Patched only after both trusted generators reach their final bytes.  Unlike
# document digests, these hashes do not participate in the Audit self-hash
# cycle because neither helper embeds this verifier's generated documents.
EXPECTED_FULL_RECONCILER_SHA256 = "f739961c2a2ee69314be75f9b2b43ba37b78270b6ec7479bc8336f5a497d38a1"
EXPECTED_PHASE1_CANDIDATE_HELPER_SHA256 = (
    "571d1f5c2c019ee23aae1a5059e0ac90974bf6611ac223ff07acf29aefdadbd3"
)

EXPECTED_AIM3_EXPERIMENT = "aim3_tcga_surgen_primary_univ1_5seed"
EXPECTED_AIM3_RUN_ROOT = (
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim3_tcga_surgen_primary_univ1_5seed_v2_20260827"
)
EXPECTED_AIM3_ANALYSIS_COMPLETION_PATH = (
    f"{EXPECTED_AIM3_RUN_ROOT}/analysis/analysis_completion.json"
)
EXPECTED_AIM3_ANALYSIS_COMPLETION_SIZE = 909
EXPECTED_AIM3_ANALYSIS_COMPLETION_SHA256 = (
    "ceff89363f891eb379bde3b675047e22c279158b800dd94e5a5b9cb1c418768d"
)
EXPECTED_AIM3_PATIENT_NATIVE_LOGITS_PATH = (
    f"{EXPECTED_AIM3_RUN_ROOT}/analysis/patient_native_logits.parquet"
)
EXPECTED_AIM3_PATIENT_NATIVE_LOGITS_SIZE = 80_321
EXPECTED_AIM3_PATIENT_NATIVE_LOGITS_SHA256 = (
    "b124eaede8934cb814a276225e7e94c16deb019ec66d0d2e3321ffa7db21ce4f"
)
EXPECTED_AIM3_BOOTSTRAP_DISTRIBUTIONS_PATH = (
    f"{EXPECTED_AIM3_RUN_ROOT}/analysis/bootstrap_distributions.npz"
)
EXPECTED_AIM3_BOOTSTRAP_DISTRIBUTIONS_SIZE = 4_504_606
EXPECTED_AIM3_BOOTSTRAP_DISTRIBUTIONS_SHA256 = (
    "74ee6e54a0202642cf9c674dfac8ab4cdbe23d4393a320fa89570cf605755e47"
)
EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT = "aim3_tcga_surgen_primary_fine_external_univ1_5seed"
EXPECTED_AIM3_FINE_EXTERNAL_RUN_ROOT = (
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim3_tcga_surgen_primary_fine_external_v1_20260828"
)
EXCLUDED_FAILED_AIM3_V1_ROOT = (
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim3_tcga_surgen_primary_univ1_5seed_v1_20260827"
)
FAILED_V1_AUDIT_MARKER = (
    "FAILED_PRE_FIT_NONCONTROLLING_ROOT: "
    f"`{EXCLUDED_FAILED_AIM3_V1_ROOT}`; zero requests, runs, or model fits; "
    "excluded from authoritative science sources."
)
MODEL_SEEDS = (42, 43, 44, 45, 46)
WT_DRAW_SEEDS = (20260823, 20260824, 20260825)
AIM3_RUNG_CONTROL = {
    "allele1": "ctrl_allele1",
    "allele2": "ctrl_allele2",
    "codon": "ctrl_codon",
    "g12c": "ctrl_g12c",
    "g12d_broad": "ctrl_g12d_broad",
}
EXPECTED_TRAINING_COUNTS = {
    "task_variants": 25,
    "chains": 125,
    "oof_folds": 625,
    "p75_refits": 125,
    "physical_mil_fits": 750,
}
EXPECTED_FINE_TRAINING_COUNTS = {
    "task_variants": 5,
    "chains": 25,
    "oof_folds": 125,
    "p75_refits": 25,
    "physical_mil_fits": 150,
}
EXPECTED_CONTROL_TRAINING_COUNTS = {
    "task_variants": 20,
    "chains": 100,
    "oof_folds": 500,
    "p75_refits": 100,
    "physical_mil_fits": 600,
}
EXPECTED_PARALLELISM = 6

PRIORITY_HEADINGS = (
    "## Priority 1 — Aim 1 TCGA+SurGen-primary main model and baseline",
    "## Priority 2 — Aim 2 fixed-model zero-shot external validation",
    "## Priority 3 — Aim 2 multi-cohort LOCO",
    "## Priority 4 — Aim 2 metastatic few-shot adaptation and theoretical ceiling",
    "## Priority 5 — Aim 3 population ladders and repeated-WT-control consensus",
    "## Priority 6 — Aim 4 reviews/k32",
)

FIREWALL_MARKERS = (
    "MAIN_MODEL_DEVELOPMENT: TCGA-COAD, TCGA-READ, SR386-primary, and SR1482-primary only.",
    "EXTERNAL_TEST_DATA: CPTAC-primary, Orion, RIH-primary, RIH-metastatic, "
    "and SurGen-metastatic never enter main-model development.",
    "EXTERNAL_ALLOWED_OPERATIONS: label-blind packing/preprocessing and frozen-model scoring only.",
    "EXTERNAL_FORBIDDEN_OPERATIONS: training, fitting, fine-tuning, adaptation, "
    "calibration, threshold selection, hyperparameter selection, vocabulary "
    "fitting, and construction/model selection.",
    "TARGET_INTERNAL_EXCEPTION: any target-label few-shot or ceiling analysis "
    "forfeits external-validation status and cannot modify or select the main model.",
)

EXTERNAL_TARGET_ROLES = {
    "CPTAC-primary": "external_primary_validation",
    "Orion": "external_cross_protocol_test",
    "RIH-metastatic": "external_metastatic_validation",
    "RIH-primary": "external_primary_validation",
    "SurGen-metastatic": "external_metastatic_test_source_family_exposed",
}

AIM3_FINE_TASKS = ("codon", "g12d_broad", "allele1", "allele2", "g12c")
AIM3_FINE_EXTERNAL_TARGETS = (
    "cptac_primary",
    "orion_primary",
    "rih_primary",
    "rih_metastatic",
    "surgen_metastatic",
)
AIM3_FINE_EXTERNAL_REPORT_SCOPES = (
    *AIM3_FINE_EXTERNAL_TARGETS,
    "primary_only",
    "metastatic_only",
    "strict_disjoint_combined",
)
AIM3_FINE_EXTERNAL_SECONDARY_SCOPES = (
    "equal_cohort_macro_strict_disjoint",
    "patient_role_pooled_clustered_sensitivity",
)
AIM3_FINE_EXTERNAL_EXPECTED_INTERNAL_CENSUS = {
    "codon": (354, 147),
    "g12d_broad": (154, 347),
    "allele1": (154, 200),
    "allele2": (102, 252),
    "g12c": (46, 308),
}
AIM3_FINE_EXTERNAL_EXPECTED_CENSUS = {
    "codon": {
        "cptac_primary": (19, 14),
        "orion_primary": (12, 3),
        "rih_primary": (47, 23),
        "rih_metastatic": (23, 14),
        "surgen_metastatic": (19, 11),
        "primary_only": (78, 40),
        "metastatic_only": (42, 25),
        "strict_disjoint_combined": (113, 63),
    },
    "g12d_broad": {
        "cptac_primary": (11, 22),
        "orion_primary": (6, 9),
        "rih_primary": (21, 49),
        "rih_metastatic": (12, 25),
        "surgen_metastatic": (6, 24),
        "primary_only": (38, 80),
        "metastatic_only": (18, 49),
        "strict_disjoint_combined": (52, 124),
    },
    "allele1": {
        "cptac_primary": (11, 8),
        "orion_primary": (6, 6),
        "rih_primary": (21, 26),
        "rih_metastatic": (12, 11),
        "surgen_metastatic": (6, 13),
        "primary_only": (38, 40),
        "metastatic_only": (18, 24),
        "strict_disjoint_combined": (52, 61),
    },
    "allele2": {
        "cptac_primary": (6, 13),
        "orion_primary": (2, 10),
        "rih_primary": (17, 30),
        "rih_metastatic": (6, 17),
        "surgen_metastatic": (7, 12),
        "primary_only": (25, 53),
        "metastatic_only": (13, 29),
        "strict_disjoint_combined": (35, 78),
    },
    "g12c": {
        "cptac_primary": (2, 17),
        "orion_primary": (2, 10),
        "rih_primary": (5, 42),
        "rih_metastatic": (2, 21),
        "surgen_metastatic": (4, 15),
        "primary_only": (9, 69),
        "metastatic_only": (6, 36),
        "strict_disjoint_combined": (15, 98),
    },
}
AIM3_FINE_EXTERNAL_DUAL_RIH_PATIENTS = (
    "RIH:RIH_001216ba7a08c070",
    "RIH:RIH_1845b46a817ef51c",
    "RIH:RIH_24bda4bdbf9140a5",
    "RIH:RIH_28ed5c0131dcfa60",
    "RIH:RIH_3c68f85359d030c4",
    "RIH:RIH_59b36f4590fc4525",
    "RIH:RIH_9db23d204671f3e8",
    "RIH:RIH_c1bac72156d3e1a8",
)

SECONDARY_MARKERS = (
    "SECONDARY_LEGACY_MULTI_COHORT_NOT_MAIN_MODEL_NOT_EXTERNAL_VALIDATION",
    "SECONDARY_TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION",
    "SECONDARY_LEGACY_ALL_VALID_NOT_EXTERNAL_COMPLIANT",
)
THEORETICAL_CEILING_NOT_RUN_MARKER = (
    "THEORETICAL_CEILING_NOT_RUN/INELIGIBLE under external-test firewall"
)
AIM3_CEILING_SCOPE_MARKER = (
    "AIM3_CEILING_SCOPE: CEILING and CONSENSUS_CEILING are governed matched-WT "
    "comparator results specific to this model, recipe, and sample; they never "
    "establish biological or theoretical absence of fine molecular signal."
)
AIM3_FIXED_NONOVERRIDE_MARKER = (
    "AIM3_FIXED_NONOVERRIDE: a fixed single-draw CEILING cannot override the repeated-WT consensus."
)
SOURCE_ONLY_RESULTS_EXCLUSION_MARKERS = (
    "No LOCO performance is reported in FINAL-v13 Results.",
    "Legacy non-source-anchored few-shot, local-training, and target-label ceiling",
    "No all-valid-patient Aim-3 performance is reported in FINAL-v13 Results.",
)
INCREMENTAL_RESULTS_EXCLUSION_MARKERS = (
    "No LOCO performance is reported in FINAL-v13 Results.",
    "Legacy non-source-anchored few-shot, local-training, and target-label ceiling",
    "No all-valid-patient Aim-3 performance is reported in FINAL-v13 Results.",
)
FORBIDDEN_INELIGIBLE_RESULTS_NUMERIC_MARKERS = (
    "0.7097 [0.6721, 0.7464]",
    "0.4525 [0.3613, 0.5445]",
    "0.5949 [0.5019, 0.6860]",
    "0.5085/0.6612",
    "ALL-primary reference AUROC 0.6788",
)
FORBIDDEN_INCREMENTAL_INELIGIBLE_NUMERIC_MARKERS = (
    *FORBIDDEN_INELIGIBLE_RESULTS_NUMERIC_MARKERS,
    "0.9111 [0.8826, 0.9354]",
    "0.445 [0.417, 0.476]",
    "0.466 [0.421, 0.508]",
    "0.6571 [0.6289, 0.6853]",
    "0.6579 [0.6284, 0.6856]",
    "0.6851 [0.6548, 0.7143]",
)
AIM4_BOUNDARY_MARKER = "LEGACY_TRANSDUCTIVE_LABEL_BLIND_NOT_EXTERNAL_VALIDATION"
RECONCILIATION_PINS_BOUNDARY_MARKER = (
    "PRE_SEAL_RECONCILIATION_PINS_METADATA_NOT_SCIENTIFIC_EVIDENCE"
)
RECONCILIATION_PINS_MEMBERSHIP = (
    "EXCLUDED_PRE_SEAL_METADATA_TO_BREAK_VERIFIER_DOCUMENT_SELF_HASH_CYCLE"
)
AIM3_TERMINAL_ANALYSIS_COMPLETION_MARKER = (
    "AIM3_TERMINAL_ANALYSIS_COMPLETION: hard-pinned controller certification "
    "of the ledgered results and its patient-logit/bootstrap primitives; not "
    "an additional scientific source row. SHA-256 "
    f"`{EXPECTED_AIM3_ANALYSIS_COMPLETION_SHA256}`."
)
AIM3_GATE_QUALIFICATION_MARKER = (
    "AIM3_GATE_RULE: displayed two-sided 95% confidence intervals are "
    "descriptive; fixed/draw verdicts and repeated-WT consensus use the "
    "predeclared one-sided 99% Bonferroni/FWER bounds, never the 95% intervals."
)

# Exact material contents of reviews/k32.  The Windows Zone.Identifier sidecar
# is authenticated and explicitly excluded as non-evidence; all other files
# must be one-to-one authoritative source-manifest records.
EXPECTED_AIM4_K32_SHA256 = {
    "README.md": "b9d1b389320abf5b99c16455c7b62429e9ac0b5e0b131b46a3f54ded12f65858",
    "completed_review.md": "0787411e819d9020540e14e734035556e78f603ebae4a4196996ce8404156cce",
    "completed_review_structured.json": "e595fb1771b07ab4ab34189f50f8348650fb4e6cac096f0995b845eb59716a91",
    "followup_review_M04_M05_M07.md": "6eae0acb1ab399fb688ac9760a16a6386e896902c994456a4260155ffbf8e6ea",
    "montages/M01.jpg": "66428fac9f86b22028d07ad1a9cfe42915269e651d9e2190371189b89bfdf146",
    "montages/M02.jpg": "45787198e3010e16d8f7afdfc9181094106e807ce14ac54ab89275bc777bb9d1",
    "montages/M03.jpg": "1f1d89d58406e22736fc96e3a43feaed273b97ced2f948188536107645874c90",
    "montages/M04.jpg": "d706f5cecfb327f422dfd2ee66d618965bb896dfbf7b35b313f1c8b924979711",
    "montages/M05.jpg": "0e8005a1a0ab4df2c878db6e131624dbc70d6cbee7f220ff967c1c78f1735725",
    "montages/M06.jpg": "e4ee668a874ad913a25227eecd7d81ad623ccd87fb3017e7ad30d87406c19ac7",
    "montages/M07.jpg": "afe3af8fe64887ddc88b08376749b14bbd52c13ee5f3091042523157179d0448",
    "montages/M08.jpg": "65af4f05ae92496e2aad0df017b181e0285fe4f8d100e8d6d0d3b3fc729d6ba9",
    "montages/M09.jpg": "70785ba7260374217329631c55de79f83f23312e53b1fb9d0760c9f0269b952f",
    "montages/M10.jpg": "f44eaefddc91cb8f6ab42c3547185075d03797a3b36c6374a51d14a2530c0788",
    "montages/M11.jpg": "02c1f33726e6fa0e8ad76b717621f94c51636bf00396ac16c08cc4364667f6c0",
    "review_form.csv": "797607199eb2bd2583b677b04bbed53bf246c69800f754b3c89d41f4ed31ca05",
}
EXPECTED_AIM4_K32_SIDECAR_SHA256 = {
    "completed_review.md:Zone.Identifier": (
        "6fad15223f42be5a5edc046d315a03e9980e24271378d86239de28c004e241f2"
    )
}
EXPECTED_AIM4_CURATED_FROM_SHA256 = (
    "5098c58b23c905163c456004099bac46b9c43166bd49dbcc1b5558c1f0ddf8cf"
)

_MANIFEST_KEYS = parent._MANIFEST_KEYS
_PENDING_SOURCE_KEYS = {"id", "aims", "experiments", "role", "path"}
_RECONCILIATION_PINS_KEYS = {
    "schema_version",
    "bundle",
    "status",
    "parent_final_v12_1",
    "extension_source_count",
    "extension_source_ids",
    "aim3_source_ids",
    "documents",
    "verifier",
    "verifier_test",
    "scientific_source_manifest_membership",
}
_RECONCILIATION_PARENT_KEYS = {
    "status",
    "receipt_sha256",
    "source_manifest_sha256",
}
_RECONCILIATION_AIM3_KEYS = {
    "results",
    "scheduler",
    "training_completion",
    "fine_results",
    "fine_scheduler",
    "fine_training_completion",
    "controls_scheduler",
    "fine_external_results",
    "fine_external_analysis_completion",
    "fine_external_inference_seal",
    "fine_external_scoring_completion",
}
_RECONCILIATION_TOOL_KEYS = {"sha256"}
_SHA256_RE = parent._SHA256_RE
_SOURCE_ID_RE = parent._SOURCE_ID_RE
_AIM3_BINDING_ID_RE = re.compile(r"[a-z0-9]+(?:[a-z0-9._-]*[a-z0-9])?")


@dataclass(frozen=True)
class BundlePaths:
    """Filesystem locations plus immutable parent and child expectations."""

    repo: Path
    final_v13: Path
    destination: Path
    reconciliation_pins: Path
    verifier_code: Path
    verifier_test: Path
    parent_dir: Path
    parent_receipt: Path
    parent_manifest: Path
    parent_verifier: Path
    parent_test: Path
    expected_parent_receipt_sha256: str
    expected_parent_manifest_sha256: str
    expected_parent_verifier_sha256: str
    expected_parent_test_sha256: str
    expected_parent_document_sha256: Mapping[str, str]
    expected_final_document_sha256: Mapping[str, str]
    expected_extension_source_ids: Sequence[str]
    aim3_results_source_id: str
    aim3_scheduler_source_id: str
    aim3_training_source_id: str
    aim3_fine_results_source_id: str
    aim3_fine_scheduler_source_id: str
    aim3_fine_training_source_id: str
    aim3_controls_scheduler_source_id: str
    aim3_fine_external_results_source_id: str
    aim3_fine_external_completion_source_id: str
    aim3_fine_external_inference_seal_source_id: str
    aim3_fine_external_scoring_source_id: str
    expected_aim3_experiment: str
    expected_aim3_run_root: str
    expected_aim3_fine_external_experiment: str
    expected_aim3_fine_external_run_root: str
    aim4_k32_root: Path
    expected_aim4_k32_sha256: Mapping[str, str]
    expected_aim4_k32_sidecar_sha256: Mapping[str, str]
    expected_aim4_curated_from_sha256: str
    expected_parent_source_count: int = EXPECTED_PARENT_SOURCE_COUNT
    expected_parent_status: str = PARENT_SEALED_STATUS
    expected_parent_manifest_status: str = PARENT_MANIFEST_STATUS
    expected_final_manifest_status: str = FINAL_MANIFEST_STATUS
    expected_phase1_manifest_status: str = PHASE1_MANIFEST_STATUS
    replay_parent_verifier: bool = True


def default_paths() -> BundlePaths:
    """Return production paths; full child pins resolve from strict metadata."""

    return BundlePaths(
        repo=REPO,
        final_v13=FINAL_V13,
        destination=FINAL_V13 / FINAL_RECEIPT_NAME,
        reconciliation_pins=FINAL_V13 / RECONCILIATION_PINS_NAME,
        verifier_code=VERIFIER_CODE,
        verifier_test=VERIFIER_TEST,
        parent_dir=PARENT_DIR,
        parent_receipt=PARENT_DIR / FINAL_RECEIPT_NAME,
        parent_manifest=PARENT_DIR / SOURCE_MANIFEST_NAME,
        parent_verifier=PARENT_VERIFIER,
        parent_test=PARENT_TEST,
        expected_parent_receipt_sha256=EXPECTED_PARENT_RECEIPT_SHA256,
        expected_parent_manifest_sha256=EXPECTED_PARENT_MANIFEST_SHA256,
        expected_parent_verifier_sha256=EXPECTED_PARENT_VERIFIER_SHA256,
        expected_parent_test_sha256=EXPECTED_PARENT_TEST_SHA256,
        expected_parent_document_sha256=dict(EXPECTED_PARENT_DOCUMENT_SHA256),
        expected_final_document_sha256=dict(EXPECTED_FINAL_DOCUMENT_SHA256),
        expected_extension_source_ids=EXPECTED_EXTENSION_SOURCE_IDS,
        aim3_results_source_id=EXPECTED_AIM3_RESULTS_SOURCE_ID,
        aim3_scheduler_source_id=EXPECTED_AIM3_SCHEDULER_SOURCE_ID,
        aim3_training_source_id=EXPECTED_AIM3_TRAINING_SOURCE_ID,
        aim3_fine_results_source_id=EXPECTED_AIM3_FINE_RESULTS_SOURCE_ID,
        aim3_fine_scheduler_source_id=EXPECTED_AIM3_FINE_SCHEDULER_SOURCE_ID,
        aim3_fine_training_source_id=EXPECTED_AIM3_FINE_TRAINING_SOURCE_ID,
        aim3_controls_scheduler_source_id=EXPECTED_AIM3_CONTROLS_SCHEDULER_SOURCE_ID,
        aim3_fine_external_results_source_id=(EXPECTED_AIM3_FINE_EXTERNAL_RESULTS_SOURCE_ID),
        aim3_fine_external_completion_source_id=(EXPECTED_AIM3_FINE_EXTERNAL_COMPLETION_SOURCE_ID),
        aim3_fine_external_inference_seal_source_id=(
            EXPECTED_AIM3_FINE_EXTERNAL_INFERENCE_SEAL_SOURCE_ID
        ),
        aim3_fine_external_scoring_source_id=(EXPECTED_AIM3_FINE_EXTERNAL_SCORING_SOURCE_ID),
        expected_aim3_experiment=EXPECTED_AIM3_EXPERIMENT,
        expected_aim3_run_root=EXPECTED_AIM3_RUN_ROOT,
        expected_aim3_fine_external_experiment=(EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT),
        expected_aim3_fine_external_run_root=EXPECTED_AIM3_FINE_EXTERNAL_RUN_ROOT,
        aim4_k32_root=REPO / "reviews" / "k32",
        expected_aim4_k32_sha256=dict(EXPECTED_AIM4_K32_SHA256),
        expected_aim4_k32_sidecar_sha256=dict(EXPECTED_AIM4_K32_SIDECAR_SHA256),
        expected_aim4_curated_from_sha256=EXPECTED_AIM4_CURATED_FROM_SHA256,
    )


def _display(path: Path, repo: Path) -> str:
    return parent._display(path, repo)


def _identity(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    return parent.identity(path, display_path=display_path)


def _strict_stable_json(
    path: Path, *, label: str, repo: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    return parent._load_stable_json(
        path,
        label=label,
        display_path=_display(path, repo),
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_production_bundle(paths: BundlePaths) -> bool:
    """Return whether paths address the one publishable repository bundle."""

    return (
        paths.repo.absolute() == REPO.absolute()
        and paths.final_v13.absolute() == FINAL_V13.absolute()
        and paths.destination.absolute() == (FINAL_V13 / FINAL_RECEIPT_NAME).absolute()
        and paths.reconciliation_pins.absolute()
        == (FINAL_V13 / RECONCILIATION_PINS_NAME).absolute()
        and paths.verifier_code.absolute() == VERIFIER_CODE.absolute()
        and paths.verifier_test.absolute() == VERIFIER_TEST.absolute()
    )


def _trusted_reconciliation_generator_identities() -> dict[str, dict[str, Any]]:
    """Authenticate the two generators before importing either one."""

    expected = {
        "full_reconciler": (FULL_RECONCILER_CODE, EXPECTED_FULL_RECONCILER_SHA256),
        "phase1_candidate_helper": (
            PHASE1_CANDIDATE_HELPER,
            EXPECTED_PHASE1_CANDIDATE_HELPER_SHA256,
        ),
    }
    identities: dict[str, dict[str, Any]] = {}
    for name, (path, digest) in expected.items():
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise BundleVerificationError(f"trusted FINAL-v13 {name} SHA-256 pin is not frozen")
        identity = _identity(path, display_path=_display(path, REPO))
        if identity["sha256"] != digest:
            raise BundleVerificationError(f"trusted FINAL-v13 {name} SHA-256 drift")
        identities[name] = identity
    return identities


def _load_trusted_full_reconciler() -> tuple[types.ModuleType, dict[str, Any]]:
    """Load the exact hash-pinned renderer without trusting import search order."""

    identities = _trusted_reconciliation_generator_identities()
    module_name = (
        f"_final_v13_trusted_full_reconciler_{identities['full_reconciler']['sha256'][:16]}"
    )
    spec = importlib.util.spec_from_file_location(module_name, FULL_RECONCILER_CODE)
    if spec is None or spec.loader is None:
        raise BundleVerificationError("trusted FINAL-v13 full reconciler cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise BundleVerificationError(
            f"trusted FINAL-v13 full reconciler import failed: {exc}"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    imported_phase1 = getattr(module, "phase1", None)
    imported_phase1_file = getattr(imported_phase1, "__file__", None)
    if (
        not isinstance(imported_phase1_file, str)
        or Path(imported_phase1_file).absolute() != PHASE1_CANDIDATE_HELPER.absolute()
    ):
        raise BundleVerificationError(
            "trusted FINAL-v13 full reconciler imported an unexpected Phase-1 helper"
        )
    return module, identities


def _load_trusted_phase1_candidate() -> tuple[types.ModuleType, dict[str, Any]]:
    """Load the exact helper used for incremental evidence and byte replay."""

    identities = _trusted_reconciliation_generator_identities()
    module_name = (
        "_final_v13_trusted_phase1_candidate_"
        f"{identities['phase1_candidate_helper']['sha256'][:16]}"
    )
    spec = importlib.util.spec_from_file_location(module_name, PHASE1_CANDIDATE_HELPER)
    if spec is None or spec.loader is None:
        raise BundleVerificationError("trusted FINAL-v13 incremental helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise BundleVerificationError(
            f"trusted FINAL-v13 incremental helper import failed: {exc}"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    helper_ids = tuple(module.extension_ids())
    if (
        helper_ids != EXPECTED_INCREMENTAL_EXTENSION_SOURCE_IDS
        or getattr(module, "INCREMENTAL_STATUS", None) != INCREMENTAL_MANIFEST_STATUS
    ):
        raise BundleVerificationError("trusted FINAL-v13 incremental helper roster/status drift")
    return module, identities


def _validate_retained_incremental_evidence(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay the 40 already-complete incremental records inside a full bundle.

    The incremental helper's public validator also asserts that no later Aim-3
    control namespace exists.  That assertion is correct for the unsealed
    225+4 candidate but intentionally becomes false after Phase 2.  The full
    verifier therefore reuses the helper's independently pinned component
    validators and omits only that lifecycle-specific absence assertion.
    """

    helper, identities = _load_trusted_phase1_candidate()
    by_id = helper._require_incremental_identity_roster(paths, list(sources))
    two_fixed, two_fixed_bindings = helper._validate_two_fixed(paths, by_id)
    pure, pure_bindings = helper._validate_adaptation_campaign(paths, by_id, family="pure-ridge")
    residual, residual_bindings = helper._validate_adaptation_campaign(
        paths, by_id, family="residual-ridge"
    )
    pure_native = pure.get("native_zero_shot", {}).get("scopes", {})
    residual_native = residual.get("native_zero_shot", {}).get("scopes", {})
    for scope in helper.ADAPTATION_SCOPES:
        for key in ("census", "auroc", "ci95"):
            if pure_native.get(scope, {}).get(key) != residual_native.get(scope, {}).get(key):
                raise BundleVerificationError(
                    "retained pure/residual frozen native comparator drift"
                )
    native_prefix = "aim2_target_internal.native."
    target_bindings = [
        *[binding for binding in pure_bindings if str(binding["id"]).startswith(native_prefix)],
        *[binding for binding in pure_bindings if not str(binding["id"]).startswith(native_prefix)],
        *[
            binding
            for binding in residual_bindings
            if not str(binding["id"]).startswith(native_prefix)
        ],
    ]
    if (
        len(target_bindings) != 348
        or len({str(binding["id"]) for binding in target_bindings}) != 348
    ):
        raise BundleVerificationError("retained target-internal binding roster drift")
    return {
        "two_fixed": two_fixed,
        "two_fixed_bindings": two_fixed_bindings,
        "pure_ridge": pure,
        "residual_ridge": residual,
        "target_internal_bindings": target_bindings,
        "generator_identities": identities,
    }


def _validate_aim2_conventional_primary_pool(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay the one patient-pooled UNI-v1 conventional-primary AUROC.

    This headline pools patients from CPTAC-primary and RIH-primary, then
    computes a single tie-aware AUROC over the five-refit mean native logit.
    It deliberately does not average the two cohort-specific AUROCs and does
    not manufacture a confidence interval that is absent from governed output.
    """

    import numpy as np
    import pandas as pd

    matches = [
        record for record in sources if record.get("id") == EXPECTED_AIM2_PRIMARY_POOL_SOURCE_ID
    ]
    if len(matches) != 1:
        raise BundleVerificationError(
            "Aim-2 conventional-primary patient-logit source is not unique"
        )
    record = matches[0]
    if (
        record.get("aims") != ["Aim 1", "Aim 2"]
        or record.get("experiments") != ["Aim 1 full pipeline and Aim 2 label-blind zero-shot"]
        or record.get("role") != EXPECTED_AIM2_PRIMARY_POOL_SOURCE_ROLE
        or record.get("path") != EXPECTED_AIM2_PRIMARY_POOL_SOURCE_PATH
        or record.get("size_bytes") != EXPECTED_AIM2_PRIMARY_POOL_SOURCE_SIZE
        or record.get("sha256") != EXPECTED_AIM2_PRIMARY_POOL_SOURCE_SHA256
    ):
        raise BundleVerificationError(
            "Aim-2 conventional-primary patient-logit source binding drift"
        )
    source_path = parent._lexical_source_path(str(record["path"]), paths)
    parent._reject_symlink_chain(source_path, context="Aim-2 conventional-primary patient logits")
    before = _identity(source_path, display_path=str(record["path"]))
    required_columns = (
        "analysis_family",
        "encoder",
        "dataset",
        "patient_id",
        "label",
        "cohort",
        "specimen_role",
        *EXPECTED_AIM2_PRIMARY_POOL_SEED_COLUMNS,
        "mean_logit_5seed",
    )
    try:
        frame = pd.read_parquet(source_path, columns=list(required_columns))
    except Exception as exc:
        raise BundleVerificationError(
            f"Aim-2 conventional-primary patient logits are unreadable: {exc}"
        ) from exc
    if _identity(source_path, display_path=str(record["path"])) != before:
        raise BundleVerificationError(
            "Aim-2 conventional-primary patient logits changed while read"
        )
    if tuple(frame.columns) != required_columns:
        raise BundleVerificationError("Aim-2 conventional-primary patient-logit columns drift")
    pooled = frame[
        (frame["analysis_family"] == EXPECTED_AIM2_PRIMARY_POOL_ANALYSIS_FAMILY)
        & (frame["encoder"] == EXPECTED_AIM2_PRIMARY_POOL_ENCODER)
        & frame["dataset"].isin(EXPECTED_AIM2_PRIMARY_POOL_DATASETS)
    ].copy()
    expected_patients, expected_mutant, expected_wild_type = EXPECTED_AIM2_PRIMARY_POOL_CENSUS
    labels = pooled["label"].to_numpy(dtype=int)
    if (
        len(pooled) != expected_patients
        or pooled["patient_id"].isna().any()
        or pooled["patient_id"].duplicated().any()
        or set(pooled["dataset"]) != set(EXPECTED_AIM2_PRIMARY_POOL_DATASETS)
        or set(pooled["specimen_role"]) != {"primary"}
        or set(np.unique(labels)) != {0, 1}
        or int(labels.sum()) != expected_mutant
        or int(len(labels) - labels.sum()) != expected_wild_type
    ):
        raise BundleVerificationError("Aim-2 conventional-primary patient-pooled census drift")
    expected_cohorts = {"cptac_primary": "CPTAC", "rih_primary": "RIH"}
    for dataset in EXPECTED_AIM2_PRIMARY_POOL_DATASETS:
        subset = pooled[pooled["dataset"] == dataset]
        patients, mutant, wild_type = EXPECTED_AIM2_PRIMARY_POOL_DATASET_CENSUS[dataset]
        if (
            len(subset) != patients
            or int(subset["label"].sum()) != mutant
            or patients - int(subset["label"].sum()) != wild_type
            or set(subset["cohort"]) != {expected_cohorts[dataset]}
        ):
            raise BundleVerificationError(
                f"Aim-2 conventional-primary dataset census drift: {dataset}"
            )
    seed_logits = pooled[list(EXPECTED_AIM2_PRIMARY_POOL_SEED_COLUMNS)].to_numpy(dtype=float)
    artifact_mean = pooled["mean_logit_5seed"].to_numpy(dtype=float)
    if (
        not np.isfinite(seed_logits).all()
        or not np.isfinite(artifact_mean).all()
        or not np.allclose(
            artifact_mean,
            seed_logits.mean(axis=1),
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise BundleVerificationError(
            "Aim-2 conventional-primary five-refit mean-native-logit drift"
        )
    mean_native_logit = seed_logits.mean(axis=1)
    ranks = pd.Series(mean_native_logit).rank(method="average").to_numpy(float)
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    auroc = float(
        (ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )
    if not math.isclose(
        auroc,
        EXPECTED_AIM2_PRIMARY_POOL_AUROC,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise BundleVerificationError("Aim-2 conventional-primary patient-pooled AUROC drift")
    bindings = [
        {
            "id": "aim2_source.conventional_primary_pool.patients",
            "value": expected_patients,
        },
        {
            "id": "aim2_source.conventional_primary_pool.mutant",
            "value": expected_mutant,
        },
        {
            "id": "aim2_source.conventional_primary_pool.wild_type",
            "value": expected_wild_type,
        },
        {
            "id": "aim2_source.conventional_primary_pool.auroc",
            "value": auroc,
        },
    ]
    return {
        "source_id": EXPECTED_AIM2_PRIMARY_POOL_SOURCE_ID,
        "encoder": "UNI-v1",
        "datasets": list(EXPECTED_AIM2_PRIMARY_POOL_DATASETS),
        "patients": expected_patients,
        "mutant": expected_mutant,
        "wild_type": expected_wild_type,
        "auroc": auroc,
        "aggregation": "patient_pooled_five_refit_mean_native_logit",
        "cohort_auroc_averaging": False,
        "governed_pooled_ci_available": False,
        "bindings": bindings,
    }


def _require_exact_installed_bytes(
    path: Path,
    expected: bytes,
    *,
    label: str,
) -> None:
    parent._reject_symlink_chain(path, context=label)
    if not path.is_file():
        raise BundleVerificationError(f"installed {label} is absent")
    before = _identity(path, display_path=_display(path, REPO))
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise BundleVerificationError(f"installed {label} is unreadable: {exc}") from exc
    after = _identity(path, display_path=_display(path, REPO))
    if before != after:
        raise BundleVerificationError(f"installed {label} changed while being read")
    if observed != expected:
        raise BundleVerificationError(
            f"installed {label} differs from trusted deterministic reconciliation"
        )


def _validate_trusted_full_reconciliation(paths: BundlePaths) -> dict[str, Any]:
    """Regenerate and byte-compare the production bundle before publication.

    Custom temporary BundlePaths used by unit tests still authenticate and bind
    the production generators, but their synthetic ledgers are validated by the
    ordinary semantic verifier rather than the production-only renderer.
    """

    production_destination = (FINAL_V13 / FINAL_RECEIPT_NAME).absolute()
    if paths.destination.absolute() == production_destination and not _is_production_bundle(paths):
        raise BundleVerificationError(
            "FINAL-v13 production receipt requires the exact trusted production paths"
        )
    if not _is_production_bundle(paths):
        return {
            "generator_identities": _trusted_reconciliation_generator_identities(),
            "exact_production_rendering": False,
        }
    reconciler, identities = _load_trusted_full_reconciler()
    try:
        payload = reconciler.build_full_payload(paths)
    except BundleVerificationError:
        raise
    except Exception as exc:
        raise BundleVerificationError(
            f"trusted FINAL-v13 full reconciliation failed: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "FULL_RECONCILIATION_RENDERED_READ_ONLY"
        or payload.get("scientific_source_count") != EXPECTED_FINAL_SOURCE_COUNT
        or payload.get("extension_source_count") != EXPECTED_EXTENSION_SOURCE_COUNT
        or payload.get("pending_source_count") != 0
        or payload.get("retained_target_internal_binding_count") != 348
        or payload.get("aim2_conventional_primary_pool_binding_count") != 4
    ):
        raise BundleVerificationError(
            "trusted FINAL-v13 full reconciler returned an invalid payload"
        )
    pins = payload.get("reconciliation_pins")
    manifest = payload.get("source_manifest")
    documents = payload.get("documents")
    if (
        not isinstance(pins, dict)
        or pins.get("extension_source_ids") != list(EXPECTED_EXTENSION_SOURCE_IDS)
        or not isinstance(manifest, dict)
        or not isinstance(documents, dict)
        or set(documents) != set(REPORT_DOCUMENTS)
    ):
        raise BundleVerificationError("trusted FINAL-v13 full reconciler payload roster drift")
    _require_exact_installed_bytes(
        paths.final_v13 / SOURCE_MANIFEST_NAME,
        _canonical_json_bytes(manifest),
        label="FINAL-v13 source manifest",
    )
    for name in REPORT_DOCUMENTS:
        text = documents[name]
        if not isinstance(text, str):
            raise BundleVerificationError(f"trusted FINAL-v13 reconciler returned non-text {name}")
        _require_exact_installed_bytes(
            paths.final_v13 / name,
            text.encode("utf-8"),
            label=f"FINAL-v13 {name}",
        )
    _require_exact_installed_bytes(
        paths.reconciliation_pins,
        _canonical_json_bytes(pins),
        label="FINAL-v13 reconciliation pins",
    )
    return {
        "generator_identities": identities,
        "exact_production_rendering": True,
    }


def _resolve_full_reconciliation_pins(
    paths: BundlePaths,
) -> tuple[BundlePaths, dict[str, Any], dict[str, Any]]:
    """Resolve full-status expectations from strict, non-scientific metadata.

    The final documents contain the scientific source index, including the
    verifier itself.  Keeping document hashes inside that verifier would form
    a verifier -> document -> verifier self-hash cycle.  The reconciliation
    pins therefore live in a separately receipt-bound metadata file which is
    expressly forbidden from the scientific source manifest.
    """

    pins, identity = _strict_stable_json(
        paths.reconciliation_pins,
        label="FINAL-v13 reconciliation pins",
        repo=paths.repo,
    )
    if set(pins) != _RECONCILIATION_PINS_KEYS:
        raise BundleVerificationError("FINAL-v13 reconciliation pin schema is not exact")
    parent_pins = pins.get("parent_final_v12_1")
    aim3_ids = pins.get("aim3_source_ids")
    documents = pins.get("documents")
    verifier_pin = pins.get("verifier")
    test_pin = pins.get("verifier_test")
    extension_ids = pins.get("extension_source_ids")
    if (
        pins.get("schema_version") != 1
        or pins.get("bundle") != "final_v13"
        or pins.get("status") != RECONCILIATION_PINS_STATUS
        or pins.get("scientific_source_manifest_membership") != RECONCILIATION_PINS_MEMBERSHIP
        or not isinstance(parent_pins, dict)
        or set(parent_pins) != _RECONCILIATION_PARENT_KEYS
        or parent_pins.get("status") != paths.expected_parent_status
        or parent_pins.get("receipt_sha256") != paths.expected_parent_receipt_sha256
        or parent_pins.get("source_manifest_sha256") != paths.expected_parent_manifest_sha256
        or not isinstance(extension_ids, list)
        or pins.get("extension_source_count") != EXPECTED_EXTENSION_SOURCE_COUNT
        or len(extension_ids) != EXPECTED_EXTENSION_SOURCE_COUNT
        or extension_ids != sorted(extension_ids)
        or len(extension_ids) != len(set(extension_ids))
        or any(
            not isinstance(source_id, str)
            or _SOURCE_ID_RE.fullmatch(source_id) is None
            or UNFROZEN in source_id
            for source_id in extension_ids
        )
        or not isinstance(aim3_ids, dict)
        or set(aim3_ids) != _RECONCILIATION_AIM3_KEYS
        or any(value not in extension_ids for value in aim3_ids.values())
        or not isinstance(documents, dict)
        or set(documents) != set(REPORT_DOCUMENTS)
        or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in documents.values()
        )
        or not isinstance(verifier_pin, dict)
        or set(verifier_pin) != _RECONCILIATION_TOOL_KEYS
        or not isinstance(test_pin, dict)
        or set(test_pin) != _RECONCILIATION_TOOL_KEYS
        or _SHA256_RE.fullmatch(str(verifier_pin.get("sha256"))) is None
        or _SHA256_RE.fullmatch(str(test_pin.get("sha256"))) is None
    ):
        raise BundleVerificationError("FINAL-v13 reconciliation pin identity is invalid")

    actual_verifier = _identity(
        paths.verifier_code,
        display_path=_display(paths.verifier_code, paths.repo),
    )
    actual_test = _identity(
        paths.verifier_test,
        display_path=_display(paths.verifier_test, paths.repo),
    )
    if verifier_pin["sha256"] != actual_verifier["sha256"]:
        raise BundleVerificationError("FINAL-v13 reconciliation verifier SHA-256 drift")
    if test_pin["sha256"] != actual_test["sha256"]:
        raise BundleVerificationError("FINAL-v13 reconciliation verifier-test SHA-256 drift")

    configured_extensions = tuple(paths.expected_extension_source_ids)
    if configured_extensions != tuple(extension_ids):
        raise BundleVerificationError(
            "FINAL-v13 reconciliation metadata cannot choose extension membership"
        )
    configured_documents = dict(paths.expected_final_document_sha256)
    if set(configured_documents.values()) != {UNFROZEN} and configured_documents != documents:
        raise BundleVerificationError("FINAL-v13 configured document pins disagree with metadata")
    configured_aim3 = {
        "results": paths.aim3_results_source_id,
        "scheduler": paths.aim3_scheduler_source_id,
        "training_completion": paths.aim3_training_source_id,
        "fine_results": paths.aim3_fine_results_source_id,
        "fine_scheduler": paths.aim3_fine_scheduler_source_id,
        "fine_training_completion": paths.aim3_fine_training_source_id,
        "controls_scheduler": paths.aim3_controls_scheduler_source_id,
        "fine_external_results": paths.aim3_fine_external_results_source_id,
        "fine_external_analysis_completion": (paths.aim3_fine_external_completion_source_id),
        "fine_external_inference_seal": (paths.aim3_fine_external_inference_seal_source_id),
        "fine_external_scoring_completion": (paths.aim3_fine_external_scoring_source_id),
    }
    if configured_aim3 != aim3_ids:
        raise BundleVerificationError(
            "FINAL-v13 reconciliation metadata cannot choose Aim-3 semantic IDs"
        )

    resolved = replace(
        paths,
        expected_final_document_sha256=dict(documents),
        expected_extension_source_ids=tuple(extension_ids),
        aim3_results_source_id=str(aim3_ids["results"]),
        aim3_scheduler_source_id=str(aim3_ids["scheduler"]),
        aim3_training_source_id=str(aim3_ids["training_completion"]),
        aim3_fine_results_source_id=str(aim3_ids["fine_results"]),
        aim3_fine_scheduler_source_id=str(aim3_ids["fine_scheduler"]),
        aim3_fine_training_source_id=str(aim3_ids["fine_training_completion"]),
        aim3_controls_scheduler_source_id=str(aim3_ids["controls_scheduler"]),
        aim3_fine_external_results_source_id=str(aim3_ids["fine_external_results"]),
        aim3_fine_external_completion_source_id=str(aim3_ids["fine_external_analysis_completion"]),
        aim3_fine_external_inference_seal_source_id=str(aim3_ids["fine_external_inference_seal"]),
        aim3_fine_external_scoring_source_id=str(aim3_ids["fine_external_scoring_completion"]),
    )
    return resolved, pins, identity


def _require_frozen_expectations(paths: BundlePaths, *, phase1_candidate: bool = False) -> None:
    document_pins = dict(paths.expected_final_document_sha256)
    document_values = tuple(document_pins.values())
    documents_unfrozen = set(document_pins) == set(REPORT_DOCUMENTS) and set(document_values) == {
        UNFROZEN
    }
    documents_frozen = set(document_pins) == set(REPORT_DOCUMENTS) and all(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
        for value in document_values
    )
    if not documents_frozen and not (phase1_candidate and documents_unfrozen):
        raise BundleVerificationError(
            "FINAL-v13 document hash pins are explicitly UNFROZEN; reconcile before sealing"
        )

    extension_ids = tuple(paths.expected_extension_source_ids)
    required_ids = (
        paths.aim3_results_source_id,
        paths.aim3_scheduler_source_id,
        paths.aim3_training_source_id,
        paths.aim3_fine_results_source_id,
        paths.aim3_fine_scheduler_source_id,
        paths.aim3_fine_training_source_id,
        paths.aim3_controls_scheduler_source_id,
        paths.aim3_fine_external_results_source_id,
        paths.aim3_fine_external_completion_source_id,
        paths.aim3_fine_external_inference_seal_source_id,
        paths.aim3_fine_external_scoring_source_id,
    )
    if (
        not extension_ids
        or any(
            not isinstance(source_id, str)
            or _SOURCE_ID_RE.fullmatch(source_id) is None
            or UNFROZEN in source_id
            for source_id in extension_ids
        )
        or len(extension_ids) != len(set(extension_ids))
        or extension_ids != tuple(sorted(extension_ids))
        or any(source_id not in extension_ids for source_id in required_ids)
    ):
        raise BundleVerificationError(
            "FINAL-v13 extension source IDs are explicitly UNFROZEN or incomplete"
        )


def _validate_parent(
    paths: BundlePaths,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Authenticate and recursively replay sealed FINAL-v12.1."""

    receipt, receipt_identity = _strict_stable_json(
        paths.parent_receipt, label="sealed FINAL-v12.1 receipt", repo=paths.repo
    )
    if receipt_identity["sha256"] != paths.expected_parent_receipt_sha256:
        raise BundleVerificationError("sealed FINAL-v12.1 receipt SHA-256 drift")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("bundle") != "reports/final_v12_1"
        or receipt.get("status") != paths.expected_parent_status
        or receipt.get("source_count") != paths.expected_parent_source_count
    ):
        raise BundleVerificationError("sealed FINAL-v12.1 receipt identity is invalid")
    checks = receipt.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(value != "PASS" for value in checks.values())
    ):
        raise BundleVerificationError("sealed FINAL-v12.1 receipt lacks all-PASS checks")

    if paths.replay_parent_verifier:
        verifier_identity = _identity(
            paths.parent_verifier,
            display_path=_display(paths.parent_verifier, paths.repo),
        )
        test_identity = _identity(
            paths.parent_test,
            display_path=_display(paths.parent_test, paths.repo),
        )
        if verifier_identity["sha256"] != paths.expected_parent_verifier_sha256:
            raise BundleVerificationError("frozen FINAL-v12.1 verifier SHA-256 drift")
        if test_identity["sha256"] != paths.expected_parent_test_sha256:
            raise BundleVerificationError("frozen FINAL-v12.1 verifier test SHA-256 drift")
        replayed = parent.verify_published_receipt(parent.default_paths())
        if replayed != receipt:
            raise BundleVerificationError("recursive FINAL-v12.1 receipt replay differs")
    else:
        verifier_identity = {
            "path": _display(paths.parent_verifier, paths.repo),
            "size_bytes": 0,
            "sha256": paths.expected_parent_verifier_sha256,
        }
        test_identity = {
            "path": _display(paths.parent_test, paths.repo),
            "size_bytes": 0,
            "sha256": paths.expected_parent_test_sha256,
        }

    manifest, manifest_identity = _strict_stable_json(
        paths.parent_manifest,
        label="sealed FINAL-v12.1 source manifest",
        repo=paths.repo,
    )
    if manifest_identity["sha256"] != paths.expected_parent_manifest_sha256:
        raise BundleVerificationError("sealed FINAL-v12.1 manifest SHA-256 drift")
    parent._require_exact_identity(
        receipt.get("source_manifest"),
        paths.parent_manifest,
        display_path=_display(paths.parent_manifest, paths.repo),
        context="sealed FINAL-v12.1 receipt source_manifest",
    )
    if set(manifest) != _MANIFEST_KEYS:
        raise BundleVerificationError("sealed FINAL-v12.1 source manifest schema is not exact")
    if (
        manifest["schema_version"] != 2
        or manifest["bundle"] != "final_v12_1"
        or manifest["status"] != paths.expected_parent_manifest_status
        or manifest["pending_artifacts"] != []
    ):
        raise BundleVerificationError("sealed FINAL-v12.1 source manifest identity is invalid")
    sources = parent._validate_source_roster(
        manifest["artifacts"],
        paths,
        expected_count=paths.expected_parent_source_count,
        rehash=False,
    )
    if receipt.get("authoritative_sources") != sources:
        raise BundleVerificationError("FINAL-v12.1 receipt/manifest source rosters differ")

    parent_pins = dict(paths.expected_parent_document_sha256)
    receipt_documents = receipt.get("documents")
    if set(parent_pins) != set(REPORT_DOCUMENTS) or not isinstance(receipt_documents, dict):
        raise BundleVerificationError("sealed FINAL-v12.1 document pin roster is invalid")
    if set(receipt_documents) != set(REPORT_DOCUMENTS):
        raise BundleVerificationError("sealed FINAL-v12.1 receipt document roster is invalid")
    documents: dict[str, Any] = {}
    for name in REPORT_DOCUMENTS:
        document_path = paths.parent_dir / name
        actual = parent._require_exact_identity(
            receipt_documents[name],
            document_path,
            display_path=_display(document_path, paths.repo),
            context=f"sealed FINAL-v12.1 {name}",
        )
        if actual["sha256"] != parent_pins[name]:
            raise BundleVerificationError(f"sealed FINAL-v12.1 {name} SHA-256 drift")
        documents[name] = actual
    return (
        receipt,
        manifest,
        sources,
        {
            "receipt": receipt_identity,
            "source_manifest": manifest_identity,
            "documents": documents,
            "verifier": verifier_identity,
            "verifier_test": test_identity,
        },
    )


def _validate_final_manifest(
    paths: BundlePaths,
    parent_sources: Sequence[Mapping[str, Any]],
    *,
    phase1_candidate: bool = False,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    manifest_path = paths.final_v13 / SOURCE_MANIFEST_NAME
    manifest, manifest_identity = _strict_stable_json(
        manifest_path, label="FINAL-v13 source manifest", repo=paths.repo
    )
    if set(manifest) != _MANIFEST_KEYS:
        raise BundleVerificationError("FINAL-v13 source manifest schema is not exact")
    expected_status = (
        paths.expected_phase1_manifest_status
        if phase1_candidate
        else paths.expected_final_manifest_status
    )
    if (
        manifest["schema_version"] != 2
        or manifest["bundle"] != "final_v13"
        or manifest["status"] != expected_status
    ):
        raise BundleVerificationError("FINAL-v13 source manifest identity is invalid")

    extension_ids = tuple(paths.expected_extension_source_ids)
    pending_raw = manifest["pending_artifacts"]
    if not isinstance(pending_raw, list):
        raise BundleVerificationError("FINAL-v13 pending source roster must be an array")
    pending: list[dict[str, Any]] = []
    for index, record in enumerate(pending_raw):
        if not isinstance(record, dict) or set(record) != _PENDING_SOURCE_KEYS:
            raise BundleVerificationError(f"FINAL-v13 pending source {index} schema is not exact")
        metadata = {**record, "size_bytes": 0, "sha256": "0" * 64}
        parent._validate_source_metadata(metadata, index=index)
        pending.append(record)
    pending_ids = [str(record["id"]) for record in pending]
    if pending_ids != sorted(pending_ids) or len(pending_ids) != len(set(pending_ids)):
        raise BundleVerificationError("FINAL-v13 pending source roster is not sorted and unique")
    if phase1_candidate:
        required_pending = {
            paths.aim3_results_source_id,
            paths.aim3_scheduler_source_id,
            paths.aim3_training_source_id,
            paths.aim3_controls_scheduler_source_id,
        }
        if set(pending_ids) != required_pending:
            raise BundleVerificationError(
                "FINAL-v13 Phase-1 candidate pending roster is not exactly the four "
                "full controls/results sources"
            )
    elif pending:
        raise BundleVerificationError("FINAL-v13 seal requires zero pending artifacts")

    expected_count = paths.expected_parent_source_count + len(extension_ids) - len(pending)
    sources = parent._validate_source_roster(
        manifest["artifacts"], paths, expected_count=expected_count, rehash=True
    )
    parent_by_id = {str(record["id"]): record for record in parent_sources}
    source_by_id = {str(record["id"]): record for record in sources}
    if set(parent_by_id) & set(extension_ids):
        raise BundleVerificationError("FINAL-v13 extension source ID collides with parent")
    if set(source_by_id) & set(pending_ids):
        raise BundleVerificationError("FINAL-v13 source is both material and pending")
    if set(source_by_id) | set(pending_ids) != set(parent_by_id) | set(extension_ids):
        raise BundleVerificationError(
            "FINAL-v13 material-plus-pending roster is not parent plus exact extension"
        )
    for source_id, inherited in parent_by_id.items():
        if source_by_id[source_id] != inherited:
            raise BundleVerificationError(f"FINAL-v13 inherited source record drift: {source_id}")
    for record in sources:
        if str(record["path"]).startswith(EXCLUDED_FAILED_AIM3_V1_ROOT):
            raise BundleVerificationError(
                "failed pre-fit Aim-3 v1 root entered authoritative science sources"
            )
    material_paths = {str(record["path"]) for record in sources}
    pending_paths = [str(record["path"]) for record in pending]
    if len(pending_paths) != len(set(pending_paths)) or material_paths.intersection(pending_paths):
        raise BundleVerificationError("FINAL-v13 material/pending source path collision")
    if phase1_candidate:
        for record in pending:
            path = parent._lexical_source_path(str(record["path"]), paths)
            if path.exists() or path.is_symlink():
                raise BundleVerificationError(
                    f"FINAL-v13 pending Phase-2 source already exists: {record['id']}"
                )
    return manifest, sources, pending, manifest_identity


def _source_json(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
    source_id: str,
    *,
    role: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    matches = [record for record in sources if record["id"] == source_id]
    if len(matches) != 1:
        raise BundleVerificationError(f"required FINAL-v13 source is not unique: {source_id}")
    record = matches[0]
    if record["role"] != role or "Aim 3" not in record["aims"]:
        raise BundleVerificationError(f"Aim-3 source metadata boundary is invalid: {source_id}")
    path = parent._lexical_source_path(str(record["path"]), paths)
    value, _ = _strict_stable_json(path, label=f"Aim-3 source {source_id}", repo=paths.repo)
    return value, record


def _require_aim3_source_path(
    paths: BundlePaths,
    record: Mapping[str, Any],
    relative: str,
) -> Path:
    """Bind an Aim-3 source record to its exact governed campaign location."""

    observed = parent._lexical_source_path(str(record["path"]), paths).absolute()
    expected = (Path(paths.expected_aim3_run_root) / relative).absolute()
    if observed != expected:
        raise BundleVerificationError(
            f"Aim-3 governed source path drift: expected {expected}, observed {observed}"
        )
    return observed


def _require_aim3_fine_external_source_path(
    paths: BundlePaths,
    record: Mapping[str, Any],
    relative: str,
) -> Path:
    """Bind external evidence to the separate, score-only campaign root."""

    observed = parent._lexical_source_path(str(record["path"]), paths).absolute()
    expected = (Path(paths.expected_aim3_fine_external_run_root) / relative).absolute()
    if observed != expected:
        raise BundleVerificationError(
            "Aim-3 fine-external governed source path drift: "
            f"expected {expected}, observed {observed}"
        )
    return observed


def _source_record(
    sources: Sequence[Mapping[str, Any]],
    source_id: str,
    *,
    role: str,
) -> Mapping[str, Any]:
    matches = [record for record in sources if record["id"] == source_id]
    if len(matches) != 1:
        raise BundleVerificationError(f"required FINAL-v13 source is not unique: {source_id}")
    record = matches[0]
    if record["role"] != role or "Aim 3" not in record["aims"]:
        raise BundleVerificationError(f"Aim-3 source metadata boundary is invalid: {source_id}")
    return record


def _require_artifact_identity(
    value: Any,
    path: Path,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "size_bytes", "sha256"}:
        raise BundleVerificationError(f"{context} artifact identity schema is invalid")
    expected = _identity(path, display_path=str(value["path"]))
    if value != expected:
        raise BundleVerificationError(f"{context} artifact identity drift")
    return expected


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleVerificationError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise BundleVerificationError(f"{context} must be finite")
    return parsed


def _seed_map(value: Any, *, context: str) -> None:
    if not isinstance(value, dict) or tuple(value) != tuple(str(seed) for seed in MODEL_SEEDS):
        raise BundleVerificationError(f"{context} must use exact seed roster 42..46")
    for seed, metric in value.items():
        _finite(metric, context=f"{context}/{seed}")


def _resolve_pointer(document: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise BundleVerificationError("Aim-3 report binding has an invalid JSON pointer")
    current = document
    for raw in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw):
            raise BundleVerificationError("Aim-3 report binding has invalid pointer escaping")
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise BundleVerificationError(f"Aim-3 report binding pointer is absent: {pointer}")
    return current


def _canonical_number(value: Any) -> str:
    _finite(value, context="Aim-3 report-bound value")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _validate_aim3_fine_results(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the source-bound fine-only Phase-1 candidate result."""

    result, record = _source_json(
        paths,
        sources,
        paths.aim3_fine_results_source_id,
        role="aim3_fine_results",
    )
    _require_aim3_source_path(paths, record, "analysis/fine_results.json")
    if (
        result.get("schema_version") != 1
        or result.get("status") != "fine_phase_complete_candidate_unsealed"
        or result.get("campaign") != paths.expected_aim3_experiment
        or result.get("phase") != "fine"
        or result.get("population") != "tcga_surgen_primary"
        or result.get("population_display") != "TCGA + SurGen primaries"
        or result.get("encoder") != "UNI-v1"
        or result.get("model_seeds") != list(MODEL_SEEDS)
        or result.get("wt_draw_seeds") != list(WT_DRAW_SEEDS)
        or result.get("controls_status") != "not_started"
        or result.get("external_development_cohorts") != []
        or result.get("aggregation") != "patient native-logit five-seed ensemble"
        or result.get("fit_accounting") != EXPECTED_FINE_TRAINING_COUNTS
        or result.get("v13_status") != "candidate_unsealed_pending_fixed_and_repeated_controls"
    ):
        raise BundleVerificationError("Aim-3 fine-only candidate result identity is invalid")
    fine = result.get("fine")
    rungs = fine.get("rungs") if isinstance(fine, dict) else None
    if not isinstance(rungs, dict) or set(rungs) != set(AIM3_RUNG_CONTROL):
        raise BundleVerificationError("Aim-3 fine-only candidate rung roster is not exact")
    expected_bindings: list[dict[str, Any]] = []
    for rung in AIM3_RUNG_CONTROL:
        rung_record = rungs[rung]
        if not isinstance(rung_record, dict):
            raise BundleVerificationError(f"Aim-3 fine-only rung is invalid: {rung}")
        _seed_map(rung_record.get("per_seed_auroc"), context=f"fine/{rung}")
        summary = rung_record.get("five_seed_ensemble")
        estimate = summary.get("estimate") if isinstance(summary, dict) else None
        _finite(estimate, context=f"fine/{rung}/five_seed_ensemble")
        for field in ("patients", "positive", "negative"):
            value = rung_record.get(field)
            if type(value) is not int or value <= 0:
                raise BundleVerificationError(f"Aim-3 fine-only {rung}/{field} is invalid")
        if rung_record["positive"] + rung_record["negative"] != rung_record["patients"]:
            raise BundleVerificationError(f"Aim-3 fine-only patient census does not add: {rung}")
        expected_bindings.append(
            {
                "id": f"aim3_source.fine.{rung}.auroc",
                "section": "fine",
                "rung": rung,
                "draw_seed": None,
                "metric": "fine_auroc",
                "value": estimate,
            }
        )
    expected_bindings.sort(key=lambda item: str(item["id"]))
    if result.get("report_bindings") != expected_bindings:
        raise BundleVerificationError(
            "Aim-3 fine-only report bindings differ from primitive source estimates"
        )
    return {"fine_results": dict(record)}, expected_bindings


def _validate_fine_execution_evidence(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scheduler, scheduler_record = _source_json(
        paths,
        sources,
        paths.aim3_fine_scheduler_source_id,
        role="fine_scheduler_evidence",
    )
    _require_aim3_source_path(paths, scheduler_record, "receipts/scheduler_fine.json")
    expected_job_keys = sorted(
        f"fine__{rung}__seed{seed}" for rung in AIM3_RUNG_CONTROL for seed in MODEL_SEEDS
    )
    events = scheduler.get("events")
    if (
        scheduler.get("schema_version") != 1
        or scheduler.get("status") != "fine_phase_completed_rc0"
        or scheduler.get("phase") != "fine"
        or scheduler.get("configured_max_parallel_chains") != EXPECTED_PARALLELISM
        or scheduler.get("observed_max_parallel_chains") != EXPECTED_PARALLELISM
        or scheduler.get("job_count") != EXPECTED_FINE_TRAINING_COUNTS["chains"]
        or scheduler.get("fit_accounting") != EXPECTED_FINE_TRAINING_COUNTS
        or scheduler.get("job_keys") != expected_job_keys
        or scheduler.get("control_artifacts_absent") is not True
        or not isinstance(events, list)
        or len(events) != EXPECTED_FINE_TRAINING_COUNTS["chains"]
        or any(not isinstance(event, dict) or event.get("returncode") != 0 for event in events)
    ):
        raise BundleVerificationError(
            "Aim-3 fine scheduler must prove 25 chains/150 fits at configured and observed six-parallel"
        )

    training, training_record = _source_json(
        paths,
        sources,
        paths.aim3_fine_training_source_id,
        role="fine_training_completion",
    )
    _require_aim3_source_path(paths, training_record, "receipts/training_complete_fine.json")
    scheduler_path = parent._lexical_source_path(str(scheduler_record["path"]), paths)
    expected_scheduler = _identity(scheduler_path, display_path=str(scheduler_record["path"]))
    if (
        training.get("schema_version") != 1
        or training.get("status") != "fine_phase_complete_and_certified"
        or training.get("campaign") != paths.expected_aim3_experiment
        or training.get("phase") != "fine"
        or training.get("population") != "tcga_surgen_primary"
        or training.get("encoder") != "UNI-v1"
        or training.get("model_seeds") != list(MODEL_SEEDS)
        or training.get("controls_status") != "not_started"
        or training.get("fit_accounting") != EXPECTED_FINE_TRAINING_COUNTS
        or training.get("scheduler") != expected_scheduler
        or training.get("control_artifacts_absent") is not True
        or training.get("external_development_cohorts") != []
        or not isinstance(training.get("job_receipts"), list)
        or len(training["job_receipts"]) != EXPECTED_FINE_TRAINING_COUNTS["chains"]
    ):
        raise BundleVerificationError("Aim-3 fine terminal training evidence is invalid")
    return {
        "fine_scheduler": dict(scheduler_record),
        "fine_training_completion": dict(training_record),
        "configured_parallel_chains": EXPECTED_PARALLELISM,
        "observed_peak_parallel_chains": EXPECTED_PARALLELISM,
        "training_counts": dict(EXPECTED_FINE_TRAINING_COUNTS),
    }


def _mean_sample_sd(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _metric01(value: Any, *, context: str) -> float:
    parsed = _finite(value, context=context)
    if parsed < 0.0 or parsed > 1.0:
        raise BundleVerificationError(f"{context} must lie in [0, 1]")
    return parsed


def _validate_external_metric_block(
    block: Any,
    *,
    context: str,
    expected_census: tuple[int, int] | None = None,
    seed_key: str = "per_seed_auroc",
) -> dict[str, Any]:
    if not isinstance(block, dict) or block.get("status") != "ESTIMABLE":
        raise BundleVerificationError(f"{context} must be an estimable metric block")
    positive = block.get("positive")
    negative = block.get("negative")
    n_records = block.get("n_records")
    if (
        type(positive) is not int
        or type(negative) is not int
        or type(n_records) is not int
        or positive <= 0
        or negative <= 0
        or n_records != positive + negative
        or (expected_census is not None and (positive, negative) != expected_census)
    ):
        raise BundleVerificationError(f"{context} patient census drift")
    per_seed = block.get(seed_key)
    _seed_map(per_seed, context=f"{context}/{seed_key}")
    values = [float(per_seed[str(seed)]) for seed in MODEL_SEEDS]
    expected_mean, expected_sd = _mean_sample_sd(values)
    observed_mean = _metric01(block.get("seed_auroc_mean"), context=f"{context}/seed_mean")
    observed_sd = _finite(block.get("seed_auroc_sample_sd"), context=f"{context}/seed_sd")
    if observed_sd < 0 or not math.isclose(observed_mean, expected_mean, abs_tol=1e-15):
        raise BundleVerificationError(f"{context} seed mean does not replay")
    if not math.isclose(observed_sd, expected_sd, abs_tol=1e-15):
        raise BundleVerificationError(f"{context} sample SD does not replay")
    ensemble = block.get("five_seed_refit_ensemble")
    if not isinstance(ensemble, dict):
        raise BundleVerificationError(f"{context} five-seed ensemble is absent")
    auroc = _metric01(ensemble.get("auroc"), context=f"{context}/ensemble_auroc")
    ci95 = ensemble.get("ci95")
    if not isinstance(ci95, list) or len(ci95) != 2:
        raise BundleVerificationError(f"{context} pointwise CI is invalid")
    low = _metric01(ci95[0], context=f"{context}/ci95_low")
    high = _metric01(ci95[1], context=f"{context}/ci95_high")
    if low > high:
        raise BundleVerificationError(f"{context} pointwise CI is reversed")
    return {
        "n": n_records,
        "positive": positive,
        "negative": negative,
        "seed_mean": observed_mean,
        "seed_sd": observed_sd,
        "auroc": auroc,
        "ci95": [low, high],
    }


def _fine_external_report_bindings(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind every displayed fine-external table cell to its source primitive."""

    bindings: list[dict[str, Any]] = []

    def add(identifier: str, value: Any) -> None:
        _finite(value, context=f"fine-external binding/{identifier}")
        bindings.append({"id": identifier, "value": value})

    tasks = result["tasks"]
    for task in AIM3_FINE_TASKS:
        internal = tasks[task]["internal_oof"]
        internal_values = {
            "n": internal["patients"],
            "positive": internal["positive"],
            "negative": internal["negative"],
            "seed_mean": internal["seed_auroc_mean"],
            "seed_sd": internal["seed_auroc_sample_sd"],
            "fold_mean": internal["fold_auroc_mean"],
            "fold_sd": internal["fold_auroc_sample_sd"],
            "auroc": internal["five_seed_oof_ensemble"]["auroc"],
            "ci95_low": internal["five_seed_oof_ensemble"]["ci95"][0],
            "ci95_high": internal["five_seed_oof_ensemble"]["ci95"][1],
        }
        for metric, value in internal_values.items():
            add(f"aim3_source.fine_external.internal.{task}.{metric}", value)
        external = tasks[task]["external_refit"]
        for scope in AIM3_FINE_EXTERNAL_REPORT_SCOPES:
            block = (
                external["per_cohort"][scope]
                if scope in AIM3_FINE_EXTERNAL_TARGETS
                else external[scope]
            )
            ensemble = block["five_seed_refit_ensemble"]
            values = {
                "n": block["n_records"],
                "positive": block["positive"],
                "negative": block["negative"],
                "seed_mean": block["seed_auroc_mean"],
                "seed_sd": block["seed_auroc_sample_sd"],
                "auroc": ensemble["auroc"],
                "ci95_low": ensemble["ci95"][0],
                "ci95_high": ensemble["ci95"][1],
            }
            for metric, value in values.items():
                add(
                    f"aim3_source.fine_external.external.{task}.{scope}.{metric}",
                    value,
                )
    if len(bindings) != 370 or len({binding["id"] for binding in bindings}) != 370:
        raise BundleVerificationError("fine-external numeric binding roster drift")
    return bindings


def _validate_aim3_fine_external_results(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate frozen-refit inference, outcome ordering, and all report rows."""

    contract, contract_record = _source_json(
        paths,
        sources,
        EXPECTED_AIM3_FINE_EXTERNAL_CONTRACT_SOURCE_ID,
        role="fine_external_campaign_contract",
    )
    contract_path = _require_aim3_fine_external_source_path(paths, contract_record, "contract.json")
    preflight, preflight_record = _source_json(
        paths,
        sources,
        EXPECTED_AIM3_FINE_EXTERNAL_PREFLIGHT_SOURCE_ID,
        role="fine_external_deep_preflight",
    )
    preflight_path = _require_aim3_fine_external_source_path(
        paths, preflight_record, "receipts/deep_preflight.json"
    )
    scoring, scoring_record = _source_json(
        paths,
        sources,
        paths.aim3_fine_external_scoring_source_id,
        role="fine_external_scoring_completion",
    )
    _require_aim3_fine_external_source_path(paths, scoring_record, "receipts/scoring_complete.json")
    inference, inference_record = _source_json(
        paths,
        sources,
        paths.aim3_fine_external_inference_seal_source_id,
        role="fine_external_inference_seal",
    )
    inference_path = _require_aim3_fine_external_source_path(
        paths, inference_record, "inference/inference_seal.json"
    )
    result, result_record = _source_json(
        paths,
        sources,
        paths.aim3_fine_external_results_source_id,
        role="fine_external_results",
    )
    results_path = _require_aim3_fine_external_source_path(
        paths, result_record, "analysis/results.json"
    )
    completion, completion_record = _source_json(
        paths,
        sources,
        paths.aim3_fine_external_completion_source_id,
        role="fine_external_analysis_completion",
    )
    _require_aim3_fine_external_source_path(
        paths, completion_record, "analysis/analysis_completion.json"
    )

    controller_record = _source_record(
        sources,
        EXPECTED_AIM3_FINE_EXTERNAL_CONTROLLER_SOURCE_ID,
        role="fine_external_campaign_controller",
    )
    test_record = _source_record(
        sources,
        EXPECTED_AIM3_FINE_EXTERNAL_TEST_SOURCE_ID,
        role="fine_external_campaign_contract_test",
    )
    implementation = contract.get("implementation")
    governance = contract.get("governance")
    score_contract = contract.get("score_contract")
    overlaps = contract.get("source_target_overlap")
    if (
        contract.get("schema_version") != 1
        or contract.get("campaign") != paths.expected_aim3_fine_external_experiment
        or contract.get("status") != "prepared_label_blind_before_outcome_join"
        or contract.get("output_root") != paths.expected_aim3_fine_external_run_root
        or contract.get("training_root") != paths.expected_aim3_run_root
        or contract.get("model_seeds") != list(MODEL_SEEDS)
        or not isinstance(implementation, dict)
        or not isinstance(governance, dict)
        or not isinstance(score_contract, dict)
        or not isinstance(overlaps, dict)
        or set(overlaps) != set(AIM3_FINE_EXTERNAL_TARGETS)
        or any(value != {"patients": 0, "slides": 0} for value in overlaps.values())
        or score_contract.get("jobs") != 25
        or score_contract.get("score_artifacts") != 25
        or score_contract.get("score_rows") != 11_975
        or score_contract.get("new_fits") != 0
        or score_contract.get("maximum_parallel_workers") != EXPECTED_PARALLELISM
        or governance.get("external_test_data")
        != [
            "CPTAC-primary",
            "Orion-primary",
            "RIH-primary",
            "RIH-metastatic",
            "SurGen-metastatic",
        ]
        or governance.get("phase_order")
        != [
            "fine_training_and_analysis",
            "external_score_and_analysis",
            "matched_and_repeated_controls",
        ]
        or governance.get("inference_seal_exactly_once") is not True
        or governance.get("analysis_requires_inference_seal") is not True
        or any(
            governance.get(key) is not False
            for key in (
                "prepare_reads_target_outcomes",
                "preflight_reads_target_outcomes",
                "score_reads_target_outcomes",
                "target_adaptation",
                "target_calibration",
                "target_encoder_or_construction_selection",
                "target_fitting",
                "target_model_or_checkpoint_selection",
                "target_refitting",
                "target_threshold_selection",
            )
        )
    ):
        raise BundleVerificationError("Aim-3 fine-external campaign contract is invalid")
    _require_artifact_identity(
        implementation.get("controller"),
        parent._lexical_source_path(str(controller_record["path"]), paths),
        context="Aim-3 fine-external controller",
    )
    _require_artifact_identity(
        implementation.get("controller_test"),
        parent._lexical_source_path(str(test_record["path"]), paths),
        context="Aim-3 fine-external controller test",
    )

    preflight_checks = preflight.get("checks")
    if (
        preflight.get("schema_version") != 1
        or preflight.get("status") != "ready_for_label_blind_inference"
        or preflight.get("maximum_parallel_workers") != EXPECTED_PARALLELISM
        or preflight.get("score_jobs") != 25
        or preflight.get("score_rows") != 11_975
        or preflight.get("authenticated_p75_checkpoints") != 25
        or preflight.get("target_outcomes_opened") is not False
        or preflight.get("target_outcomes_present") is not False
        or preflight.get("control_artifacts_absent") is not True
        or not isinstance(preflight_checks, dict)
        or not preflight_checks
        or any(value is not True for value in preflight_checks.values())
    ):
        raise BundleVerificationError("Aim-3 fine-external preflight is invalid")
    _require_artifact_identity(
        preflight.get("contract"), contract_path, context="fine-external preflight contract"
    )

    events = scoring.get("inference_events")
    expected_jobs = {(task, seed) for task in AIM3_FINE_TASKS for seed in MODEL_SEEDS}
    if (
        scoring.get("schema_version") != 1
        or scoring.get("status") != "complete_before_outcome_join"
        or scoring.get("configured_max_workers") != EXPECTED_PARALLELISM
        or scoring.get("max_observed_parallel_inference_workers") != EXPECTED_PARALLELISM
        or scoring.get("score_jobs") != 25
        or scoring.get("score_rows") != 11_975
        or scoring.get("control_artifacts_absent") is not True
        or not isinstance(events, list)
        or len(events) != 25
        or {(event.get("task"), event.get("seed")) for event in events if isinstance(event, dict)}
        != expected_jobs
        or any(
            event.get("returncode") != 0
            or event.get("score_rows") != 479
            or event.get("execution_role") != "label_blind_native_logit_inference"
            or type(event.get("started_unix_ns")) is not int
            or type(event.get("completed_unix_ns")) is not int
            or event["completed_unix_ns"] <= event["started_unix_ns"]
            for event in events
        )
    ):
        raise BundleVerificationError(
            "Aim-3 fine-external scoring must prove 25 jobs and observed six-parallel"
        )
    _require_artifact_identity(
        scoring.get("contract"), contract_path, context="fine-external scoring contract"
    )

    score_artifacts = inference.get("score_artifacts")
    if (
        inference.get("schema_version") != 1
        or inference.get("status") != "sealed_before_outcome_join"
        or inference.get("control_artifacts_absent_at_inference_seal") is not True
        or inference.get("target_outcomes_opened") is not False
        or inference.get("score_artifact_count") != 25
        or inference.get("score_rows") != 11_975
        or not isinstance(score_artifacts, list)
        or len(score_artifacts) != 25
        or {(item.get("task"), item.get("seed")) for item in score_artifacts} != expected_jobs
        or any(item.get("rows") != 479 for item in score_artifacts)
    ):
        raise BundleVerificationError("Aim-3 fine-external inference seal is invalid")
    _require_artifact_identity(
        inference.get("contract"), contract_path, context="fine-external seal contract"
    )
    _require_artifact_identity(
        inference.get("preflight"), preflight_path, context="fine-external seal preflight"
    )

    if (
        result.get("schema_version") != 1
        or result.get("campaign") != paths.expected_aim3_fine_external_experiment
        or result.get("status") != "complete_after_sealed_zero_shot_inference"
        or result.get("encoder") != "UNI-v1"
        or result.get("model_seeds") != list(MODEL_SEEDS)
        or set(result.get("tasks", {})) != set(AIM3_FINE_TASKS)
        or not isinstance(result.get("report_rows"), list)
        or len(result["report_rows"]) != 55
    ):
        raise BundleVerificationError("Aim-3 fine-external result identity is invalid")
    claim = result.get("claim_boundary")
    if (
        not isinstance(claim, dict)
        or claim.get("external_target_use") != "score-only after model freeze"
        or claim.get("new_fits") != 0
        or claim.get("target_calibration") is not False
        or claim.get("target_threshold_selection") is not False
        or claim.get("control_artifacts_absent_through_outcome_join") is not True
        or claim.get("controls_begin_only_after_external_analysis") is not True
        or claim.get("multiplicity_claim") is not False
    ):
        raise BundleVerificationError("Aim-3 fine-external claim boundary is invalid")

    fine_result, _ = _source_json(
        paths,
        sources,
        paths.aim3_fine_results_source_id,
        role="aim3_fine_results",
    )
    normalized_rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in result["report_rows"]:
        if not isinstance(row, dict):
            raise BundleVerificationError("Aim-3 fine-external report row is not an object")
        key = (str(row.get("task")), str(row.get("scope")), str(row.get("population")))
        if key in normalized_rows:
            raise BundleVerificationError("Aim-3 fine-external report row is duplicated")
        normalized_rows[key] = row
    expected_row_keys: list[tuple[str, str, str]] = []
    for task in AIM3_FINE_TASKS:
        task_record = result["tasks"][task]
        internal = task_record.get("internal_oof")
        if not isinstance(internal, dict):
            raise BundleVerificationError(f"fine-external internal block is absent: {task}")
        positive, negative = AIM3_FINE_EXTERNAL_EXPECTED_INTERNAL_CENSUS[task]
        if (
            internal.get("design") != "honest inherited five-fold OOF on TCGA+SurGen primaries"
            or internal.get("patients") != positive + negative
            or internal.get("positive") != positive
            or internal.get("negative") != negative
        ):
            raise BundleVerificationError(f"fine-external internal census drift: {task}")
        _seed_map(
            internal.get("per_seed_pooled_oof_auroc"),
            context=f"fine-external/internal/{task}/seeds",
        )
        seed_values = [
            float(internal["per_seed_pooled_oof_auroc"][str(seed)]) for seed in MODEL_SEEDS
        ]
        seed_mean, seed_sd = _mean_sample_sd(seed_values)
        fold_map = internal.get("five_fold_ensemble_auroc")
        if not isinstance(fold_map, dict) or tuple(fold_map) != tuple(str(i) for i in range(5)):
            raise BundleVerificationError(f"fine-external fold roster drift: {task}")
        fold_values = [
            _metric01(fold_map[str(index)], context=f"{task}/fold{index}") for index in range(5)
        ]
        fold_mean, fold_sd = _mean_sample_sd(fold_values)
        ensemble = internal.get("five_seed_oof_ensemble")
        ci95 = ensemble.get("ci95") if isinstance(ensemble, dict) else None
        governed = fine_result["fine"]["rungs"][task]
        if (
            not math.isclose(
                _metric01(internal.get("seed_auroc_mean"), context=f"{task}/seed_mean"),
                seed_mean,
                abs_tol=1e-15,
            )
            or not math.isclose(
                _finite(internal.get("seed_auroc_sample_sd"), context=f"{task}/seed_sd"),
                seed_sd,
                abs_tol=1e-15,
            )
            or not math.isclose(
                _metric01(internal.get("fold_auroc_mean"), context=f"{task}/fold_mean"),
                fold_mean,
                abs_tol=1e-15,
            )
            or not math.isclose(
                _finite(internal.get("fold_auroc_sample_sd"), context=f"{task}/fold_sd"),
                fold_sd,
                abs_tol=1e-15,
            )
            or not isinstance(ci95, list)
            or len(ci95) != 2
            or internal["per_seed_pooled_oof_auroc"] != governed["per_seed_auroc"]
            or ensemble.get("auroc") != governed["five_seed_ensemble"]["estimate"]
            or ci95 != governed["five_seed_ensemble"]["ci95_two_sided"]
        ):
            raise BundleVerificationError(
                f"fine-external internal OOF does not replay fine seal: {task}"
            )
        expected_row_keys.append((task, "internal_oof", "TCGA+SurGen-primary"))

        external = task_record.get("external_refit")
        per_cohort = external.get("per_cohort") if isinstance(external, dict) else None
        if (
            not isinstance(per_cohort, dict)
            or set(per_cohort) != set(AIM3_FINE_EXTERNAL_TARGETS)
            or set(external)
            != {
                "per_cohort",
                *AIM3_FINE_EXTERNAL_REPORT_SCOPES[5:],
                *AIM3_FINE_EXTERNAL_SECONDARY_SCOPES,
            }
        ):
            raise BundleVerificationError(f"fine-external scope roster drift: {task}")
        for scope in AIM3_FINE_EXTERNAL_REPORT_SCOPES:
            block = per_cohort[scope] if scope in AIM3_FINE_EXTERNAL_TARGETS else external[scope]
            summary = _validate_external_metric_block(
                block,
                context=f"fine-external/{task}/{scope}",
                expected_census=AIM3_FINE_EXTERNAL_EXPECTED_CENSUS[task][scope],
            )
            if scope == "strict_disjoint_combined" and (
                block.get("role") != "headline patient-pooled AUROC"
                or block.get("dual_role_exclusion") != list(AIM3_FINE_EXTERNAL_DUAL_RIH_PATIENTS)
                or block.get("all_eight_excluded_from_both_rih_roles") is not True
            ):
                raise BundleVerificationError(f"fine-external strict combined role drift: {task}")
            key = (task, "external_refit", scope)
            row = normalized_rows.get(key)
            if not isinstance(row, dict) or any(
                row.get(field) != value
                for field, value in (
                    ("n", summary["n"]),
                    ("positive", summary["positive"]),
                    ("negative", summary["negative"]),
                    ("seed_mean", summary["seed_mean"]),
                    ("seed_sample_sd", summary["seed_sd"]),
                    ("ensemble_auroc", summary["auroc"]),
                    ("ci95", summary["ci95"]),
                    ("status", "ESTIMABLE"),
                )
            ):
                raise BundleVerificationError(
                    f"fine-external report-row replay drift: {task}/{scope}"
                )
            expected_row_keys.append(key)
        for scope in AIM3_FINE_EXTERNAL_SECONDARY_SCOPES:
            seed_key = (
                "per_seed_macro_auroc"
                if scope == "equal_cohort_macro_strict_disjoint"
                else "per_seed_auroc"
            )
            _validate_external_metric_block(
                external[scope],
                context=f"fine-external/{task}/{scope}",
                seed_key=seed_key,
            )
            expected_row_keys.append((task, "external_refit", scope))
    if set(normalized_rows) != set(expected_row_keys):
        raise BundleVerificationError("Aim-3 fine-external 55-row roster drift")

    analysis_contract_path = (
        Path(paths.expected_aim3_fine_external_run_root) / "analysis/contract.json"
    )
    patient_logits_path = (
        Path(paths.expected_aim3_fine_external_run_root) / "analysis/patient_native_logits.parquet"
    )
    bootstrap_path = (
        Path(paths.expected_aim3_fine_external_run_root) / "analysis/bootstrap_distributions.npz"
    )
    analysis_contract, _ = _strict_stable_json(
        analysis_contract_path,
        label="Aim-3 fine-external analysis contract",
        repo=paths.repo,
    )
    if (
        analysis_contract.get("schema_version") != 1
        or analysis_contract.get("status") != "outcome_join_governed_after_inference_seal"
        or analysis_contract.get("bootstrap_draws") != 10_000
        or analysis_contract.get("bootstrap_seed") != 20_260_828
        or analysis_contract.get("seal_status_observed_before_outcome_open")
        != "sealed_before_outcome_join"
        or analysis_contract.get("target_outcomes_opened_after_inference_seal") is not True
        or analysis_contract.get("control_artifacts_absent_at_outcome_join") is not True
        or set(analysis_contract.get("outcome_sources", {})) != set(AIM3_FINE_EXTERNAL_TARGETS)
    ):
        raise BundleVerificationError("Aim-3 fine-external analysis ordering is invalid")
    _require_artifact_identity(
        analysis_contract.get("contract"), contract_path, context="analysis contract campaign"
    )
    _require_artifact_identity(
        analysis_contract.get("inference_seal"),
        inference_path,
        context="analysis contract inference seal",
    )
    for artifact_name, artifact_path in (
        ("analysis_contract", analysis_contract_path),
        ("patient_native_logits", patient_logits_path),
        ("bootstrap_distributions", bootstrap_path),
    ):
        _require_artifact_identity(
            result.get(artifact_name), artifact_path, context=f"result {artifact_name}"
        )
        _require_artifact_identity(
            completion.get(artifact_name),
            artifact_path,
            context=f"completion {artifact_name}",
        )
    _require_artifact_identity(
        result.get("inference_seal"), inference_path, context="result inference seal"
    )
    if (
        completion.get("schema_version") != 1
        or completion.get("status") != "complete_after_sealed_zero_shot_inference"
        or completion.get("bootstrap_array_count") != 50
        or completion.get("report_row_count") != 55
    ):
        raise BundleVerificationError("Aim-3 fine-external analysis completion is invalid")
    _require_artifact_identity(
        completion.get("results"), results_path, context="analysis completion results"
    )
    bindings = _fine_external_report_bindings(result)
    return {
        "campaign_contract": dict(contract_record),
        "preflight": dict(preflight_record),
        "scoring_completion": dict(scoring_record),
        "inference_seal": dict(inference_record),
        "results": dict(result_record),
        "analysis_completion": dict(completion_record),
        "new_fits": 0,
        "score_jobs": 25,
        "score_rows": 11_975,
        "observed_peak_parallel_inference_workers": EXPECTED_PARALLELISM,
        "report_row_count": 55,
        "numeric_binding_count": len(bindings),
    }, bindings


def _require_no_control_artifacts(paths: BundlePaths) -> None:
    """Prove a Phase-1 candidate predates every fixed/repeated control artifact."""

    root = Path(paths.expected_aim3_run_root)
    candidates = (
        root / "train/fixed",
        root / "train/repeated",
        root / "receipts/scheduler_controls.json",
        root / "receipts/scheduler.json",
        root / "receipts/training_complete.json",
        root / "analysis/results.json",
        root / "analysis/analysis_completion.json",
    )
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            raise BundleVerificationError(
                f"Aim-3 Phase-1 candidate already contains a control/full artifact: {candidate}"
            )
    for directory in (
        root / "requests/jobs",
        root / "requests/failures",
        root / "logs/jobs",
        root / "receipts/jobs",
    ):
        if not directory.is_dir():
            continue
        for pattern in ("fixed__*", "repeated__*"):
            if next(directory.glob(pattern), None) is not None:
                raise BundleVerificationError(
                    f"Aim-3 Phase-1 candidate contains control artifact(s) below {directory}"
                )


def _validate_aim3_terminal_analysis_completion(
    paths: BundlePaths,
    result: Mapping[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    """Bind the controller-native completion receipt and result primitives.

    The 229-row ledger uses ``analysis/results.json`` as its terminal scientific
    result row.  The controller's completion receipt is an auxiliary
    certification of that row plus its patient-logit and bootstrap primitives;
    it is hard-pinned and replayed here without changing scientific membership.
    """

    def require_pinned_identity(
        path: Path,
        *,
        expected_path: str,
        expected_size: int,
        expected_sha256: str,
        context: str,
    ) -> dict[str, Any]:
        if path.absolute() != Path(expected_path).absolute():
            raise BundleVerificationError(f"{context} governed path drift")
        identity = _identity(path, display_path=expected_path)
        if (
            identity["path"] != expected_path
            or identity["size_bytes"] != expected_size
            or identity["sha256"] != expected_sha256
        ):
            raise BundleVerificationError(f"{context} pinned identity drift")
        return identity

    def require_link(value: Any, path: Path, *, context: str) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or set(value) != {"path", "size_bytes", "sha256"}
            or Path(str(value.get("path"))).absolute() != path.absolute()
        ):
            raise BundleVerificationError(f"{context} artifact link drift")
        return _require_artifact_identity(value, path, context=context)

    completion_path = Path(EXPECTED_AIM3_ANALYSIS_COMPLETION_PATH)
    completion_identity = require_pinned_identity(
        completion_path,
        expected_path=EXPECTED_AIM3_ANALYSIS_COMPLETION_PATH,
        expected_size=EXPECTED_AIM3_ANALYSIS_COMPLETION_SIZE,
        expected_sha256=EXPECTED_AIM3_ANALYSIS_COMPLETION_SHA256,
        context="Aim-3 analysis completion",
    )
    completion, stable_identity = _strict_stable_json(
        completion_path,
        label="Aim-3 terminal analysis completion",
        repo=paths.repo,
    )
    if stable_identity != completion_identity:
        raise BundleVerificationError("Aim-3 analysis completion changed during replay")
    if (
        set(completion)
        != {
            "schema_version",
            "status",
            "results",
            "patient_native_logits",
            "bootstrap_distributions",
            "report_binding_count",
        }
        or completion.get("schema_version") != 1
        or completion.get("status") != "complete"
        or completion.get("report_binding_count") != 65
        or len(result.get("report_bindings", ())) != 65
    ):
        raise BundleVerificationError("Aim-3 terminal analysis completion schema/accounting drift")
    result_identity = require_link(
        completion.get("results"),
        result_path,
        context="Aim-3 analysis completion results",
    )
    logits_path = Path(EXPECTED_AIM3_PATIENT_NATIVE_LOGITS_PATH)
    logits_identity = require_pinned_identity(
        logits_path,
        expected_path=EXPECTED_AIM3_PATIENT_NATIVE_LOGITS_PATH,
        expected_size=EXPECTED_AIM3_PATIENT_NATIVE_LOGITS_SIZE,
        expected_sha256=EXPECTED_AIM3_PATIENT_NATIVE_LOGITS_SHA256,
        context="Aim-3 patient-native logits",
    )
    bootstrap_path = Path(EXPECTED_AIM3_BOOTSTRAP_DISTRIBUTIONS_PATH)
    bootstrap_identity = require_pinned_identity(
        bootstrap_path,
        expected_path=EXPECTED_AIM3_BOOTSTRAP_DISTRIBUTIONS_PATH,
        expected_size=EXPECTED_AIM3_BOOTSTRAP_DISTRIBUTIONS_SIZE,
        expected_sha256=EXPECTED_AIM3_BOOTSTRAP_DISTRIBUTIONS_SHA256,
        context="Aim-3 bootstrap distributions",
    )
    if (
        require_link(
            result.get("patient_native_logits"),
            logits_path,
            context="Aim-3 results patient-native logits",
        )
        != logits_identity
        or require_link(
            completion.get("patient_native_logits"),
            logits_path,
            context="Aim-3 completion patient-native logits",
        )
        != logits_identity
        or require_link(
            result.get("bootstrap_distributions"),
            bootstrap_path,
            context="Aim-3 results bootstrap distributions",
        )
        != bootstrap_identity
        or require_link(
            completion.get("bootstrap_distributions"),
            bootstrap_path,
            context="Aim-3 completion bootstrap distributions",
        )
        != bootstrap_identity
    ):
        raise BundleVerificationError("Aim-3 terminal analysis primitive linkage drift")
    return {
        "analysis_completion": completion_identity,
        "results": result_identity,
        "patient_native_logits": logits_identity,
        "bootstrap_distributions": bootstrap_identity,
        "report_binding_count": 65,
    }


def _replay_aim3_gate(
    fine: Any,
    control: Any,
    delta: Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Replay the predeclared gate from one-sided 99% FWER bounds only."""

    def bounds(value: Any, *, metric: str) -> tuple[float, float]:
        if not isinstance(value, dict) or set(value) != {
            "estimate",
            "ci95_two_sided",
            "primary_fwer_one_sided",
        }:
            raise BundleVerificationError(f"{context}/{metric} summary schema drift")
        _finite(value["estimate"], context=f"{context}/{metric}/estimate")
        ci95 = value["ci95_two_sided"]
        if (
            not isinstance(ci95, list)
            or len(ci95) != 2
            or _finite(ci95[0], context=f"{context}/{metric}/ci95/lower")
            > _finite(ci95[1], context=f"{context}/{metric}/ci95/upper")
        ):
            raise BundleVerificationError(f"{context}/{metric} descriptive 95% interval drift")
        fwer = value["primary_fwer_one_sided"]
        if (
            not isinstance(fwer, dict)
            or set(fwer) != {"lower", "upper", "confidence"}
            or not math.isclose(
                _finite(
                    fwer.get("confidence"),
                    context=f"{context}/{metric}/fwer/confidence",
                ),
                0.99,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise BundleVerificationError(f"{context}/{metric} one-sided 99% FWER schema drift")
        lower = _finite(fwer["lower"], context=f"{context}/{metric}/fwer/lower")
        upper = _finite(fwer["upper"], context=f"{context}/{metric}/fwer/upper")
        if lower > upper:
            raise BundleVerificationError(f"{context}/{metric} one-sided 99% FWER bounds drift")
        return lower, upper

    fine_lower, fine_upper = bounds(fine, metric="fine")
    control_lower, _ = bounds(control, metric="control")
    delta_lower, _ = bounds(delta, metric="delta")
    conditions = {
        "fine_upper_lt_0p60": fine_upper < 0.60,
        "control_lower_gt_0p50": control_lower > 0.50,
        "delta_lower_gt_zero": delta_lower > 0.0,
    }
    ceiling = all(conditions.values())
    signal = fine_lower > 0.50
    if control_lower <= 0.50:
        verdict = "UNDERPOWERED"
    elif ceiling and signal:
        verdict = "CEILING_WITH_RESIDUAL_SIGNAL"
    elif ceiling:
        verdict = "CEILING"
    elif signal:
        verdict = "FINE_RESOLUTION_EVIDENCE"
    else:
        verdict = "INCONCLUSIVE"
    return {"conditions": conditions, "ceiling": ceiling, "verdict": verdict}


def _validate_aim3_results(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    result, result_record = _source_json(
        paths,
        sources,
        paths.aim3_results_source_id,
        role="aim3_results",
    )
    result_path = _require_aim3_source_path(paths, result_record, "analysis/results.json")
    if (
        result.get("schema_version") != 1
        or result.get("status") != "complete"
        or result.get("campaign") != paths.expected_aim3_experiment
        or result.get("population") != "tcga_surgen_primary"
        or result.get("population_display") != "TCGA + SurGen primaries"
        or result.get("encoder") != "UNI-v1"
        or result.get("model_seeds") != list(MODEL_SEEDS)
        or result.get("wt_draw_seeds") != list(WT_DRAW_SEEDS)
        or result.get("external_development_cohorts") != []
        or result.get("aggregation") != "patient native-logit five-seed ensemble"
    ):
        raise BundleVerificationError("Aim-3 result identity is invalid")

    fixed = result.get("fixed")
    repeated = result.get("repeated")
    fixed_rungs = fixed.get("rungs") if isinstance(fixed, dict) else None
    repeated_rungs = repeated.get("rungs") if isinstance(repeated, dict) else None
    expected_rungs = set(AIM3_RUNG_CONTROL)
    if not isinstance(fixed_rungs, dict) or set(fixed_rungs) != expected_rungs:
        raise BundleVerificationError("Aim-3 fixed rung roster is not exact")
    if not isinstance(repeated_rungs, dict) or set(repeated_rungs) != expected_rungs:
        raise BundleVerificationError("Aim-3 repeated rung roster is not exact")

    consensus: dict[str, str] = {}
    for rung, control in AIM3_RUNG_CONTROL.items():
        fixed_record = fixed_rungs[rung]
        repeated_record = repeated_rungs[rung]
        if not isinstance(fixed_record, dict) or fixed_record.get("control_task") != control:
            raise BundleVerificationError(f"Aim-3 fixed control task drift: {rung}")
        _seed_map(fixed_record.get("fine_per_seed"), context=f"fixed/{rung}/fine")
        _seed_map(fixed_record.get("control_per_seed"), context=f"fixed/{rung}/control")
        replayed_fixed_gate = _replay_aim3_gate(
            fixed_record.get("fine"),
            fixed_record.get("control"),
            fixed_record.get("delta_control_minus_fine"),
            context=f"fixed/{rung}",
        )
        if fixed_record.get("gate") != replayed_fixed_gate:
            raise BundleVerificationError(
                f"Aim-3 fixed gate does not replay from one-sided 99% FWER bounds: {rung}"
            )
        if not isinstance(repeated_record, dict) or repeated_record.get("control_task") != control:
            raise BundleVerificationError(f"Aim-3 repeated control task drift: {rung}")
        _seed_map(repeated_record.get("fine_per_seed"), context=f"repeated/{rung}/fine")
        if repeated_record.get("all_three_draws_required") is not True:
            raise BundleVerificationError(f"Aim-3 repeated three-draw rule drift: {rung}")
        draws = repeated_record.get("draws")
        expected_draw_keys = tuple(str(seed) for seed in WT_DRAW_SEEDS)
        if not isinstance(draws, dict) or tuple(draws) != expected_draw_keys:
            raise BundleVerificationError(
                f"Aim-3 repeated {rung} must contain the exact three WT draws"
            )
        draw_verdicts: list[str] = []
        for draw_seed in expected_draw_keys:
            draw = draws[draw_seed]
            if not isinstance(draw, dict):
                raise BundleVerificationError(f"Aim-3 repeated draw is invalid: {rung}/{draw_seed}")
            _seed_map(
                draw.get("control_per_seed"),
                context=f"repeated/{rung}/{draw_seed}/control",
            )
            replayed_draw_gate = _replay_aim3_gate(
                draw.get("fine"),
                draw.get("control"),
                draw.get("delta_control_minus_fine"),
                context=f"repeated/{rung}/{draw_seed}",
            )
            gate = draw.get("gate")
            if gate != replayed_draw_gate:
                raise BundleVerificationError(
                    "Aim-3 repeated draw gate does not replay from one-sided "
                    f"99% FWER bounds: {rung}/{draw_seed}"
                )
            verdict = replayed_draw_gate["verdict"]
            draw_verdicts.append(verdict)
        ceiling = {"CEILING", "CEILING_WITH_RESIDUAL_SIGNAL"}
        if all(verdict in ceiling for verdict in draw_verdicts):
            replayed_consensus = "CONSENSUS_CEILING"
        elif all(verdict == "UNDERPOWERED" for verdict in draw_verdicts):
            replayed_consensus = "CONSENSUS_UNDERPOWERED"
        else:
            replayed_consensus = "NO_CEILING_CONSENSUS"
        if repeated_record.get("consensus_verdict") != replayed_consensus:
            raise BundleVerificationError(f"Aim-3 repeated consensus does not replay: {rung}")
        consensus[rung] = replayed_consensus

    allele2_trap = repeated_rungs["allele2"]["draws"]["20260824"]
    g12c_trap = repeated_rungs["g12c"]["draws"]["20260825"]
    if (
        allele2_trap["delta_control_minus_fine"]["ci95_two_sided"][0] <= 0.0
        or allele2_trap["gate"]["verdict"] != "INCONCLUSIVE"
        or allele2_trap["delta_control_minus_fine"]["primary_fwer_one_sided"]["lower"] > 0.0
        or g12c_trap["control"]["ci95_two_sided"][0] <= 0.50
        or g12c_trap["gate"]["verdict"] != "UNDERPOWERED"
        or g12c_trap["control"]["primary_fwer_one_sided"]["lower"] > 0.50
    ):
        raise BundleVerificationError("Aim-3 95%-versus-99%-FWER audit trap drift")

    raw_bindings = result.get("report_bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise BundleVerificationError("Aim-3 result has no numeric report bindings")
    expected_bindings: list[dict[str, Any]] = []
    summary_for = {
        "fine_auroc": "fine",
        "control_auroc": "control",
        "delta_auroc": "delta_control_minus_fine",
    }
    for rung in AIM3_RUNG_CONTROL:
        for metric, summary_key in summary_for.items():
            value = fixed_rungs[rung].get(summary_key)
            estimate = value.get("estimate") if isinstance(value, dict) else None
            _finite(estimate, context=f"fixed/{rung}/{metric}")
            expected_bindings.append(
                {
                    "id": f"aim3_source.fixed.{rung}.{metric}",
                    "section": "fixed",
                    "rung": rung,
                    "draw_seed": None,
                    "metric": metric,
                    "value": estimate,
                }
            )
        for draw_seed in WT_DRAW_SEEDS:
            draw = repeated_rungs[rung]["draws"][str(draw_seed)]
            for metric, summary_key in summary_for.items():
                value = draw.get(summary_key)
                estimate = value.get("estimate") if isinstance(value, dict) else None
                _finite(estimate, context=f"repeated/{rung}/{draw_seed}/{metric}")
                expected_bindings.append(
                    {
                        "id": f"aim3_source.repeated.{rung}.wt{draw_seed}.{metric}",
                        "section": "repeated",
                        "rung": rung,
                        "draw_seed": draw_seed,
                        "metric": metric,
                        "value": estimate,
                    }
                )
        expected_bindings.append(
            {
                "id": f"aim3_source.repeated.{rung}.consensus_verdict",
                "section": "repeated",
                "rung": rung,
                "draw_seed": None,
                "metric": "consensus_verdict",
                "value": consensus[rung],
            }
        )
    expected_bindings.sort(key=lambda item: str(item["id"]))
    if raw_bindings != expected_bindings:
        raise BundleVerificationError(
            "Aim-3 report bindings differ from the complete primitive result mapping"
        )
    ids = [str(binding["id"]) for binding in raw_bindings]
    if any(_AIM3_BINDING_ID_RE.fullmatch(binding_id) is None for binding_id in ids):
        raise BundleVerificationError("Aim-3 report binding ID is invalid")
    numeric_bindings = [
        dict(binding) for binding in raw_bindings if binding["metric"] != "consensus_verdict"
    ]
    terminal_analysis = (
        _validate_aim3_terminal_analysis_completion(paths, result, result_path)
        if result_record.get("id") == EXPECTED_AIM3_RESULTS_SOURCE_ID
        else None
    )
    return (
        {
            "results": dict(result_record),
            "terminal_analysis": terminal_analysis,
        },
        numeric_bindings,
        consensus,
    )


def _validate_execution_evidence(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scheduler, scheduler_record = _source_json(
        paths,
        sources,
        paths.aim3_scheduler_source_id,
        role="scheduler_evidence",
    )
    _require_aim3_source_path(paths, scheduler_record, "receipts/scheduler.json")
    controls_scheduler, controls_record = _source_json(
        paths,
        sources,
        paths.aim3_controls_scheduler_source_id,
        role="controls_scheduler_evidence",
    )
    _require_aim3_source_path(paths, controls_record, "receipts/scheduler_controls.json")
    controls_events = controls_scheduler.get("events")
    if (
        controls_scheduler.get("schema_version") != 1
        or controls_scheduler.get("status") != "control_phase_completed_rc0"
        or controls_scheduler.get("phase") != "controls"
        or controls_scheduler.get("configured_max_parallel_chains") != EXPECTED_PARALLELISM
        or controls_scheduler.get("observed_max_parallel_chains") != EXPECTED_PARALLELISM
        or controls_scheduler.get("job_count") != EXPECTED_CONTROL_TRAINING_COUNTS["chains"]
        or controls_scheduler.get("fit_accounting") != EXPECTED_CONTROL_TRAINING_COUNTS
        or controls_scheduler.get("fixed_scheduled_before_repeated") is not True
        or not isinstance(controls_events, list)
        or len(controls_events) != EXPECTED_CONTROL_TRAINING_COUNTS["chains"]
        or any(
            not isinstance(event, dict) or event.get("returncode") != 0 for event in controls_events
        )
    ):
        raise BundleVerificationError(
            "Aim-3 controls scheduler must prove 100 chains/600 fits at configured and observed six-parallel"
        )
    scheduler_events = scheduler.get("events")
    phased = scheduler.get("status") == "completed_phased_rc0"
    if (
        scheduler.get("schema_version") != 1
        or scheduler.get("status") not in {"completed_rc0", "completed_phased_rc0"}
        or scheduler.get("configured_max_parallel_chains") != EXPECTED_PARALLELISM
        or scheduler.get("observed_max_parallel_chains") != EXPECTED_PARALLELISM
        or scheduler.get("job_count") != EXPECTED_TRAINING_COUNTS["chains"]
        or scheduler.get("fit_accounting") != EXPECTED_TRAINING_COUNTS
        or scheduler.get("fixed_ladder_scheduled_before_repeated") is not True
        or not isinstance(scheduler_events, list)
        or len(scheduler_events) != EXPECTED_TRAINING_COUNTS["chains"]
        or any(
            not isinstance(event, dict) or event.get("returncode") != 0
            for event in scheduler_events
        )
    ):
        raise BundleVerificationError(
            "Aim-3 scheduler must prove configured and observed six-parallel execution"
        )
    if phased:
        fine_scheduler_record = next(
            (record for record in sources if record["id"] == paths.aim3_fine_scheduler_source_id),
            None,
        )
        if fine_scheduler_record is None:
            raise BundleVerificationError("Aim-3 phased scheduler lacks fine scheduler source")
        fine_path = parent._lexical_source_path(str(fine_scheduler_record["path"]), paths)
        controls_path = parent._lexical_source_path(str(controls_record["path"]), paths)
        if (
            scheduler.get("phase_order") != ["fine", "fine_analysis_candidate", "controls"]
            or scheduler.get("fine_scheduler")
            != _identity(fine_path, display_path=str(fine_scheduler_record["path"]))
            or scheduler.get("controls_scheduler")
            != _identity(controls_path, display_path=str(controls_record["path"]))
        ):
            raise BundleVerificationError("Aim-3 phased scheduler linkage is invalid")

    training, training_record = _source_json(
        paths,
        sources,
        paths.aim3_training_source_id,
        role="training_completion",
    )
    _require_aim3_source_path(paths, training_record, "receipts/training_complete.json")
    if (
        training.get("schema_version") != 1
        or training.get("status") != "complete_and_certified"
        or training.get("campaign") != paths.expected_aim3_experiment
        or training.get("population") != "tcga_surgen_primary"
        or training.get("encoder") != "UNI-v1"
        or training.get("model_seeds") != list(MODEL_SEEDS)
        or training.get("wt_draw_seeds") != list(WT_DRAW_SEEDS)
        or training.get("fit_accounting") != EXPECTED_TRAINING_COUNTS
        or training.get("external_development_cohorts") != []
        or not isinstance(training.get("job_receipts"), list)
        or len(training["job_receipts"]) != EXPECTED_TRAINING_COUNTS["chains"]
    ):
        raise BundleVerificationError("Aim-3 terminal training accounting is invalid")
    scheduler_path = parent._lexical_source_path(str(scheduler_record["path"]), paths)
    expected_scheduler_identity = _identity(
        scheduler_path, display_path=str(scheduler_record["path"])
    )
    if training.get("scheduler") != expected_scheduler_identity:
        raise BundleVerificationError("Aim-3 training completion does not bind scheduler evidence")
    return {
        "scheduler": dict(scheduler_record),
        "controls_scheduler": dict(controls_record),
        "training_completion": dict(training_record),
        "configured_parallel_chains": EXPECTED_PARALLELISM,
        "observed_peak_parallel_chains": EXPECTED_PARALLELISM,
        "training_counts": dict(EXPECTED_TRAINING_COUNTS),
    }


def _validate_aim4_k32(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = paths.aim4_k32_root
    parent._reject_symlink_chain(root, context="Aim-4 reviews/k32 root")
    if not root.is_dir():
        raise BundleVerificationError(f"Aim-4 reviews/k32 root is missing: {root}")
    expected = dict(paths.expected_aim4_k32_sha256)
    sidecars = dict(paths.expected_aim4_k32_sidecar_sha256)
    if not expected or any(_SHA256_RE.fullmatch(value) is None for value in expected.values()):
        raise BundleVerificationError("Aim-4 reviews/k32 evidence pins are invalid")
    if any(_SHA256_RE.fullmatch(value) is None for value in sidecars.values()):
        raise BundleVerificationError("Aim-4 reviews/k32 sidecar pins are invalid")

    actual_paths: dict[str, Path] = {}
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise BundleVerificationError(f"Aim-4 reviews/k32 contains a symlink: {candidate}")
        if candidate.is_file():
            actual_paths[candidate.relative_to(root).as_posix()] = candidate
    if set(actual_paths) != set(expected) | set(sidecars):
        raise BundleVerificationError("Aim-4 reviews/k32 file roster is not exact")

    identities: dict[str, dict[str, Any]] = {}
    for relative, digest in {**expected, **sidecars}.items():
        path = actual_paths[relative]
        artifact = _identity(path, display_path=_display(path, paths.repo))
        if artifact["sha256"] != digest:
            raise BundleVerificationError(f"Aim-4 reviews/k32 identity drift: {relative}")
        identities[relative] = artifact

    sources_by_path: dict[Path, list[Mapping[str, Any]]] = {}
    for record in sources:
        source_path = parent._lexical_source_path(str(record["path"]), paths).absolute()
        sources_by_path.setdefault(source_path, []).append(record)
    for relative in expected:
        records = sources_by_path.get((root / relative).absolute(), [])
        if len(records) != 1:
            raise BundleVerificationError(
                f"Aim-4 reviews/k32 source-manifest binding is not unique: {relative}"
            )
        if records[0]["role"] != "aim4_k32_review_provenance" or "Aim 4" not in records[0]["aims"]:
            raise BundleVerificationError(
                f"Aim-4 reviews/k32 source metadata is invalid: {relative}"
            )
    if any((root / relative).absolute() in sources_by_path for relative in sidecars):
        raise BundleVerificationError("Aim-4 non-evidence sidecar entered the source manifest")

    structured, _ = _strict_stable_json(
        root / "completed_review_structured.json",
        label="Aim-4 completed structured review",
        repo=paths.repo,
    )
    provenance = structured.get("provenance")
    montage_pins = {
        name.removeprefix("montages/").removesuffix(".jpg"): digest
        for name, digest in expected.items()
        if name
        in {
            "montages/M04.jpg",
            "montages/M05.jpg",
            "montages/M07.jpg",
            "montages/M11.jpg",
        }
    }
    if (
        structured.get("schema_version") != 1
        or structured.get("source_document") != "completed_review.md"
        or structured.get("source_sha256") != paths.expected_aim4_curated_from_sha256
        or structured.get("extraction_status") != "curated_from_completed_blinded_followup"
        or not isinstance(provenance, dict)
        or provenance.get("base_form_sha256") != expected.get("review_form.csv")
        or provenance.get("montage_sha256") != montage_pins
        or set(structured.get("montages", {})) != {"M04", "M05", "M07", "M11"}
        or not isinstance(structured.get("global_quality_flags"), list)
        or not structured["global_quality_flags"]
    ):
        raise BundleVerificationError("Aim-4 reviews/k32 structured provenance is invalid")
    live_digest = expected["completed_review.md"]
    if live_digest == paths.expected_aim4_curated_from_sha256:
        raise BundleVerificationError(
            "Aim-4 disclosed live-versus-curated review provenance distinction disappeared"
        )
    return {
        "evidence_files": {name: identities[name] for name in sorted(expected)},
        "excluded_non_evidence_sidecars": {name: identities[name] for name in sorted(sidecars)},
        "live_completed_review_sha256": live_digest,
        "structured_curated_from_sha256": paths.expected_aim4_curated_from_sha256,
        "boundary": AIM4_BOUNDARY_MARKER,
    }


def _exact_once(text: str, marker: str, *, context: str) -> None:
    if text.count(marker) != 1:
        raise BundleVerificationError(f"{context} must contain exactly one {marker!r}")


def _validate_priority_topology(text: str, *, document: str) -> None:
    positions: list[int] = []
    for heading in PRIORITY_HEADINGS:
        matches = list(re.finditer(rf"^{re.escape(heading)}$", text, re.MULTILINE))
        if len(matches) != 1:
            raise BundleVerificationError(f"{document} must contain exactly one {heading}")
        positions.append(matches[0].start())
    if positions != sorted(positions):
        raise BundleVerificationError(f"{document} six-priority section order is invalid")


def _validate_document_topology(
    paths: BundlePaths,
    sources: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    consensus: Mapping[str, str],
    aim4: Mapping[str, Any],
    *,
    fine_external_bindings: Sequence[Mapping[str, Any]] = (),
    fixed_bindings: Sequence[Mapping[str, Any]] = (),
    target_internal_bindings: Sequence[Mapping[str, Any]] = (),
    aim2_primary_pool_bindings: Sequence[Mapping[str, Any]] = (),
    phase1_candidate: bool = False,
    incremental_candidate: bool = False,
    pending_sources: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    titles = {
        "Experimental_Setup.md": "# FINAL-v13 experimental setup",
        "Results.md": "# FINAL-v13 results",
        "Audit.md": "# FINAL-v13 evidence audit",
    }
    texts: dict[str, str] = {}
    identities: dict[str, Any] = {}
    for name in REPORT_DOCUMENTS:
        path = paths.final_v13 / name
        before = _identity(path, display_path=_display(path, paths.repo))
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise BundleVerificationError(f"FINAL-v13 {name} is not readable UTF-8: {exc}") from exc
        if not text.startswith(titles[name] + "\n"):
            raise BundleVerificationError(f"FINAL-v13 {name} title is invalid")
        if _identity(path, display_path=_display(path, paths.repo)) != before:
            raise BundleVerificationError(f"FINAL-v13 {name} changed while being read")
        _validate_priority_topology(text, document=name)
        texts[name] = text
        identities[name] = before

    for name in ("Experimental_Setup.md", "Results.md"):
        for marker in FIREWALL_MARKERS:
            _exact_once(texts[name], marker, context=name)
        for cohort, role in EXTERNAL_TARGET_ROLES.items():
            _exact_once(
                texts[name],
                f"| {cohort} | {role} | frozen_model_zero_shot_only |",
                context=name,
            )

    results = texts["Results.md"]
    for marker in SECONDARY_MARKERS:
        _exact_once(results, marker, context="Results.md isolated-secondary ledger")
    exclusion_markers = (
        INCREMENTAL_RESULTS_EXCLUSION_MARKERS
        if incremental_candidate
        else SOURCE_ONLY_RESULTS_EXCLUSION_MARKERS
    )
    for marker in exclusion_markers:
        _exact_once(results, marker, context="Results.md source-only eligibility")
    forbidden_numeric = (
        FORBIDDEN_INCREMENTAL_INELIGIBLE_NUMERIC_MARKERS
        if incremental_candidate
        else FORBIDDEN_INELIGIBLE_RESULTS_NUMERIC_MARKERS
    )
    if any(marker in results for marker in forbidden_numeric):
        raise BundleVerificationError(
            "Results.md numerically reports an ineligible non-source-developed model"
        )
    _exact_once(results, AIM4_BOUNDARY_MARKER, context="Results.md Aim-4 boundary")
    _exact_once(
        texts["Audit.md"],
        FAILED_V1_AUDIT_MARKER,
        context="Audit.md failed pre-fit v1 exclusion",
    )
    if not phase1_candidate:
        _exact_once(
            texts["Audit.md"],
            AIM3_TERMINAL_ANALYSIS_COMPLETION_MARKER,
            context="Audit.md Aim-3 terminal analysis completion",
        )
    if phase1_candidate:
        if RECONCILIATION_PINS_BOUNDARY_MARKER in texts["Audit.md"]:
            raise BundleVerificationError(
                "Audit.md Phase-1 candidate mentions unavailable reconciliation pins"
            )
    else:
        _exact_once(
            texts["Audit.md"],
            RECONCILIATION_PINS_BOUNDARY_MARKER,
            context="Audit.md reconciliation-pin metadata boundary",
        )
    required_headings = (
        (
            "### TCGA+SurGen-primary UNI-v1 five-seed fine tasks",
            "### Frozen-refit fine-task external/test performance",
            "### Two conditional canonical fixed matched-WT contrasts",
            "### Remaining canonical fixed and repeated-WT controls — pending",
            "### Legacy all-valid-patient Aim 3 — isolated secondary",
        )
        if incremental_candidate
        else (
            "### Phase-1 TCGA+SurGen-primary UNI-v1 five-seed fine ladders",
            "### Frozen-refit fine-task external/test performance",
            "### Repeated three-draw WT-control consensus — pending Phase 2",
            "### Legacy all-valid-patient Aim 3 — isolated secondary",
        )
        if phase1_candidate
        else (
            "### Controlling TCGA+SurGen-primary UNI-v1 five-seed ladder",
            "### Frozen-refit fine-task external/test performance",
            "### Repeated three-draw WT-control consensus",
            "### Legacy all-valid-patient Aim 3 — isolated secondary",
        )
    )
    for required_heading in required_headings:
        _exact_once(results, required_heading, context="Results.md Aim-3 topology")
    phase1_marker = (
        "AIM3_PHASE1_STATUS: CANDIDATE_UNSEALED; FIXED_AND_REPEATED_CONTROLS_PENDING; DO_NOT_SEAL."
    )
    incremental_marker = (
        "FINAL_V13_INCREMENTAL_STATUS: CANDIDATE_UNSEALED; "
        "FINE_EXTERNAL_COMPLETE; TWO_FIXED_COMPLETE; "
        "SOURCE_ANCHORED_FEWSHOT_COMPLETE; "
        "REMAINING_CANONICAL_AND_REPEATED_CONTROLS_PENDING; DO_NOT_SEAL."
    )
    if incremental_candidate:
        _exact_once(
            results,
            incremental_marker,
            context="Results.md incremental status",
        )
        if phase1_marker in results:
            raise BundleVerificationError(
                "Results.md incremental candidate retains stale Phase-1 status"
            )
    elif phase1_candidate:
        _exact_once(results, phase1_marker, context="Results.md Phase-1 status")
    elif phase1_marker in results or incremental_marker in results:
        raise BundleVerificationError("Results.md retains stale Phase-1 candidate status")
    if not phase1_candidate:
        _exact_once(
            results,
            AIM3_GATE_QUALIFICATION_MARKER,
            context="Results.md Aim-3 99% FWER gate qualification",
        )
        for name in REPORT_DOCUMENTS:
            _exact_once(
                texts[name],
                THEORETICAL_CEILING_NOT_RUN_MARKER,
                context=f"{name} theoretical-ceiling eligibility disclosure",
            )
            _exact_once(
                texts[name],
                AIM3_CEILING_SCOPE_MARKER,
                context=f"{name} matched-WT ceiling scope disclosure",
            )
            _exact_once(
                texts[name],
                AIM3_FIXED_NONOVERRIDE_MARKER,
                context=f"{name} fixed-versus-consensus precedence disclosure",
            )

    observed_binding_lines = re.findall(
        r"^<!-- AIM3_SOURCE_VALUE ([a-z0-9][a-z0-9._-]*[a-z0-9]|[a-z0-9]) (\S+) -->$",
        results,
        flags=re.MULTILINE,
    )
    expected_binding_lines = [
        (str(binding["id"]), _canonical_number(binding["value"])) for binding in bindings
    ] + [
        (str(binding["id"]), _canonical_number(binding["value"]))
        for binding in fine_external_bindings
    ]
    if observed_binding_lines != expected_binding_lines:
        raise BundleVerificationError(
            "Results.md Aim-3 numeric source bindings are not exact, complete, and ordered"
        )
    observed_fixed_lines = re.findall(
        r"^<!-- AIM3_CONDITIONAL_FIXED_VALUE ([a-z0-9][a-z0-9._-]*[a-z0-9]|[a-z0-9]) (\S+) -->$",
        results,
        flags=re.MULTILINE,
    )
    expected_fixed_lines = [
        (str(binding["id"]), _canonical_number(binding["value"])) for binding in fixed_bindings
    ]
    if observed_fixed_lines != expected_fixed_lines:
        raise BundleVerificationError(
            "Results.md conditional fixed-control bindings are not exact, complete, and ordered"
        )
    observed_target_lines = re.findall(
        r"^<!-- AIM2_TARGET_INTERNAL_VALUE ([a-z0-9][a-z0-9._-]*[a-z0-9]|[a-z0-9]) (\S+) -->$",
        results,
        flags=re.MULTILINE,
    )
    expected_target_lines = [
        (str(binding["id"]), _canonical_number(binding["value"]))
        for binding in target_internal_bindings
    ]
    if observed_target_lines != expected_target_lines:
        raise BundleVerificationError(
            "Results.md target-internal bindings are not exact, complete, and ordered"
        )
    observed_primary_pool_lines = re.findall(
        r"^<!-- AIM2_PRIMARY_POOL_VALUE ([a-z0-9][a-z0-9._-]*[a-z0-9]|[a-z0-9]) (\S+) -->$",
        results,
        flags=re.MULTILINE,
    )
    expected_primary_pool_lines = [
        (str(binding["id"]), _canonical_number(binding["value"]))
        for binding in aim2_primary_pool_bindings
    ]
    if observed_primary_pool_lines != expected_primary_pool_lines:
        raise BundleVerificationError(
            "Results.md Aim-2 conventional-primary pooled bindings are not exact, "
            "complete, and ordered"
        )
    observed_consensus = re.findall(
        r"^<!-- AIM3_CONSENSUS ([a-z0-9_]+) ([A-Z_]+) -->$",
        results,
        flags=re.MULTILINE,
    )
    expected_consensus = (
        [] if phase1_candidate else [(rung, consensus[rung]) for rung in sorted(consensus)]
    )
    if observed_consensus != expected_consensus:
        raise BundleVerificationError("Results.md Aim-3 three-draw consensus bindings drift")

    _exact_once(
        results,
        f"reviews/k32/completed_review.md live SHA-256: `{aim4['live_completed_review_sha256']}`",
        context="Results.md Aim-4 live review provenance",
    )
    _exact_once(
        results,
        "reviews/k32/completed_review_structured.json curated-from SHA-256: "
        f"`{aim4['structured_curated_from_sha256']}`",
        context="Results.md Aim-4 curated provenance",
    )

    audit_rows = re.findall(
        r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$",
        texts["Audit.md"],
        flags=re.MULTILINE,
    )
    observed_sources = {source_id: digest for source_id, digest in audit_rows}
    expected_sources = {str(source["id"]): str(source["sha256"]) for source in sources}
    if len(audit_rows) != len(observed_sources) or observed_sources != expected_sources:
        raise BundleVerificationError("Audit.md source index is not one-to-one and exact")
    pending_rows = re.findall(
        r"^\| `([^`]+)` \| PENDING_PHASE2 \|$",
        texts["Audit.md"],
        flags=re.MULTILINE,
    )
    expected_pending = [str(source["id"]) for source in pending_sources]
    if pending_rows != expected_pending:
        raise BundleVerificationError("Audit.md Phase-2 pending-source ledger is not exact")

    pins = dict(paths.expected_final_document_sha256)
    if set(pins.values()) != {UNFROZEN}:
        for name in REPORT_DOCUMENTS:
            if identities[name]["sha256"] != pins[name]:
                raise BundleVerificationError(f"FINAL-v13 {name} SHA-256 drift")
    return identities


def _validate_bundle(paths: BundlePaths) -> dict[str, Any]:
    paths, reconciliation_pins, reconciliation_pins_identity = _resolve_full_reconciliation_pins(
        paths
    )
    _require_frozen_expectations(paths)
    parent_receipt, _, parent_sources, parent_identities = _validate_parent(paths)
    manifest, sources, pending, manifest_identity = _validate_final_manifest(paths, parent_sources)
    if pending:
        raise BundleVerificationError("FINAL-v13 full verification encountered pending sources")
    pins_path = paths.reconciliation_pins.absolute()
    source_paths = {
        parent._lexical_source_path(str(record["path"]), paths).absolute() for record in sources
    }
    if pins_path in source_paths:
        raise BundleVerificationError(
            "FINAL-v13 reconciliation pins entered the scientific source manifest"
        )
    fine, fine_bindings = _validate_aim3_fine_results(paths, sources)
    fine_execution = _validate_fine_execution_evidence(paths, sources)
    fine_external, fine_external_bindings = _validate_aim3_fine_external_results(paths, sources)
    aim3, bindings, consensus = _validate_aim3_results(paths, sources)
    execution = _validate_execution_evidence(paths, sources)
    source_ids = {str(record["id"]) for record in sources}
    aim2_primary_pool: dict[str, Any] | None = None
    if _is_production_bundle(paths) or EXPECTED_AIM2_PRIMARY_POOL_SOURCE_ID in source_ids:
        aim2_primary_pool = _validate_aim2_conventional_primary_pool(paths, sources)
    retained_incremental: dict[str, Any] | None = None
    target_internal_bindings: Sequence[Mapping[str, Any]] = ()
    if set(EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS).issubset(source_ids):
        retained_incremental = _validate_retained_incremental_evidence(paths, sources)
        target_internal_bindings = retained_incremental["target_internal_bindings"]
    aim4 = _validate_aim4_k32(paths, sources)
    documents = _validate_document_topology(
        paths,
        sources,
        bindings,
        consensus,
        aim4,
        fine_external_bindings=fine_external_bindings,
        target_internal_bindings=target_internal_bindings,
        aim2_primary_pool_bindings=(
            aim2_primary_pool["bindings"] if aim2_primary_pool is not None else ()
        ),
    )
    return {
        "resolved_paths": paths,
        "reconciliation_pins": reconciliation_pins,
        "reconciliation_pins_identity": reconciliation_pins_identity,
        "parent_receipt": parent_receipt,
        "parent_identities": parent_identities,
        "manifest": manifest,
        "manifest_identity": manifest_identity,
        "sources": sources,
        "fine": fine,
        "fine_bindings": fine_bindings,
        "fine_execution": fine_execution,
        "fine_external": fine_external,
        "fine_external_bindings": fine_external_bindings,
        "aim3": aim3,
        "aim3_bindings": bindings,
        "aim3_consensus": consensus,
        "execution": execution,
        "retained_incremental": retained_incremental,
        "aim2_primary_pool": aim2_primary_pool,
        "aim4": aim4,
        "documents": documents,
    }


def _validate_phase1_candidate_bundle(
    paths: BundlePaths, *, incremental_candidate: bool = False
) -> dict[str, Any]:
    """Validate a fine-plus-external candidate while publication stays impossible."""

    if paths.reconciliation_pins.exists() or paths.reconciliation_pins.is_symlink():
        raise BundleVerificationError(
            "FINAL-v13 Phase-1 candidate must not contain full reconciliation pins"
        )
    _require_frozen_expectations(paths, phase1_candidate=True)
    parent_receipt, _, parent_sources, parent_identities = _validate_parent(paths)
    manifest, sources, pending, manifest_identity = _validate_final_manifest(
        paths, parent_sources, phase1_candidate=True
    )
    fine, bindings = _validate_aim3_fine_results(paths, sources)
    execution = _validate_fine_execution_evidence(paths, sources)
    fine_external, fine_external_bindings = _validate_aim3_fine_external_results(paths, sources)
    incremental: dict[str, Any] | None = None
    fixed_bindings: Sequence[Mapping[str, Any]] = ()
    target_bindings: Sequence[Mapping[str, Any]] = ()
    if incremental_candidate:
        helper, _ = _load_trusted_phase1_candidate()
        if (
            tuple(paths.expected_extension_source_ids) != EXPECTED_INCREMENTAL_EXTENSION_SOURCE_IDS
            or paths.expected_phase1_manifest_status != INCREMENTAL_MANIFEST_STATUS
        ):
            raise BundleVerificationError(
                "incremental verifier paths are not hard-bound to the 87-source roster"
            )
        if _is_production_bundle(paths):
            payload = helper.build_candidate_payload()
            if (
                not isinstance(payload, dict)
                or set(payload) != {"source_manifest", "documents"}
                or not isinstance(payload.get("documents"), dict)
            ):
                raise BundleVerificationError(
                    "trusted incremental helper returned an invalid payload"
                )
            _require_exact_installed_bytes(
                paths.final_v13 / SOURCE_MANIFEST_NAME,
                _canonical_json_bytes(payload["source_manifest"]),
                label="FINAL-v13 incremental source manifest",
            )
            for name in REPORT_DOCUMENTS:
                _require_exact_installed_bytes(
                    paths.final_v13 / name,
                    payload["documents"][name].encode("utf-8"),
                    label=f"FINAL-v13 incremental {name}",
                )
        incremental = helper.validate_incremental_evidence(paths, sources)
        fixed_bindings = incremental["two_fixed_bindings"]
        target_bindings = incremental["target_internal_bindings"]
    else:
        _require_no_control_artifacts(paths)
    aim4 = _validate_aim4_k32(paths, sources)
    documents = _validate_document_topology(
        paths,
        sources,
        bindings,
        {},
        aim4,
        fine_external_bindings=fine_external_bindings,
        fixed_bindings=fixed_bindings,
        target_internal_bindings=target_bindings,
        phase1_candidate=True,
        incremental_candidate=incremental_candidate,
        pending_sources=pending,
    )
    return {
        "parent_receipt": parent_receipt,
        "parent_identities": parent_identities,
        "manifest": manifest,
        "manifest_identity": manifest_identity,
        "sources": sources,
        "pending": pending,
        "fine": fine,
        "fine_bindings": bindings,
        "fine_execution": execution,
        "fine_external": fine_external,
        "fine_external_bindings": fine_external_bindings,
        "incremental": incremental,
        "aim4": aim4,
        "documents": documents,
    }


def build_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Build the deterministic receipt without publishing it."""

    requested = default_paths() if paths is None else paths
    manifest_path = requested.final_v13 / SOURCE_MANIFEST_NAME
    if manifest_path.is_file():
        manifest, _ = _strict_stable_json(
            manifest_path,
            label="FINAL-v13 source manifest",
            repo=requested.repo,
        )
        if manifest.get("status") in {
            PHASE1_MANIFEST_STATUS,
            INCREMENTAL_MANIFEST_STATUS,
        }:
            raise BundleVerificationError(
                "FINAL-v13 unsealed candidate cannot build or publish a receipt; "
                "reconciliation_pins.json is unavailable"
            )
    trusted_reconciliation = _validate_trusted_full_reconciliation(requested)
    validated = _validate_bundle(requested)
    if _validate_trusted_full_reconciliation(requested) != trusted_reconciliation:
        raise BundleVerificationError(
            "trusted FINAL-v13 reconciliation changed during receipt construction"
        )
    selected = validated["resolved_paths"]
    parent_identities = validated["parent_identities"]
    retained = validated.get("retained_incremental")
    aim2_primary_pool = validated.get("aim2_primary_pool")
    retained_target_internal = (
        {
            "methods": [
                "pure_ridge_linear_probe",
                "source_anchored_residual_ridge",
            ],
            "support_per_class_total": [2, 4, 8, 16],
            "numeric_binding_count": len(retained["target_internal_bindings"]),
            "result_role": "TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION",
            "external_validation_claim_permitted": False,
            "source_model_modified_or_selected": False,
            "surgen_metastatic_source_family_exposed": True,
        }
        if isinstance(retained, dict)
        else None
    )
    return {
        "schema_version": 1,
        "bundle": "reports/final_v13",
        "status": FINAL_SEALED_STATUS,
        "organization": "six_priority_firewalled_integrated_report",
        "parent_final_v12_1": {
            "status": validated["parent_receipt"]["status"],
            "receipt": parent_identities["receipt"],
            "source_manifest": parent_identities["source_manifest"],
            "documents": parent_identities["documents"],
            "verifier": parent_identities["verifier"],
            "verifier_test": parent_identities["verifier_test"],
            "authoritative_source_count": selected.expected_parent_source_count,
        },
        "documents": validated["documents"],
        "source_manifest": validated["manifest_identity"],
        "reconciliation_pins": {
            "identity": validated["reconciliation_pins_identity"],
            "status": validated["reconciliation_pins"]["status"],
            "scientific_source_manifest_membership": (
                validated["reconciliation_pins"]["scientific_source_manifest_membership"]
            ),
        },
        "reconciliation_generator": {
            "exact_production_rendering": trusted_reconciliation["exact_production_rendering"],
            **trusted_reconciliation["generator_identities"],
        },
        "authoritative_sources": validated["sources"],
        "source_count": len(validated["sources"]),
        "extension_source_count": len(selected.expected_extension_source_ids),
        "extension_source_ids": list(selected.expected_extension_source_ids),
        "main_model_external_firewall": {
            "development": "TCGA+SurGen primary only",
            "external_target_roles": dict(EXTERNAL_TARGET_ROLES),
            "allowed_external_operations": [
                "label_blind_packing_preprocessing",
                "frozen_model_scoring",
            ],
            "forbidden_external_operations": [
                "training",
                "fitting",
                "fine_tuning",
                "adaptation",
                "calibration",
                "threshold_selection",
                "hyperparameter_selection",
                "vocabulary_fitting",
                "construction_or_model_selection",
            ],
        },
        "isolated_secondary_evidence": {
            "legacy_multi_cohort_loco": SECONDARY_MARKERS[0],
            "target_internal_few_shot": SECONDARY_MARKERS[1],
            "theoretical_ceiling": THEORETICAL_CEILING_NOT_RUN_MARKER,
            "legacy_all_valid_aim3": SECONDARY_MARKERS[2],
        },
        "aim2_target_internal_few_shot": retained_target_internal,
        "aim2_conventional_external_primary_pool": aim2_primary_pool,
        "aim3_tcga_surgen_primary": {
            "phase1_fine": {
                **validated["fine"],
                "numeric_binding_count": len(validated["fine_bindings"]),
                "execution": validated["fine_execution"],
            },
            **validated["aim3"],
            "model_seeds": list(MODEL_SEEDS),
            "encoder": "UNI-v1",
            "wt_draw_seeds": list(WT_DRAW_SEEDS),
            "numeric_binding_count": len(validated["aim3_bindings"]),
            "repeated_consensus": dict(validated["aim3_consensus"]),
            "execution": validated["execution"],
            "ceiling_interpretation": {
                "scope": AIM3_CEILING_SCOPE_MARKER,
                "fixed_single_draw_precedence": AIM3_FIXED_NONOVERRIDE_MARKER,
                "theoretical_ceiling": THEORETICAL_CEILING_NOT_RUN_MARKER,
            },
            "fine_external_zero_shot": {
                **validated["fine_external"],
                "numeric_binding_count": len(validated["fine_external_bindings"]),
            },
        },
        "aim4_reviews_k32": validated["aim4"],
        "verification": {
            "verifier": _identity(
                selected.verifier_code,
                display_path=_display(selected.verifier_code, selected.repo),
            ),
            "tests": _identity(
                selected.verifier_test,
                display_path=_display(selected.verifier_test, selected.repo),
            ),
        },
        "checks": {
            "recursive_sealed_final_v12_1_replay": "PASS",
            "exact_142_record_parent_reuse_plus_frozen_extension": "PASS",
            "direct_complete_source_rehash": "PASS",
            "strict_json_and_no_symlinks": "PASS",
            "six_priority_sections_exact_and_ordered": "PASS",
            "main_model_external_firewall_and_target_roles": "PASS",
            "legacy_and_target_internal_evidence_isolated": "PASS",
            "aim3_tcga_surgen_primary_univ1_seed_roster": "PASS",
            "aim3_repeated_three_draw_consensus_replayed": "PASS",
            "aim3_numeric_results_source_bound": "PASS",
            "aim3_fine_external_zero_shot_source_bound": "PASS",
            "aim3_fine_external_no_fit_firewall": "PASS",
            "aim2_conventional_primary_patient_pool_source_replayed": (
                "PASS" if aim2_primary_pool is not None else "NOT_APPLICABLE_SYNTHETIC_FIXTURE"
            ),
            "retained_target_internal_bindings_source_replayed": (
                "PASS" if retained_target_internal is not None else "NOT_PRESENT"
            ),
            "aim3_configured_and_observed_six_parallel": "PASS",
            "aim3_terminal_training_accounting": "PASS",
            "aim4_reviews_k32_exact_provenance": "PASS",
            "one_to_one_audit_source_index": "PASS",
            "pre_seal_reconciliation_pins_excluded_and_receipt_bound": "PASS",
            "hash_pinned_reconciliation_generators": "PASS",
            "deterministic_full_reconciliation_byte_match": "PASS",
            "final_v13_document_manifest_verifier_test_hashes": "PASS",
        },
    }


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def check_bundle(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Verify either a fine-plus-external candidate or the full bundle."""

    selected = default_paths() if paths is None else paths
    manifest_path = selected.final_v13 / SOURCE_MANIFEST_NAME
    if not manifest_path.is_file():
        # Preserve the intentionally-unfrozen scaffold failure before any
        # candidate files exist.
        _require_frozen_expectations(selected)
        raise BundleVerificationError("FINAL-v13 source manifest is absent")
    manifest, _ = _strict_stable_json(
        manifest_path, label="FINAL-v13 source manifest", repo=selected.repo
    )
    manifest_status = manifest.get("status")
    incremental_candidate = manifest_status == INCREMENTAL_MANIFEST_STATUS
    phase1_candidate = manifest_status == selected.expected_phase1_manifest_status
    if incremental_candidate:
        # Installed bytes cannot choose their own scientific roster.  Dispatch
        # only through the verifier's independent, hard-coded 87-ID authority.
        selected = replace(
            selected,
            expected_extension_source_ids=EXPECTED_INCREMENTAL_EXTENSION_SOURCE_IDS,
            expected_phase1_manifest_status=INCREMENTAL_MANIFEST_STATUS,
        )
    if phase1_candidate or incremental_candidate:
        if selected.destination.exists() or selected.destination.is_symlink():
            raise BundleVerificationError(
                "FINAL-v13 Phase-1 candidate must remain unsealed; receipt presence is invalid"
            )
        validated = _validate_phase1_candidate_bundle(
            selected, incremental_candidate=incremental_candidate
        )
        if incremental_candidate:
            incremental = validated["incremental"]
            if not isinstance(incremental, dict):
                raise BundleVerificationError("incremental evidence validation return drift")
            return {
                "bundle": "reports/final_v13",
                "status": "INCREMENTAL_TWO_FIXED_AND_SOURCE_ANCHORED_FEWSHOT_CANDIDATE_VERIFIED_UNSEALED",
                "seal_ready": False,
                "published_receipt_present": False,
                "parent_status": validated["parent_receipt"]["status"],
                "material_source_count": len(validated["sources"]),
                "incremental_extension_source_count": len(EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS),
                "pending_phase2_source_count": len(validated["pending"]),
                "pending_phase2_source_ids": [str(record["id"]) for record in validated["pending"]],
                "documents": validated["documents"],
                "document_pins": "UNFROZEN_INCREMENTAL_CANDIDATE",
                "source_manifest": validated["manifest_identity"],
                "aim3_fine": {
                    **validated["fine"],
                    "numeric_binding_count": len(validated["fine_bindings"]),
                    "execution": validated["fine_execution"],
                },
                "aim3_fine_external": {
                    **validated["fine_external"],
                    "numeric_binding_count": len(validated["fine_external_bindings"]),
                },
                "aim3_two_conditional_fixed": {
                    "task_count": 2,
                    "numeric_binding_count": len(incremental["two_fixed_bindings"]),
                    "claim_scope": "conditional_single_canonical_draw_not_consensus_or_ceiling",
                },
                "aim2_target_internal_few_shot": {
                    "methods": [
                        "pure_ridge_linear_probe",
                        "source_anchored_residual_ridge",
                    ],
                    "support_per_class_total": [2, 4, 8, 16],
                    "numeric_binding_count": len(incremental["target_internal_bindings"]),
                    "result_role": "TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION",
                    "external_validation_claim_permitted": False,
                },
                "publication_blockers": [
                    "remaining_canonical_fixed_controls_pending",
                    "repeated_wt_controls_pending",
                    "three_draw_consensus_pending",
                    "final_document_hashes_unfrozen",
                ],
            }
        return {
            "bundle": "reports/final_v13",
            "status": "PHASE1_FINE_EXTERNAL_CANDIDATE_VERIFIED_UNSEALED",
            "seal_ready": False,
            "published_receipt_present": selected.destination.exists()
            or selected.destination.is_symlink(),
            "parent_status": validated["parent_receipt"]["status"],
            "material_source_count": len(validated["sources"]),
            "pending_phase2_source_count": len(validated["pending"]),
            "pending_phase2_source_ids": [str(record["id"]) for record in validated["pending"]],
            "documents": validated["documents"],
            "document_pins": (
                "UNFROZEN_PHASE1_CANDIDATE"
                if set(selected.expected_final_document_sha256.values()) == {UNFROZEN}
                else "FROZEN"
            ),
            "source_manifest": validated["manifest_identity"],
            "aim3_fine": {
                **validated["fine"],
                "numeric_binding_count": len(validated["fine_bindings"]),
                "execution": validated["fine_execution"],
            },
            "aim3_fine_external": {
                **validated["fine_external"],
                "numeric_binding_count": len(validated["fine_external_bindings"]),
            },
            "publication_blockers": [
                "fixed_controls_pending",
                "repeated_wt_controls_pending",
                "three_draw_consensus_pending",
                "final_document_hashes_unfrozen",
            ],
        }
    receipt = build_receipt(selected)
    return {
        "bundle": receipt["bundle"],
        "status": "READY_TO_SEAL",
        "published_receipt_present": selected.destination.exists()
        or selected.destination.is_symlink(),
        "parent_status": receipt["parent_final_v12_1"]["status"],
        "source_count": receipt["source_count"],
        "documents": receipt["documents"],
        "source_manifest": receipt["source_manifest"],
        "reconciliation_pins": receipt["reconciliation_pins"],
        "aim2_target_internal_few_shot": receipt["aim2_target_internal_few_shot"],
        "aim2_conventional_external_primary_pool": receipt[
            "aim2_conventional_external_primary_pool"
        ],
        "verification": receipt["verification"],
        "checks": receipt["checks"],
    }


def verify_published_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Rebuild and byte-compare an already-published receipt."""

    selected = default_paths() if paths is None else paths
    parent._reject_symlink_chain(selected.destination, context="published FINAL-v13 receipt")
    if not selected.destination.is_file():
        raise BundleVerificationError("published FINAL-v13 receipt is absent")
    published, _ = _strict_stable_json(
        selected.destination, label="published FINAL-v13 receipt", repo=selected.repo
    )
    expected = build_receipt(selected)
    if published != expected or selected.destination.read_bytes() != _receipt_bytes(expected):
        raise BundleVerificationError("published FINAL-v13 receipt byte identity drift")
    return published


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Publish exactly once and refuse every overwrite or symlink target."""

    selected = default_paths() if paths is None else paths
    if selected.destination.exists() or selected.destination.is_symlink():
        raise BundleVerificationError("refusing to overwrite FINAL-v13 receipt")
    parent._reject_symlink_chain(selected.destination.parent, context="FINAL-v13 receipt directory")
    receipt = build_receipt(selected)
    content = _receipt_bytes(receipt)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected.destination.name}.",
        suffix=".tmp",
        dir=selected.destination.parent,
    )
    temporary = Path(temporary_name)
    temporary_inode: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            stat = os.fstat(handle.fileno())
            temporary_inode = (stat.st_dev, stat.st_ino)
        try:
            os.link(temporary, selected.destination)
        except FileExistsError as exc:
            raise BundleVerificationError("FINAL-v13 receipt was concurrently published") from exc
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
    actions.add_argument("--check", action="store_true", help="run read-only verification")
    actions.add_argument("--seal", action="store_true", help="create the receipt exactly once")
    actions.add_argument(
        "--verify-published",
        action="store_true",
        help="rebuild and compare an existing receipt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.seal:
            result = seal()
        elif args.verify_published:
            result = verify_published_receipt()
        else:
            result = check_bundle()
    except BundleVerificationError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
