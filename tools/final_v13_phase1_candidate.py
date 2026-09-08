#!/usr/bin/env python3
"""Reconcile the unsealed FINAL-v13 incremental evidence candidate.

The user-requested publication sequence has an intentional intermediate state.
The five fine molecular tasks and their frozen-refit external scoring are
complete; two separately governed canonical fixed matched-WT contrasts are
complete; and source-anchored pure-ridge and residual-ridge combined-metastatic
few-shot analyses are complete.  The latter are target-internal analyses, not
external validation.  The remaining canonical controls and all repeated-WT
draws remain pending.

The resulting candidate inherits the sealed FINAL-v12.1 roster byte-for-byte,
adds 83 material extension records, and retains four future control/full-result
records as explicit ``PENDING_PHASE2`` entries.  Every newly added evidence
byte is pinned below before it can enter the manifest.  ``check`` delegates to
the production FINAL-v13 verifier.  ``install --apply`` is the only write path:
it atomically replaces the manifest and three rendered documents, refuses any
receipt or reconciliation pins, and never seals the bundle.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import final_v13_bundle_receipt as verifier  # noqa: E402


@dataclass(frozen=True)
class ExtensionSource:
    """One exact FINAL-v13 extension-source declaration."""

    source_id: str
    aims: tuple[str, ...]
    experiments: tuple[str, ...]
    role: str
    path: Path
    pending_phase2: bool = False


RUN_ROOT = Path(verifier.EXPECTED_AIM3_RUN_ROOT)
FINE_EXTERNAL_ROOT = Path(verifier.EXPECTED_AIM3_FINE_EXTERNAL_RUN_ROOT)
TWO_FIXED_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim3_tcga_surgen_primary_two_fixed_controls_v1_20260828"
)
PURE_RIDGE_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_tcga_surgen_pure_ridge_combined_fewshot_univ1_5seed_v1_20260828"
)
RESIDUAL_RIDGE_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_tcga_surgen_residual_combined_fewshot_univ1_5seed_v1_20260828"
)
E2C_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_cap8192_e2c_offset_v1_20260819"
)
AIM4_RECEIPT = (
    REPO
    / "reports/reruns/final_v5_additions_20260820/"
    "aim4_human_read_legacy_v2/receipt.json"
)

AIM3_RESULTS_SOURCE_ID = "aim3-source-primary-results"
AIM3_SCHEDULER_SOURCE_ID = "aim3-source-primary-scheduler"
AIM3_TRAINING_SOURCE_ID = "aim3-source-primary-training-completion"
AIM3_FINE_RESULTS_SOURCE_ID = "aim3-source-primary-fine-results"
AIM3_FINE_SCHEDULER_SOURCE_ID = "aim3-source-primary-fine-scheduler"
AIM3_FINE_TRAINING_SOURCE_ID = "aim3-source-primary-fine-training-completion"
AIM3_CONTROLS_SCHEDULER_SOURCE_ID = "aim3-source-primary-controls-scheduler"
AIM3_FINE_EXTERNAL_RESULTS_SOURCE_ID = (
    "aim3-source-primary-fine-external-results"
)
AIM3_FINE_EXTERNAL_COMPLETION_SOURCE_ID = (
    "aim3-source-primary-fine-external-analysis-completion"
)
AIM3_FINE_EXTERNAL_INFERENCE_SEAL_SOURCE_ID = (
    "aim3-source-primary-fine-external-inference-seal"
)
AIM3_FINE_EXTERNAL_SCORING_SOURCE_ID = (
    "aim3-source-primary-fine-external-scoring-completion"
)

PENDING_PHASE2_SOURCE_IDS = (
    AIM3_CONTROLS_SCHEDULER_SOURCE_ID,
    AIM3_RESULTS_SOURCE_ID,
    AIM3_SCHEDULER_SOURCE_ID,
    AIM3_TRAINING_SOURCE_ID,
)

FINE_DISPLAY_ORDER = ("codon", "g12d_broad", "allele1", "allele2", "g12c")
FINE_DISPLAY_NAMES = {
    "codon": "Codon (G12 vs other KRAS-mutant)",
    "g12d_broad": "G12D broad (G12D vs non-G12D)",
    "allele1": "G12D within G12",
    "allele2": "G12V within G12",
    "g12c": "G12C within G12",
}

TARGET_INTERNAL_ROLE = "TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION"
INCREMENTAL_STATUS = (
    "candidate_incremental_two_fixed_and_source_anchored_fewshot_complete_"
    "unsealed_final_v13"
)
EXPECTED_INCREMENTAL_COUNTS = {
    "extension": 87,
    "material_extension": 83,
    "parent": 142,
    "material_total": 225,
    "pending": 4,
}
ADAPTATION_LAYOUT_SEEDS = (20260817, 20260818, 20260819, 20260820, 20260821)
ADAPTATION_BUDGETS = (2, 4, 8, 16)
ADAPTATION_SCOPES = ("RIH-M", "SurGen-M", "pooled_combined", "equal_cohort_macro")
ADAPTATION_SCOPE_NAMES = {
    "RIH-M": "RIH-metastatic",
    "SurGen-M": "SurGen-metastatic",
    "pooled_combined": "Pooled combined",
    "equal_cohort_macro": "Equal-cohort macro",
}

# Immutable terminal identities audited after campaign-native terminal verify.
# These pins prevent a later evidence mutation from being accepted by merely
# regenerating the candidate manifest (anti-TOFU).
EXPECTED_INCREMENTAL_IDENTITIES: dict[str, tuple[int, str]] = {
    "aim2-source-anchored-pure-ridge-analysis-completion": (2347, "a383ba3b10e9623b85a2edde890a6ab0cf3a8ef4055198bf8dc3d596238cabad"),
    "aim2-source-anchored-pure-ridge-contract": (14894, "ddc1764ccaabe27c0e9945b5115d6726e7854971547a4d3fa707e1377e7b0656"),
    "aim2-source-anchored-pure-ridge-controller": (84678, "f99d61d58a5c3d5348890010e23f0caba9bb25846debc7291c030b8da2f06929"),
    "aim2-source-anchored-pure-ridge-controller-test": (15796, "beb32d54a7d12c29028ed6e79a7cee5c5e65f0544c73e2d790f4ea0c59151523"),
    "aim2-source-anchored-pure-ridge-deep-preflight": (1700, "80bfaa5129c010ed5fe3224e16ece4baef4d9d508801d3cf98b9143c8ec51f2a"),
    "aim2-source-anchored-pure-ridge-inference-environment": (406, "b736cec9ace4e070f3ebaaca50c964976f87dc4362fe775e151aa1117750e51e"),
    "aim2-source-anchored-pure-ridge-inference-seal": (9201, "96a6ece499877123ff59ab89b55900b399742fbcb0c6a1003e0ad94beedabca2"),
    "aim2-source-anchored-pure-ridge-label-blind-rih-manifest": (5177, "dec443d99e437e7cb3fc3297a4f98302081ba4455386e3b7d930ff142feb1e5c"),
    "aim2-source-anchored-pure-ridge-label-blind-surgen-manifest": (6772, "90eb00ef0baf6a33fc4b423affbce33dde1d38068add59cc136f870f13cf6d15"),
    "aim2-source-anchored-pure-ridge-labeled-target-manifest": (35475, "c3fa9d98d9ff475c4872c91bc04f5fa2362a9589d150fcdbd4c25dae9d0c19c6"),
    "aim2-source-anchored-pure-ridge-oof-predictions": (973209, "7abddd272236fb181f38a25bb498bf7499c1a89a997c08dc3d629b1bde3a11d4"),
    "aim2-source-anchored-pure-ridge-results": (33407, "ae5f2d5093e255d79aaf8755f0dc7357d3e1d19c9eed3233adbdf0181d8a0a1b"),
    "aim2-source-anchored-pure-ridge-scheduler": (17968, "ae71ce09d548c4879fbc29b7e3f063791e782289f52ad5b13c70bb31169d977b"),
    "aim2-source-anchored-pure-ridge-source-embedding-scheduler": (8837, "ce862226095f7c75ed81b7e4d8681590004a18fe88e3e1178ca83cb5d4c6dbfc"),
    "aim2-source-anchored-pure-ridge-target-internal-open": (1550, "df591bd9f34ae88ce60ac96c5b0debb8ea6192d4102dc5a9f58e61b1e19e74d7"),
    "aim2-source-anchored-residual-analysis-completion": (2721, "0e6e045be83697f2a40cf1f65fca10d508b7e9819c528a377a258652b09e95e7"),
    "aim2-source-anchored-residual-contract": (14648, "5c307f23eb96c6166267411e3e7216673718699bb1c43062ddefa04158c2071a"),
    "aim2-source-anchored-residual-controller": (70752, "a97c2b8d1734c8056473994166332b50154c8e7db9ef31a13597c9ad4700b0c6"),
    "aim2-source-anchored-residual-controller-test": (17500, "680638e6d38fb50af1f6660f11895236d10b0bea352b328526b84fac906c64fc"),
    "aim2-source-anchored-residual-oof-predictions": (3196590, "1d87462e64a08d5cf9444eb9fa9dbbf88b6739c6c222cbb51197fa154d1af08a"),
    "aim2-source-anchored-residual-preflight": (10307, "6314ceaed6ba8e404bb6725bacfb40c3a7ef19f8b27a5baa0c13f31886e38c45"),
    "aim2-source-anchored-residual-results": (34640, "4ce7c1f1c47bf7554fc4e2ccbdf87e330d23703d18250f1d7d98a41e04a75832"),
    "aim2-source-anchored-residual-scheduler": (18194, "2908fd4c0b70be40545f1aadd5ec41e18fd3601e5f990568e9bcf83daaf68b61"),
    "aim3-source-primary-two-fixed-analysis-completion": (1029, "518c74536c536aacc44e968f94c53e6a4118ea0661940613a71349aa0eaade1a"),
    "aim3-source-primary-two-fixed-bootstrap-distributions": (348378, "ab8700affc31ce50d9793d5c4088b643742cc3c83bb77911d9e4bdcb5680d922"),
    "aim3-source-primary-two-fixed-contract": (8791, "23e3cd38cc45242619368835651142928baf4990e9cddf05cecabd7d131fdb54"),
    "aim3-source-primary-two-fixed-control-manifest-codon": (181692, "03914cd60403f19d7fab45be0648a68891d3a98bb8f712a6091cb2b0327befb1"),
    "aim3-source-primary-two-fixed-control-manifest-g12d-broad": (182390, "dbabe63eb7e0ab7aae389cdc56913a01f2ee860c7655e64da3d2c0d5a481fbae"),
    "aim3-source-primary-two-fixed-control-split-integrity-codon": (228, "a06b3fac13c9427ea0b665badc02aa94904468ddff2056059815379a3acb0350"),
    "aim3-source-primary-two-fixed-control-split-integrity-g12d-broad": (233, "9fd42a96718e19c7ad47f153389052e5639e4170960fe5378dee127eaae0e863"),
    "aim3-source-primary-two-fixed-control-splits-codon": (62287, "6b8c8a5823c487d43f2fb9198c9ef1eaf2c07eac4a119c1f345524e56eaedbf4"),
    "aim3-source-primary-two-fixed-control-splits-g12d-broad": (62101, "7ccff1db3a614133a0777a5f1e5d6dd0a1b2b2fbbe1f0889b0fb2232a2a8da66"),
    "aim3-source-primary-two-fixed-controller": (60430, "9e02473aa37d09c049242f03e877ad85f810857b11f0384b7914a4ca685b5ebc"),
    "aim3-source-primary-two-fixed-controller-test": (14899, "83eea4bb66d7f7144126b9bae1426ba3ce00cd961b9babd55daf3b5d9f993516"),
    "aim3-source-primary-two-fixed-job-plan": (31412, "5fb1e1207c4bfb3c11135290e126391b74422930fe8a3b4fe3861a0e6aa5a9e9"),
    "aim3-source-primary-two-fixed-patient-native-logits": (25803, "b178bfa7b3034b69856eed32c5fb97958175b8dbe582fc546742a7b8535c302e"),
    "aim3-source-primary-two-fixed-preflight": (7546, "32e04b14b6a218dd99c137dfdadd118055ecd2c52d2f65abea247cd184df9c12"),
    "aim3-source-primary-two-fixed-results": (6710, "8e7fd3c95b2b2720196b05ca786751dcb5b1317b2b8cabbf44d978331886c565"),
    "aim3-source-primary-two-fixed-scheduler": (10460, "d9328ddb8be6af2fecc1799b389833d5c5624f01b351d5adffd9bb0746bdd28a"),
    "aim3-source-primary-two-fixed-training-completion": (7187, "9f87414c9464d9edd84113424d186065685e5ed204820ffda7fb2e4f4ee9c77c"),
}


def _incremental_sources() -> tuple[ExtensionSource, ...]:
    """Return the exact 40 terminal additions beyond the original ledger."""

    fixed_experiment = "Aim 3 two conditional canonical fixed matched-WT contrasts"
    pure_experiment = "source-anchored combined-met pure-ridge few-shot"
    residual_experiment = "source-anchored combined-met residual-ridge few-shot"
    source_root = Path(verifier.EXPECTED_AIM3_RUN_ROOT)
    sources = (
        ExtensionSource("aim3-source-primary-two-fixed-contract", ("Aim 3",), (fixed_experiment,), "two_fixed_campaign_contract", TWO_FIXED_ROOT / "contract.json"),
        ExtensionSource("aim3-source-primary-two-fixed-controller", ("Aim 3",), (fixed_experiment,), "two_fixed_campaign_controller", REPO / "aim3_tcga_surgen_primary_two_fixed_controls_campaign.py"),
        ExtensionSource("aim3-source-primary-two-fixed-controller-test", ("Aim 3",), (fixed_experiment,), "two_fixed_campaign_contract_test", REPO / "tests/test_aim3_tcga_surgen_primary_two_fixed_controls_campaign.py"),
        ExtensionSource("aim3-source-primary-two-fixed-job-plan", ("Aim 3",), (fixed_experiment,), "two_fixed_exact_job_plan", TWO_FIXED_ROOT / "jobs/job_plan.json"),
        ExtensionSource("aim3-source-primary-two-fixed-preflight", ("Aim 3",), (fixed_experiment,), "two_fixed_deep_preflight", TWO_FIXED_ROOT / "receipts/preflight.json"),
        ExtensionSource("aim3-source-primary-two-fixed-scheduler", ("Aim 3",), (fixed_experiment,), "two_fixed_scheduler_evidence", TWO_FIXED_ROOT / "receipts/scheduler.json"),
        ExtensionSource("aim3-source-primary-two-fixed-training-completion", ("Aim 3",), (fixed_experiment,), "two_fixed_training_completion", TWO_FIXED_ROOT / "receipts/training_complete.json"),
        ExtensionSource("aim3-source-primary-two-fixed-results", ("Aim 3",), (fixed_experiment,), "two_fixed_results", TWO_FIXED_ROOT / "analysis/results.json"),
        ExtensionSource("aim3-source-primary-two-fixed-analysis-completion", ("Aim 3",), (fixed_experiment,), "two_fixed_analysis_completion", TWO_FIXED_ROOT / "analysis/analysis_completion.json"),
        ExtensionSource("aim3-source-primary-two-fixed-patient-native-logits", ("Aim 3",), (fixed_experiment,), "two_fixed_patient_native_logits", TWO_FIXED_ROOT / "analysis/patient_native_logits.parquet"),
        ExtensionSource("aim3-source-primary-two-fixed-bootstrap-distributions", ("Aim 3",), (fixed_experiment,), "two_fixed_bootstrap_distributions", TWO_FIXED_ROOT / "analysis/bootstrap_distributions.npz"),
        ExtensionSource("aim3-source-primary-two-fixed-control-manifest-codon", ("Aim 3",), (fixed_experiment,), "two_fixed_control_manifest", source_root / "inputs/manifests/fixed__ctrl_codon.csv"),
        ExtensionSource("aim3-source-primary-two-fixed-control-manifest-g12d-broad", ("Aim 3",), (fixed_experiment,), "two_fixed_control_manifest", source_root / "inputs/manifests/fixed__ctrl_g12d_broad.csv"),
        ExtensionSource("aim3-source-primary-two-fixed-control-splits-codon", ("Aim 3",), (fixed_experiment,), "two_fixed_control_splits", source_root / "inputs/splits/fixed__ctrl_codon/aim1_balanced5/splits.parquet"),
        ExtensionSource("aim3-source-primary-two-fixed-control-splits-g12d-broad", ("Aim 3",), (fixed_experiment,), "two_fixed_control_splits", source_root / "inputs/splits/fixed__ctrl_g12d_broad/aim1_balanced5/splits.parquet"),
        ExtensionSource("aim3-source-primary-two-fixed-control-split-integrity-codon", ("Aim 3",), (fixed_experiment,), "two_fixed_control_split_integrity", source_root / "inputs/splits/fixed__ctrl_codon/aim1_balanced5/.integrity_hash"),
        ExtensionSource("aim3-source-primary-two-fixed-control-split-integrity-g12d-broad", ("Aim 3",), (fixed_experiment,), "two_fixed_control_split_integrity", source_root / "inputs/splits/fixed__ctrl_g12d_broad/aim1_balanced5/.integrity_hash"),
        ExtensionSource("aim2-source-anchored-pure-ridge-contract", ("Aim 2",), (pure_experiment,), "target_internal_campaign_contract", PURE_RIDGE_ROOT / "contract.json"),
        ExtensionSource("aim2-source-anchored-pure-ridge-controller", ("Aim 2",), (pure_experiment,), "target_internal_campaign_controller", REPO / "aim2_tcga_surgen_pure_ridge_combined_fewshot_campaign.py"),
        ExtensionSource("aim2-source-anchored-pure-ridge-controller-test", ("Aim 2",), (pure_experiment,), "target_internal_campaign_contract_test", REPO / "tests/test_aim2_tcga_surgen_pure_ridge_combined_fewshot_campaign.py"),
        ExtensionSource("aim2-source-anchored-pure-ridge-label-blind-rih-manifest", ("Aim 2",), (pure_experiment,), "target_internal_label_blind_roster", PURE_RIDGE_ROOT / "inputs/label_blind/rih_metastatic.csv"),
        ExtensionSource("aim2-source-anchored-pure-ridge-label-blind-surgen-manifest", ("Aim 2",), (pure_experiment,), "target_internal_label_blind_roster", PURE_RIDGE_ROOT / "inputs/label_blind/surgen_metastatic.csv"),
        ExtensionSource("aim2-source-anchored-pure-ridge-labeled-target-manifest", ("Aim 2",), (pure_experiment,), "target_internal_labeled_roster", PURE_RIDGE_ROOT / "inputs/target_internal/labeled_metastatic.csv"),
        ExtensionSource("aim2-source-anchored-pure-ridge-deep-preflight", ("Aim 2",), (pure_experiment,), "target_internal_deep_preflight", PURE_RIDGE_ROOT / "receipts/deep_preflight.json"),
        ExtensionSource("aim2-source-anchored-pure-ridge-source-embedding-scheduler", ("Aim 2",), (pure_experiment,), "target_internal_label_blind_embedding_scheduler", PURE_RIDGE_ROOT / "receipts/source_embedding_scheduler.json"),
        ExtensionSource("aim2-source-anchored-pure-ridge-inference-environment", ("Aim 2",), (pure_experiment,), "target_internal_inference_environment", PURE_RIDGE_ROOT / "source_inference/environment.json"),
        ExtensionSource("aim2-source-anchored-pure-ridge-inference-seal", ("Aim 2",), (pure_experiment,), "target_internal_inference_seal", PURE_RIDGE_ROOT / "source_inference/inference_seal.json"),
        ExtensionSource("aim2-source-anchored-pure-ridge-target-internal-open", ("Aim 2",), (pure_experiment,), "target_internal_outcome_open", PURE_RIDGE_ROOT / "receipts/target_internal_open.json"),
        ExtensionSource("aim2-source-anchored-pure-ridge-scheduler", ("Aim 2",), (pure_experiment,), "target_internal_scheduler_evidence", PURE_RIDGE_ROOT / "receipts/pure_ridge_scheduler.json"),
        ExtensionSource("aim2-source-anchored-pure-ridge-oof-predictions", ("Aim 2",), (pure_experiment,), "target_internal_oof_predictions", PURE_RIDGE_ROOT / "adaptation/pure_ridge_oof.parquet"),
        ExtensionSource("aim2-source-anchored-pure-ridge-results", ("Aim 2",), (pure_experiment,), "target_internal_results", PURE_RIDGE_ROOT / "analysis/results.json"),
        ExtensionSource("aim2-source-anchored-pure-ridge-analysis-completion", ("Aim 2",), (pure_experiment,), "target_internal_analysis_completion", PURE_RIDGE_ROOT / "analysis/completion.json"),
        ExtensionSource("aim2-source-anchored-residual-contract", ("Aim 2",), (residual_experiment,), "target_internal_campaign_contract", RESIDUAL_RIDGE_ROOT / "contract.json"),
        ExtensionSource("aim2-source-anchored-residual-controller", ("Aim 2",), (residual_experiment,), "target_internal_campaign_controller", REPO / "aim2_tcga_surgen_residual_combined_fewshot_campaign.py"),
        ExtensionSource("aim2-source-anchored-residual-controller-test", ("Aim 2",), (residual_experiment,), "target_internal_campaign_contract_test", REPO / "tests/test_aim2_tcga_surgen_residual_combined_fewshot_campaign.py"),
        ExtensionSource("aim2-source-anchored-residual-preflight", ("Aim 2",), (residual_experiment,), "target_internal_deep_preflight", RESIDUAL_RIDGE_ROOT / "receipts/residual_preflight.json"),
        ExtensionSource("aim2-source-anchored-residual-scheduler", ("Aim 2",), (residual_experiment,), "target_internal_scheduler_evidence", RESIDUAL_RIDGE_ROOT / "receipts/residual_scheduler.json"),
        ExtensionSource("aim2-source-anchored-residual-oof-predictions", ("Aim 2",), (residual_experiment,), "target_internal_oof_predictions", RESIDUAL_RIDGE_ROOT / "adaptation/residual_ridge_oof.parquet"),
        ExtensionSource("aim2-source-anchored-residual-results", ("Aim 2",), (residual_experiment,), "target_internal_results", RESIDUAL_RIDGE_ROOT / "analysis/results.json"),
        ExtensionSource("aim2-source-anchored-residual-analysis-completion", ("Aim 2",), (residual_experiment,), "target_internal_analysis_completion", RESIDUAL_RIDGE_ROOT / "analysis/completion.json"),
    )
    if len(sources) != 40 or {source.source_id for source in sources} != set(EXPECTED_INCREMENTAL_IDENTITIES):
        raise verifier.BundleVerificationError("incremental source/pin roster drift")
    return sources


def extension_sources() -> tuple[ExtensionSource, ...]:
    """Return the exact sorted FINAL-v13 extension ledger."""

    sources = [
        ExtensionSource(
            "aim2-e2c-direct-analysis-audit",
            ("Aim 2",),
            ("E2c target-internal residual adaptation",),
            "target_internal_analysis_audit",
            E2C_ROOT / "receipts/analysis_audit.json",
        ),
        ExtensionSource(
            "aim2-e2c-direct-lineage-completion",
            ("Aim 2",),
            ("E2c target-internal residual adaptation",),
            "target_internal_lineage_completion",
            E2C_ROOT / "lineage_complete.json",
        ),
        ExtensionSource(
            "aim2-e2c-direct-results",
            ("Aim 2",),
            ("E2c target-internal residual adaptation",),
            "target_internal_results",
            E2C_ROOT / "analysis/e2c_native_logit_offset_cap8192.json",
        ),
        ExtensionSource(
            "aim3-source-primary-campaign-contract",
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "campaign_contract",
            RUN_ROOT / "contract.json",
        ),
        ExtensionSource(
            "aim3-source-primary-campaign-controller",
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "campaign_controller",
            RUN_ROOT / "source_snapshot/aim3_tcga_surgen_primary_five_seed_campaign.py",
        ),
        ExtensionSource(
            "aim3-source-primary-campaign-test",
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "campaign_contract_test",
            RUN_ROOT
            / "source_snapshot/tests/test_aim3_tcga_surgen_primary_five_seed_campaign.py",
        ),
        ExtensionSource(
            AIM3_CONTROLS_SCHEDULER_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "controls_scheduler_evidence",
            RUN_ROOT / "receipts/scheduler_controls.json",
            pending_phase2=True,
        ),
        ExtensionSource(
            AIM3_FINE_EXTERNAL_COMPLETION_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT,),
            "fine_external_analysis_completion",
            FINE_EXTERNAL_ROOT / "analysis/analysis_completion.json",
        ),
        ExtensionSource(
            "aim3-source-primary-fine-external-campaign-contract",
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT,),
            "fine_external_campaign_contract",
            FINE_EXTERNAL_ROOT / "contract.json",
        ),
        ExtensionSource(
            "aim3-source-primary-fine-external-campaign-controller",
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT,),
            "fine_external_campaign_controller",
            REPO / "aim3_tcga_surgen_primary_fine_external_campaign.py",
        ),
        ExtensionSource(
            "aim3-source-primary-fine-external-campaign-test",
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT,),
            "fine_external_campaign_contract_test",
            REPO / "tests/test_aim3_tcga_surgen_primary_fine_external_campaign.py",
        ),
        ExtensionSource(
            AIM3_FINE_EXTERNAL_INFERENCE_SEAL_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT,),
            "fine_external_inference_seal",
            FINE_EXTERNAL_ROOT / "inference/inference_seal.json",
        ),
        ExtensionSource(
            "aim3-source-primary-fine-external-preflight",
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT,),
            "fine_external_deep_preflight",
            FINE_EXTERNAL_ROOT / "receipts/deep_preflight.json",
        ),
        ExtensionSource(
            AIM3_FINE_EXTERNAL_RESULTS_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT,),
            "fine_external_results",
            FINE_EXTERNAL_ROOT / "analysis/results.json",
        ),
        ExtensionSource(
            AIM3_FINE_EXTERNAL_SCORING_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_FINE_EXTERNAL_EXPERIMENT,),
            "fine_external_scoring_completion",
            FINE_EXTERNAL_ROOT / "receipts/scoring_complete.json",
        ),
        ExtensionSource(
            AIM3_FINE_RESULTS_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "aim3_fine_results",
            RUN_ROOT / "analysis/fine_results.json",
        ),
        ExtensionSource(
            AIM3_FINE_SCHEDULER_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "fine_scheduler_evidence",
            RUN_ROOT / "receipts/scheduler_fine.json",
        ),
        ExtensionSource(
            AIM3_FINE_TRAINING_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "fine_training_completion",
            RUN_ROOT / "receipts/training_complete_fine.json",
        ),
        ExtensionSource(
            "aim3-source-primary-preflight",
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "deep_preflight",
            RUN_ROOT / "receipts/preflight.json",
        ),
        ExtensionSource(
            AIM3_RESULTS_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "aim3_results",
            RUN_ROOT / "analysis/results.json",
            pending_phase2=True,
        ),
        ExtensionSource(
            AIM3_SCHEDULER_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "scheduler_evidence",
            RUN_ROOT / "receipts/scheduler.json",
            pending_phase2=True,
        ),
        ExtensionSource(
            "aim3-source-primary-source-contract",
            ("Aim 1", "Aim 3"),
            ("TCGA+SurGen-primary source lineage",),
            "frozen_source_contract",
            RUN_ROOT / "inputs/source_lineage/source_contract.json",
        ),
        ExtensionSource(
            "aim3-source-primary-source-integrity",
            ("Aim 1", "Aim 3"),
            ("TCGA+SurGen-primary source lineage",),
            "frozen_source_split_integrity",
            RUN_ROOT / "inputs/source_lineage/splits/.integrity_hash",
        ),
        ExtensionSource(
            "aim3-source-primary-source-manifest",
            ("Aim 1", "Aim 3"),
            ("TCGA+SurGen-primary source lineage",),
            "frozen_source_manifest",
            RUN_ROOT / "inputs/source_lineage/tcga_surgen_primary.csv",
        ),
        ExtensionSource(
            "aim3-source-primary-source-splits",
            ("Aim 1", "Aim 3"),
            ("TCGA+SurGen-primary source lineage",),
            "frozen_source_splits",
            RUN_ROOT / "inputs/source_lineage/splits/splits.parquet",
        ),
        ExtensionSource(
            "aim3-source-primary-source-summary",
            ("Aim 1", "Aim 3"),
            ("TCGA+SurGen-primary source lineage",),
            "frozen_source_split_summary",
            RUN_ROOT / "inputs/source_lineage/splits/summary.json",
        ),
        ExtensionSource(
            "aim3-source-primary-source-training-seal",
            ("Aim 1", "Aim 3"),
            ("TCGA+SurGen-primary source lineage",),
            "frozen_source_training_completion",
            RUN_ROOT / "inputs/source_lineage/source_training_complete.json",
        ),
        ExtensionSource(
            AIM3_TRAINING_SOURCE_ID,
            ("Aim 3",),
            (verifier.EXPECTED_AIM3_EXPERIMENT,),
            "training_completion",
            RUN_ROOT / "receipts/training_complete.json",
            pending_phase2=True,
        ),
        ExtensionSource(
            "aim4-k32-human-read-receipt",
            ("Aim 4",),
            ("reviews/k32 human read",),
            "aim4_k32_review_receipt",
            AIM4_RECEIPT,
        ),
        ExtensionSource(
            "final-v13-bundle-verifier",
            ("Shared",),
            ("FINAL-v13 bundle verification",),
            "bundle_verifier",
            REPO / "tools/final_v13_bundle_receipt.py",
        ),
        ExtensionSource(
            "final-v13-bundle-verifier-test",
            ("Shared",),
            ("FINAL-v13 bundle verification",),
            "bundle_verifier_test",
            REPO / "tests/test_final_v13_bundle_receipt.py",
        ),
    ]
    k32_root = REPO / "reviews/k32"
    for index, relative in enumerate(sorted(verifier.EXPECTED_AIM4_K32_SHA256)):
        sources.append(
            ExtensionSource(
                f"aim4-k32-source-{index + 1:02d}",
                ("Aim 4",),
                ("reviews/k32 human read",),
                "aim4_k32_review_provenance",
                k32_root / relative,
            )
        )
    sources.extend(_incremental_sources())
    ordered = tuple(sorted(sources, key=lambda item: item.source_id))
    ids = tuple(item.source_id for item in ordered)
    paths = tuple(str(item.path) for item in ordered)
    if (
        len(ordered) != EXPECTED_INCREMENTAL_COUNTS["extension"]
        or len(ids) != len(set(ids))
        or len(paths) != len(set(paths))
    ):
        raise verifier.BundleVerificationError(
            "FINAL-v13 incremental extension ledger is not exactly 87 unique records"
        )
    if tuple(item.source_id for item in ordered if item.pending_phase2) != (
        "aim3-source-primary-controls-scheduler",
        "aim3-source-primary-results",
        "aim3-source-primary-scheduler",
        "aim3-source-primary-training-completion",
    ):
        raise verifier.BundleVerificationError("Phase-2 pending-source roster drift")
    return ordered


def extension_ids() -> tuple[str, ...]:
    return tuple(source.source_id for source in extension_sources())


def candidate_paths() -> verifier.BundlePaths:
    """Bind the production verifier to the frozen extension ledger."""

    return dataclasses.replace(
        verifier.default_paths(),
        expected_extension_source_ids=extension_ids(),
        expected_phase1_manifest_status=INCREMENTAL_STATUS,
        aim3_results_source_id=AIM3_RESULTS_SOURCE_ID,
        aim3_scheduler_source_id=AIM3_SCHEDULER_SOURCE_ID,
        aim3_training_source_id=AIM3_TRAINING_SOURCE_ID,
        aim3_fine_results_source_id=AIM3_FINE_RESULTS_SOURCE_ID,
        aim3_fine_scheduler_source_id=AIM3_FINE_SCHEDULER_SOURCE_ID,
        aim3_fine_training_source_id=AIM3_FINE_TRAINING_SOURCE_ID,
        aim3_controls_scheduler_source_id=AIM3_CONTROLS_SCHEDULER_SOURCE_ID,
        aim3_fine_external_results_source_id=AIM3_FINE_EXTERNAL_RESULTS_SOURCE_ID,
        aim3_fine_external_completion_source_id=AIM3_FINE_EXTERNAL_COMPLETION_SOURCE_ID,
        aim3_fine_external_inference_seal_source_id=(
            AIM3_FINE_EXTERNAL_INFERENCE_SEAL_SOURCE_ID
        ),
        aim3_fine_external_scoring_source_id=AIM3_FINE_EXTERNAL_SCORING_SOURCE_ID,
    )


def _display(path: Path) -> str:
    absolute = path.absolute()
    try:
        return absolute.relative_to(REPO.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def _material_record(source: ExtensionSource) -> dict[str, Any]:
    identity = verifier._identity(source.path, display_path=_display(source.path))
    return {
        "id": source.source_id,
        "aims": list(source.aims),
        "experiments": list(source.experiments),
        "role": source.role,
        **identity,
    }


def _pending_record(source: ExtensionSource) -> dict[str, Any]:
    if source.path.exists() or source.path.is_symlink():
        raise verifier.BundleVerificationError(
            f"Phase-2 pending artifact already exists: {source.path}"
        )
    return {
        "id": source.source_id,
        "aims": list(source.aims),
        "experiments": list(source.experiments),
        "role": source.role,
        "path": _display(source.path),
    }


def build_source_manifest() -> dict[str, Any]:
    """Build, but do not write, the exact Phase-1 source manifest."""

    paths = candidate_paths()
    if verifier.sha256_file(paths.parent_manifest) != paths.expected_parent_manifest_sha256:
        raise verifier.BundleVerificationError("sealed FINAL-v12.1 manifest SHA-256 drift")
    parent_manifest, _ = verifier._strict_stable_json(
        paths.parent_manifest,
        label="sealed FINAL-v12.1 source manifest",
        repo=paths.repo,
    )
    parent_sources = parent_manifest.get("artifacts")
    if not isinstance(parent_sources, list) or len(parent_sources) != 142:
        raise verifier.BundleVerificationError("sealed parent source roster is not 142 records")

    material = [
        _material_record(source)
        for source in extension_sources()
        if not source.pending_phase2
    ]
    pending = [
        _pending_record(source)
        for source in extension_sources()
        if source.pending_phase2
    ]
    artifacts = sorted([*parent_sources, *material], key=lambda record: record["id"])
    pending.sort(key=lambda record: record["id"])
    if (
        len(artifacts) != EXPECTED_INCREMENTAL_COUNTS["material_total"]
        or len(pending) != EXPECTED_INCREMENTAL_COUNTS["pending"]
    ):
        raise verifier.BundleVerificationError(
            "incremental material/pending source counts drift"
        )
    return {
        "schema_version": 2,
        "bundle": "final_v13",
        "status": INCREMENTAL_STATUS,
        "artifacts": artifacts,
        "pending_artifacts": pending,
    }


def _source_map(
    sources: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id = {str(record["id"]): record for record in sources}
    if len(by_id) != len(sources):
        raise verifier.BundleVerificationError("incremental source IDs are not unique")
    return by_id


def _source_path(paths: verifier.BundlePaths, record: dict[str, Any]) -> Path:
    return verifier.parent._lexical_source_path(str(record["path"]), paths)


def _source_json(
    paths: verifier.BundlePaths,
    by_id: dict[str, dict[str, Any]],
    source_id: str,
) -> tuple[dict[str, Any], Path]:
    record = by_id.get(source_id)
    if record is None:
        raise verifier.BundleVerificationError(
            f"incremental evidence source is absent: {source_id}"
        )
    path = _source_path(paths, record)
    value, _ = verifier._strict_stable_json(
        path,
        label=f"incremental source {source_id}",
        repo=paths.repo,
    )
    return value, path


def _require_link(link: Any, path: Path, *, context: str) -> None:
    if not isinstance(link, dict):
        raise verifier.BundleVerificationError(f"{context} identity is absent")
    identity = verifier._identity(path, display_path=str(path))
    if (
        link.get("size_bytes") != identity["size_bytes"]
        or link.get("sha256") != identity["sha256"]
        or Path(str(link.get("path"))).absolute() != path.absolute()
    ):
        raise verifier.BundleVerificationError(f"{context} identity drift")


def _require_incremental_identity_roster(
    paths: verifier.BundlePaths,
    sources: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id = _source_map(sources)
    specs = {source.source_id: source for source in _incremental_sources()}
    if set(specs) != set(EXPECTED_INCREMENTAL_IDENTITIES):
        raise verifier.BundleVerificationError("incremental evidence pin roster drift")
    for source_id, (size_bytes, digest) in EXPECTED_INCREMENTAL_IDENTITIES.items():
        record = by_id.get(source_id)
        spec = specs[source_id]
        if record is None:
            raise verifier.BundleVerificationError(
                f"incremental evidence source is absent: {source_id}"
            )
        expected_path = spec.path.absolute()
        observed_path = _source_path(paths, record).absolute()
        if (
            observed_path != expected_path
            or record.get("role") != spec.role
            or tuple(record.get("aims", ())) != spec.aims
            or tuple(record.get("experiments", ())) != spec.experiments
            or record.get("size_bytes") != size_bytes
            or record.get("sha256") != digest
        ):
            raise verifier.BundleVerificationError(
                f"incremental evidence manifest binding drift: {source_id}"
            )
        identity = verifier._identity(expected_path, display_path=str(record["path"]))
        if identity["size_bytes"] != size_bytes or identity["sha256"] != digest:
            raise verifier.BundleVerificationError(
                f"incremental evidence terminal identity drift: {source_id}"
            )
    return by_id


def _auc(labels: Any, scores: Any) -> float:
    """Compute tie-aware AUROC directly from average ranks."""

    import numpy as np
    import pandas as pd

    y = np.asarray(labels, dtype=int)
    eta = np.asarray(scores, dtype=float)
    if len(y) != len(eta) or len(y) == 0 or not np.isfinite(eta).all():
        raise verifier.BundleVerificationError("AUROC replay input is invalid")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if set(np.unique(y)) != {0, 1} or n_pos == 0 or n_neg == 0:
        raise verifier.BundleVerificationError("AUROC replay requires both classes")
    ranks = pd.Series(eta).rank(method="average").to_numpy(float)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _close(observed: Any, expected: Any, *, context: str) -> None:
    if not isinstance(observed, (int, float)) or not math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=1e-15
    ):
        raise verifier.BundleVerificationError(f"{context} numeric replay drift")


def _interval(value: Any, *, context: str) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
        or float(value[0]) > float(value[1])
    ):
        raise verifier.BundleVerificationError(f"{context} interval is invalid")
    return float(value[0]), float(value[1])


def _validate_two_fixed(
    paths: verifier.BundlePaths,
    by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    import pandas as pd

    contract, contract_path = _source_json(
        paths, by_id, "aim3-source-primary-two-fixed-contract"
    )
    expected_counts = {
        "task_variants": 2,
        "chains": 10,
        "oof_folds": 50,
        "p75_refits": 10,
        "physical_mil_fits": 60,
    }
    if (
        contract.get("schema_version") != 1
        or contract.get("campaign")
        != "aim3_tcga_surgen_primary_two_canonical_fixed_wt_controls"
        or contract.get("population") != "tcga_surgen_primary"
        or contract.get("encoder") != "UNI-v1"
        or contract.get("model_seeds") != list(verifier.MODEL_SEEDS)
        or contract.get("fine_tasks") != ["codon", "g12d_broad"]
        or contract.get("control_tasks") != ["ctrl_codon", "ctrl_g12d_broad"]
        or contract.get("fixed_wt_draw_seeds")
        != {"codon": 20260818, "g12d_broad": 20260822}
        or contract.get("fit_accounting") != expected_counts
        or contract.get("configured_max_parallel_chains") != 6
        or contract.get("external_development_cohorts") != []
        or contract.get("nonselected_fixed_controls_scheduled") is not False
        or contract.get("repeated_wt_controls_scheduled") is not False
        or "no repeated-WT consensus or performance-bound claim"
        not in str(contract.get("claim_scope"))
    ):
        raise verifier.BundleVerificationError("two-fixed campaign contract drift")
    _require_link(contract.get("controller"), REPO / "aim3_tcga_surgen_primary_two_fixed_controls_campaign.py", context="two-fixed controller")
    _require_link(contract.get("focused_test"), REPO / "tests/test_aim3_tcga_surgen_primary_two_fixed_controls_campaign.py", context="two-fixed test")

    preflight, _ = _source_json(
        paths, by_id, "aim3-source-primary-two-fixed-preflight"
    )
    scheduler, scheduler_path = _source_json(
        paths, by_id, "aim3-source-primary-two-fixed-scheduler"
    )
    training, training_path = _source_json(
        paths, by_id, "aim3-source-primary-two-fixed-training-completion"
    )
    results, results_path = _source_json(
        paths, by_id, "aim3-source-primary-two-fixed-results"
    )
    completion, _ = _source_json(
        paths, by_id, "aim3-source-primary-two-fixed-analysis-completion"
    )
    if (
        preflight.get("status") != "PASS"
        or preflight.get("fit_accounting") != expected_counts
        or scheduler.get("status") != "completed_rc0"
        or scheduler.get("job_count") != 10
        or scheduler.get("fit_accounting") != expected_counts
        or scheduler.get("configured_max_parallel_chains") != 6
        or scheduler.get("observed_max_parallel_chains") != 6
        or scheduler.get("nonselected_fixed_or_repeated_jobs_dispatched") != 0
        or len(scheduler.get("events", ())) != 10
        or any(event.get("returncode") != 0 for event in scheduler["events"])
        or training.get("status") != "complete_and_certified"
        or training.get("fit_accounting") != expected_counts
        or len(training.get("continuation_job_receipts", ())) != 10
        or training.get("repeated_wt_controls_scheduled") is not False
        or results.get("status") != "complete"
        or results.get("control_kind") != "canonical_fixed_matched_wt"
        or results.get("repeated_wt_controls_scheduled") is not False
        or results.get("fit_accounting") != expected_counts
        or completion.get("status") != "complete_and_verified"
        or completion.get("report_binding_count") != 6
        or completion.get("repeated_wt_controls_scheduled") is not False
    ):
        raise verifier.BundleVerificationError("two-fixed terminal evidence drift")
    _require_link(results.get("contract"), contract_path, context="two-fixed results contract")
    _require_link(results.get("training_seal"), training_path, context="two-fixed results training seal")
    _require_link(completion.get("results"), results_path, context="two-fixed completion results")

    logits_record = by_id["aim3-source-primary-two-fixed-patient-native-logits"]
    bootstrap_record = by_id["aim3-source-primary-two-fixed-bootstrap-distributions"]
    logits_path = _source_path(paths, logits_record)
    bootstrap_path = _source_path(paths, bootstrap_record)
    _require_link(results.get("patient_native_logits"), logits_path, context="two-fixed patient logits")
    _require_link(results.get("bootstrap_distributions"), bootstrap_path, context="two-fixed bootstrap")
    _require_link(completion.get("patient_native_logits"), logits_path, context="two-fixed completion patient logits")
    _require_link(completion.get("bootstrap_distributions"), bootstrap_path, context="two-fixed completion bootstrap")
    frame = pd.read_parquet(logits_path)
    if (
        len(frame) != 2004
        or set(frame["rung"]) != {"codon", "g12d_broad"}
        or set(frame["section"]) != {"fine", "canonical_fixed_matched_wt"}
        or frame[["rung", "section", "patient_id"]].duplicated().any()
    ):
        raise verifier.BundleVerificationError("two-fixed patient-logit roster drift")

    rungs = results.get("canonical_fixed_matched_wt", {}).get("rungs")
    if not isinstance(rungs, dict) or set(rungs) != {"codon", "g12d_broad"}:
        raise verifier.BundleVerificationError("two-fixed result rung roster drift")
    bindings: list[dict[str, Any]] = []
    with np.load(bootstrap_path, allow_pickle=False) as bootstrap:
        expected_arrays = {
            f"{rung}__{metric}"
            for rung in ("codon", "g12d_broad")
            for metric in (
                "fine_auroc",
                "control_auroc",
                "delta_control_minus_fine",
            )
        }
        if set(bootstrap.files) != expected_arrays:
            raise verifier.BundleVerificationError("two-fixed bootstrap roster drift")
        for rung, positive, negative, draw_seed in (
            ("codon", 354, 147, 20260818),
            ("g12d_broad", 154, 347, 20260822),
        ):
            record = rungs[rung]
            if (
                record.get("patients_per_arm") != 501
                or record.get("positive_per_arm") != positive
                or record.get("negative_per_arm") != negative
                or record.get("canonical_fixed_wt_draw_seed") != draw_seed
                or record.get("claim")
                != "conditional canonical fixed matched-WT contrast only"
            ):
                raise verifier.BundleVerificationError(f"two-fixed census drift: {rung}")
            prefix = f"aim3_source.canonical_fixed.{rung}"
            for name, value in (
                ("patients", 501),
                ("positive", positive),
                ("negative", negative),
                ("draw_seed", draw_seed),
            ):
                bindings.append({"id": f"{prefix}.{name}", "value": value})
            points: dict[str, float] = {}
            for section, result_name, npz_name in (
                ("fine", "fine_five_seed_ensemble_auroc", "fine_auroc"),
                (
                    "canonical_fixed_matched_wt",
                    "control_five_seed_ensemble_auroc",
                    "control_auroc",
                ),
            ):
                subset = frame[(frame["rung"] == rung) & (frame["section"] == section)]
                if len(subset) != 501 or int(subset["label"].sum()) != positive:
                    raise verifier.BundleVerificationError(f"two-fixed frame census drift: {rung}")
                point = _auc(subset["label"], subset["mean_logit"])
                summary = record[result_name]
                _close(summary.get("estimate"), point, context=f"two-fixed {rung} {npz_name}")
                ci = _interval(summary.get("ci95_two_sided"), context=f"two-fixed {rung} {npz_name}")
                array = bootstrap[f"{rung}__{npz_name}"]
                if array.shape != (10_000,):
                    raise verifier.BundleVerificationError("two-fixed bootstrap length drift")
                quantiles = tuple(float(value) for value in np.quantile(array, [0.025, 0.975]))
                for observed, expected in zip(ci, quantiles, strict=True):
                    _close(observed, expected, context=f"two-fixed {rung} {npz_name} CI")
                points[npz_name] = point
                for suffix, value in (
                    (npz_name, point),
                    (f"{npz_name}.ci95_lower", ci[0]),
                    (f"{npz_name}.ci95_upper", ci[1]),
                ):
                    bindings.append({"id": f"{prefix}.{suffix}", "value": value})
            delta = record["delta_control_minus_fine_auroc"]
            expected_delta = points["control_auroc"] - points["fine_auroc"]
            _close(delta.get("estimate"), expected_delta, context=f"two-fixed {rung} delta")
            delta_ci = _interval(delta.get("ci95_two_sided"), context=f"two-fixed {rung} delta")
            delta_array = bootstrap[f"{rung}__delta_control_minus_fine"]
            delta_quantiles = tuple(
                float(value) for value in np.quantile(delta_array, [0.025, 0.975])
            )
            for observed, expected in zip(delta_ci, delta_quantiles, strict=True):
                _close(observed, expected, context=f"two-fixed {rung} delta CI")
            for suffix, value in (
                ("delta_control_minus_fine_auroc", expected_delta),
                ("delta_control_minus_fine_auroc.ci95_lower", delta_ci[0]),
                ("delta_control_minus_fine_auroc.ci95_upper", delta_ci[1]),
            ):
                bindings.append({"id": f"{prefix}.{suffix}", "value": value})
    if len(bindings) != 26:
        raise verifier.BundleVerificationError("two-fixed binding count drift")
    return results, bindings


def _procedure_aurocs(frame: Any, score: str) -> dict[str, list[float]]:
    values = {scope: [] for scope in ADAPTATION_SCOPES}
    for (_layout, _draw), procedure in frame.groupby(
        ["layout_seed", "draw"], sort=True
    ):
        rih = procedure[procedure["test_cohort"] == "RIH"]
        surgen = procedure[procedure["test_cohort"] == "SurGen"]
        rih_auc = _auc(rih["label"], rih[score])
        surgen_auc = _auc(surgen["label"], surgen[score])
        values["RIH-M"].append(rih_auc)
        values["SurGen-M"].append(surgen_auc)
        values["pooled_combined"].append(_auc(procedure["label"], procedure[score]))
        values["equal_cohort_macro"].append((rih_auc + surgen_auc) / 2)
    if any(len(scope_values) != 100 for scope_values in values.values()):
        raise verifier.BundleVerificationError("few-shot procedure count drift")
    return values


def _mean_sd(values: list[float]) -> tuple[float, float]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def _adaptation_accounting_bindings(
    *, method: str, accounting: dict[str, Any]
) -> list[dict[str, Any]]:
    if method == "pure_ridge_linear_probe":
        mapping = (
            ("embedding_artifacts", "label_blind_embedding_jobs"),
            ("support_fold_procedures", "unique_support_procedures_by_fold"),
            ("final_heads", "final_probe_head_decisions"),
            ("test_applications", "fit_to_test_cohort_applications"),
            ("solver_calls", "inner_plus_final_solver_calls"),
            ("source_model_fits", "source_main_model_fits"),
            ("other_method_fits", "residual_adapter_fits"),
            ("local_mil_fits", "local_mil_fits"),
            ("full_label_fits", "full_label_fits"),
            ("platt_fits", "platt_fits"),
        )
    else:
        mapping = (
            ("embedding_artifacts", "reused_sealed_embedding_artifacts"),
            ("support_fold_procedures", "unique_support_procedures_by_fold"),
            ("final_heads", "final_residual_head_decisions"),
            ("test_applications", "fit_to_test_cohort_applications"),
            ("solver_calls", "inner_plus_final_solver_calls"),
            ("source_model_fits", "source_main_model_fits"),
            ("other_method_fits", "pure_probe_fits"),
            ("local_mil_fits", "local_mil_fits"),
            ("full_label_fits", "full_label_fits"),
            ("platt_fits", "platt_fits"),
        )
    if any(key not in accounting for _, key in mapping):
        raise verifier.BundleVerificationError(f"{method} accounting field drift")
    return [
        {
            "id": f"aim2_target_internal.{method}.accounting.{label}",
            "value": accounting[key],
        }
        for label, key in mapping
    ]


def _validate_adaptation_campaign(
    paths: verifier.BundlePaths,
    by_id: dict[str, dict[str, Any]],
    *,
    family: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    import pandas as pd

    if family == "pure-ridge":
        prefix = "aim2-source-anchored-pure-ridge"
        root = PURE_RIDGE_ROOT
        method = "pure_ridge_linear_probe"
        score = "eta_pure_ridge"
        expected_contract_status = "PREPARED_LABEL_BLIND_LEAN_PURE_RIDGE"
        expected_result_status = "COMPLETE_LEAN_COMBINED_MET_PURE_RIDGE_FEWSHOT"
        expected_scheduler_status = "COMPLETE_LEAN_PURE_RIDGE_SCHEDULER"
        scheduler_name = "scheduler"
        scheduler_path = root / "receipts/pure_ridge_scheduler.json"
        oof_path = root / "adaptation/pure_ridge_oof.parquet"
        expected_phase = "lean_pure_ridge_few_shot"
        accounting = {
            "label_blind_embedding_jobs": 10,
            "unique_support_procedures_by_fold": 2000,
            "final_probe_head_decisions": 10000,
            "fit_to_test_cohort_applications": 20000,
            "inner_plus_final_solver_calls": 1330000,
            "source_main_model_fits": 0,
            "residual_adapter_fits": 0,
            "local_mil_fits": 0,
            "full_label_fits": 0,
            "platt_fits": 0,
        }
    else:
        prefix = "aim2-source-anchored-residual"
        root = RESIDUAL_RIDGE_ROOT
        method = "source_anchored_residual_ridge"
        score = "eta_residual_ridge"
        expected_contract_status = "PREPARED_SEALED_UPSTREAM_RESIDUAL_HEAD"
        expected_result_status = "COMPLETE_COMBINED_MET_SOURCE_ANCHORED_RESIDUAL_FEWSHOT"
        expected_scheduler_status = "COMPLETE_LEAN_SOURCE_ANCHORED_RESIDUAL_SCHEDULER"
        scheduler_name = "scheduler"
        scheduler_path = root / "receipts/residual_scheduler.json"
        oof_path = root / "adaptation/residual_ridge_oof.parquet"
        expected_phase = "lean_source_anchored_residual_few_shot"
        accounting = {
            "reused_sealed_embedding_artifacts": 10,
            "unique_support_procedures_by_fold": 2000,
            "final_residual_head_decisions": 10000,
            "fit_to_test_cohort_applications": 20000,
            "inner_plus_final_solver_calls": 1330000,
            "source_main_model_fits": 0,
            "pure_probe_fits": 0,
            "local_mil_fits": 0,
            "full_label_fits": 0,
            "platt_fits": 0,
        }
    contract, contract_path = _source_json(paths, by_id, f"{prefix}-contract")
    results, results_path = _source_json(paths, by_id, f"{prefix}-results")
    scheduler, observed_scheduler_path = _source_json(
        paths, by_id, f"{prefix}-{scheduler_name}"
    )
    completion, _ = _source_json(
        paths, by_id, f"{prefix}-analysis-completion"
    )
    if observed_scheduler_path.absolute() != scheduler_path.absolute():
        raise verifier.BundleVerificationError(f"{family} scheduler path drift")
    protocol = contract.get("protocol")
    firewall = contract.get("firewall")
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != expected_contract_status
        or contract.get("output_root") != str(root)
        or not isinstance(protocol, dict)
        or protocol.get("method") != method
        or protocol.get("source_models_frozen") is not True
        or protocol.get("source_model_seeds")
        not in (None, list(verifier.MODEL_SEEDS))
        or protocol.get("support_per_class_total") != list(ADAPTATION_BUDGETS)
        or protocol.get("layout_seeds") != list(ADAPTATION_LAYOUT_SEEDS)
        or protocol.get("draws_per_layout") != 20
        or protocol.get("procedures_per_budget") != 100
        or protocol.get("outer_folds") != [0, 1, 2, 3, 4]
        or protocol.get("bootstrap_draws") != 10_000
        or protocol.get("combined_support_balance")
        != "exactly k/2 patients from each cohort within each outcome class"
        or not isinstance(firewall, dict)
        or firewall.get("result_role") != TARGET_INTERNAL_ROLE
        or firewall.get("external_validation_claim_permitted") is not False
        or firewall.get("combined_support_only") is not True
        or firewall.get("primary_support_permitted") is not False
        or firewall.get("surgen_metastatic_source_family_exposed") is not True
        or firewall.get("all_budgets_reported") is not True
    ):
        raise verifier.BundleVerificationError(f"{family} campaign contract drift")
    observed_accounting = contract.get("accounting")
    if not isinstance(observed_accounting, dict) or any(
        observed_accounting.get(key) != value for key, value in accounting.items()
    ):
        raise verifier.BundleVerificationError(f"{family} contract accounting drift")
    if (
        scheduler.get("schema_version") != 1
        or scheduler.get("status") != expected_scheduler_status
        or scheduler.get("configured_max_workers") != 6
        or scheduler.get("observed_peak_workers") != 6
        or scheduler.get("configured_child_workers") != 1
        or scheduler.get("shard_jobs") != 20
        or len(scheduler.get("events", ())) != 20
        or any(event.get("returncode") != 0 for event in scheduler["events"])
        or any(scheduler.get("accounting", {}).get(key) != value for key, value in accounting.items())
    ):
        raise verifier.BundleVerificationError(f"{family} scheduler evidence drift")
    if (
        results.get("schema_version") != 1
        or results.get("status") != expected_result_status
        or results.get("result_role") != TARGET_INTERNAL_ROLE
        or results.get("external_validation_claim_permitted") is not False
        or results.get("few_shot", {}).get("methods") != [method]
        or results.get("few_shot", {}).get("support_regimes") != ["COMBINED"]
        or results.get("few_shot", {}).get("support_per_class_total")
        != list(ADAPTATION_BUDGETS)
        or results.get("few_shot", {}).get(
            "method_regime_or_budget_selected_on_target_results"
        )
        is not False
        or results.get("few_shot", {}).get("positive_repair_claim_permitted")
        is not False
        or any(results.get("fit_and_solver_accounting", {}).get(key) != value for key, value in accounting.items())
        or completion.get("status") != expected_result_status
        or completion.get("result_role") != TARGET_INTERNAL_ROLE
        or completion.get("external_validation_claim_permitted") is not False
        or any(completion.get("accounting", {}).get(key) != value for key, value in accounting.items())
    ):
        raise verifier.BundleVerificationError(f"{family} result/completion drift")
    _require_link(completion.get("contract"), contract_path, context=f"{family} completion contract")
    _require_link(completion.get("scheduler"), scheduler_path, context=f"{family} completion scheduler")
    _require_link(completion.get("aggregate_oof"), oof_path, context=f"{family} completion OOF")
    _require_link(completion.get("results"), results_path, context=f"{family} completion results")

    frame = pd.read_parquet(oof_path)
    expected_columns = {
        "phase",
        "layout_seed",
        "support_per_class",
        "support_total",
        "draw",
        "test_cohort",
        "patient_id",
        "label",
        "fold",
        "eta_native",
        score,
        *(f"eta_native_seed{seed}" for seed in verifier.MODEL_SEEDS),
        *(f"{score}_seed{seed}" for seed in verifier.MODEL_SEEDS),
    }
    key_columns = [
        "layout_seed",
        "support_per_class",
        "draw",
        "test_cohort",
        "patient_id",
    ]
    if (
        len(frame) != 63_600
        or set(frame.columns) != expected_columns
        or set(frame["phase"]) != {expected_phase}
        or set(frame["layout_seed"]) != set(ADAPTATION_LAYOUT_SEEDS)
        or set(frame["support_per_class"]) != set(ADAPTATION_BUDGETS)
        or set(frame["draw"]) != set(range(20))
        or set(frame["test_cohort"]) != {"RIH", "SurGen"}
        or frame[key_columns].duplicated().any()
        or not np.isfinite(frame.select_dtypes(include=["number"]).to_numpy()).all()
    ):
        raise verifier.BundleVerificationError(f"{family} aggregate OOF roster drift")
    census = (
        frame[["test_cohort", "patient_id", "label"]]
        .drop_duplicates()
        .groupby("test_cohort")
        .agg(patients=("patient_id", "size"), mutant=("label", "sum"))
    )
    if census.to_dict("index") != {
        "RIH": {"patients": 85, "mutant": 37},
        "SurGen": {"patients": 74, "mutant": 30},
    }:
        raise verifier.BundleVerificationError(f"{family} target census drift")

    native_results = results.get("native_zero_shot", {}).get("scopes")
    if not isinstance(native_results, dict) or set(native_results) != set(ADAPTATION_SCOPES):
        raise verifier.BundleVerificationError(f"{family} native scope roster drift")
    native_procedures = _procedure_aurocs(
        frame[frame["support_per_class"] == 2], "eta_native"
    )
    native_points = {
        scope: _mean_sd(values)[0] for scope, values in native_procedures.items()
    }
    bindings: list[dict[str, Any]] = []
    for scope in ADAPTATION_SCOPES:
        block = native_results[scope]
        expected_census = (
            (85, 37)
            if scope == "RIH-M"
            else (74, 30)
            if scope == "SurGen-M"
            else (159, 67)
        )
        if block.get("census", {}).get("patients") != expected_census[0] or block.get(
            "census", {}
        ).get("mutant") != expected_census[1]:
            raise verifier.BundleVerificationError(f"{family} {scope} census drift")
        _close(block.get("auroc"), native_points[scope], context=f"{family} {scope} native")
        ci = _interval(block.get("ci95"), context=f"{family} {scope} native")
        scope_id = scope.lower().replace("-", "_")
        for label, value in (
            ("patients", expected_census[0]),
            ("mutant", expected_census[1]),
            ("auroc", native_points[scope]),
            ("ci95_lower", ci[0]),
            ("ci95_upper", ci[1]),
        ):
            bindings.append(
                {"id": f"aim2_target_internal.native.{scope_id}.{label}", "value": value}
            )
        published_seed_block = (
            results.get("few_shot", {})
            .get("cells", {})
            .get("2", {})
            .get("per_source_seed_expected_auroc", {})
            .get("native", {})
            .get(scope, {})
        )
        observed_by_seed = published_seed_block.get("by_seed")
        replayed_by_seed: dict[str, float] = {}
        for seed in verifier.MODEL_SEEDS:
            replayed_by_seed[str(seed)] = _mean_sd(
                _procedure_aurocs(
                    frame[frame["support_per_class"] == 2],
                    f"eta_native_seed{seed}",
                )[scope]
            )[0]
        if not isinstance(observed_by_seed, dict) or set(observed_by_seed) != set(
            replayed_by_seed
        ):
            raise verifier.BundleVerificationError(
                f"{family} {scope} native source-seed replay drift"
            )
        for seed, replayed in replayed_by_seed.items():
            _close(
                observed_by_seed[seed],
                replayed,
                context=f"{family} {scope} native source seed {seed}",
            )
        for seed, value in replayed_by_seed.items():
            bindings.append(
                {
                    "id": f"aim2_target_internal.native.{scope_id}.source_seed_{seed}",
                    "value": value,
                }
            )

    cells = results.get("few_shot", {}).get("cells")
    if not isinstance(cells, dict) or set(cells) != {str(k) for k in ADAPTATION_BUDGETS}:
        raise verifier.BundleVerificationError(f"{family} budget roster drift")
    for budget in ADAPTATION_BUDGETS:
        budget_frame = frame[frame["support_per_class"] == budget]
        cell = cells[str(budget)]
        if (
            cell.get("support_per_class_total") != budget
            or cell.get("support_total") != 2 * budget
            or cell.get("procedures") != 100
            or cell.get("bootstrap_draws") != 10_000
            or cell.get("combined_support_balance")
            != "k/2 per cohort within each class"
        ):
            raise verifier.BundleVerificationError(f"{family} k={budget} cell drift")
        point_values = _procedure_aurocs(budget_frame, score)
        seed_replays: dict[str, dict[str, float]] = {
            scope: {} for scope in ADAPTATION_SCOPES
        }
        for seed in verifier.MODEL_SEEDS:
            seed_values = _procedure_aurocs(budget_frame, f"{score}_seed{seed}")
            for scope in ADAPTATION_SCOPES:
                seed_replays[scope][str(seed)] = _mean_sd(seed_values[scope])[0]
        for scope in ADAPTATION_SCOPES:
            block = cell.get("scopes", {}).get(scope)
            if not isinstance(block, dict):
                raise verifier.BundleVerificationError(f"{family} k={budget} {scope} absent")
            point, procedure_sd = _mean_sd(point_values[scope])
            method_block = block.get(method)
            gain_block = block.get(f"{method}_minus_native")
            if not isinstance(method_block, dict) or not isinstance(gain_block, dict):
                raise verifier.BundleVerificationError(f"{family} k={budget} {scope} method drift")
            _close(method_block.get("auroc"), point, context=f"{family} k={budget} {scope} point")
            _close(method_block.get("procedure_sample_sd"), procedure_sd, context=f"{family} k={budget} {scope} procedure SD")
            ci = _interval(method_block.get("ci95"), context=f"{family} k={budget} {scope} CI")
            gain = point - native_points[scope]
            _close(gain_block.get("auroc_gain"), gain, context=f"{family} k={budget} {scope} gain")
            gain_ci = _interval(gain_block.get("ci95"), context=f"{family} k={budget} {scope} gain CI")
            seed_block = cell.get("per_source_seed_expected_auroc", {}).get(method, {}).get(scope)
            observed_seed_values = (
                seed_block.get("by_seed") if isinstance(seed_block, dict) else None
            )
            if not isinstance(observed_seed_values, dict) or set(
                observed_seed_values
            ) != set(seed_replays[scope]):
                raise verifier.BundleVerificationError(f"{family} k={budget} {scope} seed replay drift")
            for seed, replayed in seed_replays[scope].items():
                _close(
                    observed_seed_values[seed],
                    replayed,
                    context=f"{family} k={budget} {scope} source seed {seed}",
                )
            seed_mean, seed_sd = _mean_sd(list(seed_replays[scope].values()))
            _close(seed_block.get("mean"), seed_mean, context=f"{family} k={budget} {scope} seed mean")
            _close(seed_block.get("sample_sd"), seed_sd, context=f"{family} k={budget} {scope} seed SD")
            scope_id = scope.lower().replace("-", "_")
            binding_prefix = f"aim2_target_internal.{method}.k{budget}.{scope_id}"
            for label, value in (
                ("auroc", point),
                ("procedure_sample_sd", procedure_sd),
                ("ci95_lower", ci[0]),
                ("ci95_upper", ci[1]),
                ("source_seed_mean", seed_mean),
                ("source_seed_sample_sd", seed_sd),
                ("gain", gain),
                ("gain_ci95_lower", gain_ci[0]),
                ("gain_ci95_upper", gain_ci[1]),
            ):
                bindings.append({"id": f"{binding_prefix}.{label}", "value": value})
    bindings.extend(_adaptation_accounting_bindings(method=method, accounting=accounting))
    if len(bindings) != 194:
        raise verifier.BundleVerificationError(f"{family} binding count drift")
    return results, bindings


def validate_incremental_evidence(
    paths: verifier.BundlePaths,
    sources: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate all new terminal bytes and independently replay point metrics."""

    by_id = _require_incremental_identity_roster(paths, sources)
    fixed, fixed_bindings = _validate_two_fixed(paths, by_id)
    pure, pure_bindings = _validate_adaptation_campaign(
        paths, by_id, family="pure-ridge"
    )
    residual, residual_bindings = _validate_adaptation_campaign(
        paths, by_id, family="residual-ridge"
    )
    pure_native = pure.get("native_zero_shot", {}).get("scopes", {})
    residual_native = residual.get("native_zero_shot", {}).get("scopes", {})
    for scope in ADAPTATION_SCOPES:
        for key in ("census", "auroc", "ci95"):
            if pure_native.get(scope, {}).get(key) != residual_native.get(
                scope, {}
            ).get(key):
                raise verifier.BundleVerificationError(
                    "pure/residual frozen native comparator drift"
                )
    target_bindings = [*pure_bindings, *residual_bindings]
    # Native comparator values are identical; retain one exact 20-value copy.
    native_prefix = "aim2_target_internal.native."
    target_bindings = [
        *[binding for binding in pure_bindings if binding["id"].startswith(native_prefix)],
        *[binding for binding in pure_bindings if not binding["id"].startswith(native_prefix)],
        *[binding for binding in residual_bindings if not binding["id"].startswith(native_prefix)],
    ]
    if len(target_bindings) != 348 or len(
        {binding["id"] for binding in target_bindings}
    ) != 348:
        raise verifier.BundleVerificationError("target-internal binding roster drift")

    source_root = Path(paths.expected_aim3_run_root)
    fixed_dirs = source_root / "train/fixed"
    observed_fixed = (
        {path.name for path in fixed_dirs.iterdir() if path.is_dir()}
        if fixed_dirs.is_dir()
        else set()
    )
    repeated_root = source_root / "train/repeated"
    observed_repeated = (
        [path for path in repeated_root.iterdir()] if repeated_root.is_dir() else []
    )
    if observed_fixed != {"ctrl_codon", "ctrl_g12d_broad"} or observed_repeated:
        raise verifier.BundleVerificationError(
            "Aim-3 source contains nonselected fixed or repeated control training"
        )
    allowed_job_keys = {
        f"fixed__{task}__seed{seed}"
        for task in ("ctrl_codon", "ctrl_g12d_broad")
        for seed in verifier.MODEL_SEEDS
    }
    for relative, suffix in (
        ("requests/jobs", ".json"),
        ("logs/jobs", ".log"),
        ("receipts/jobs", ".json"),
        ("state/hydra", ""),
    ):
        directory = source_root / relative
        observed = {
            path.name.removesuffix(suffix)
            for path in directory.iterdir()
            if path.name.startswith(("fixed__", "repeated__"))
            and ((path.is_file() and suffix) or (path.is_dir() and not suffix))
        } if directory.is_dir() else set()
        if observed != allowed_job_keys:
            raise verifier.BundleVerificationError(
                f"Aim-3 source selected-control namespace drift: {relative}"
            )
    failures = source_root / "requests/failures"
    if failures.is_dir() and any(
        path.name.startswith(("fixed__", "repeated__"))
        for path in failures.iterdir()
    ):
        raise verifier.BundleVerificationError(
            "Aim-3 source contains a selected/nonselected control failure artifact"
        )
    return {
        "two_fixed": fixed,
        "two_fixed_bindings": fixed_bindings,
        "pure_ridge": pure,
        "residual_ridge": residual,
        "target_internal_bindings": target_bindings,
        "incremental_source_count": len(EXPECTED_INCREMENTAL_IDENTITIES),
    }


def _firewall_block() -> str:
    rows = "\n".join(
        f"| {cohort} | {role} | frozen_model_zero_shot_only |"
        for cohort, role in verifier.EXTERNAL_TARGET_ROLES.items()
    )
    return "\n".join(
        (
            *verifier.FIREWALL_MARKERS,
            "",
            "| Cohort | FINAL-v13 role | Permitted use in main-model analysis |",
            "|---|---|---|",
            rows,
        )
    )


def _priority_1_setup() -> str:
    return f"""{verifier.PRIORITY_HEADINGS[0]}

The main development population is exactly 1,239 conventional-primary patients
(501 KRAS-mutant, 738 wild type; 1,389 slides) from TCGA-COAD, TCGA-READ,
SR386, and SR1482. UNI-v1 and the Virchow2-CLS encoder sensitivity use shared
patient-stratified five-fold layouts and seeds 42--46. OOF native logits define
development performance; one source-only p75 refit per seed and encoder exists
only for frozen target scoring.

The source-compliant whole Aim-1 pipeline comprises E0 gene-level KRAS OOF
ranking and the paired encoder contrast; E1a fixed-score molecular, site,
stage, and sidedness restrictions; E1a-S common-support composition
standardization; four zero-MIL-fit Why-D diagnostics; and E1d cross-fitted
routine-clinical and fixed-fusion comparators. E1v/cap, E1e, worklist/DCA, and
extended-RAS/MAPK/pathway-quiet branches remain named provenance, but inherited
numeric rows that use ALL-primary or three-seed/non-source lineages are
ineligible here. E1b did not fire; E1c and E0b were not run."""


def _priority_2_setup() -> str:
    return f"""{verifier.PRIORITY_HEADINGS[1]}

Five full-source TCGA+SurGen-primary refits per encoder were frozen before
outcome joins, then scored CPTAC-primary, Orion, RIH-primary, RIH-metastatic,
and SurGen-metastatic in a 2-encoder x 5-seed x 5-target grid: 50 label-blind
score files and 4,790 slide rows. The post-outcome construction sensitivity
also reuses each seed's five source-CV best checkpoints (250 label-blind
checkpoint-target passes; zero fits) to compare p75 refit with a within-seed
fold5 mean-native-logit ensemble. That comparison cannot select a deployment
construction. Orion is retrospective; SurGen-metastatic is source-family
exposed because SR1482 primary participates in source training."""


def _priority_3_setup() -> str:
    return f"""{verifier.PRIORITY_HEADINGS[2]}

The inherited family and sibling LOCO panels deliberately train on multiple
cohorts, including cohorts otherwise reserved as targets. They are isolated
robustness characterizations and cannot train, tune, or select the FINAL-v13
main model."""


def _priority_4_setup() -> str:
    return f"""{verifier.PRIORITY_HEADINGS[3]}

Two lean combined-metastatic few-shot analyses reuse the frozen UNI-v1 p75
refits from the TCGA+SurGen-primary Aim-1 source model. Support contains exactly
k/2 RIH-M and k/2 SurGen-M patients in each outcome class for k=2,4,8,16 per
class total, across five fixed layouts and 20 draws per layout. The permitted
methods are a pure ridge linear probe and a source-anchored residual ridge
head. Target labels are used inside RIH-M and SurGen-M; therefore every adapted
row is TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION, SurGen-M is source-family
exposed, and no result may feed back to the source model or any external claim.
Legacy non-source-anchored E2c/E2e/E2f values remain provenance-only."""


def _priority_5_setup() -> str:
    return f"""{verifier.PRIORITY_HEADINGS[4]}

The controlling population is TCGA+SurGen primaries only and the encoder is
UNI-v1. Phase 1 runs exactly the five fine tasks (codon, G12D-broad,
G12D-within-G12, G12V, and G12C), each at seeds 42--46: 25 chains, 125 OOF
fold fits, 25 p75 refits, and 150 physical MIL fits, with six parallel chains.
The 25 frozen p75 refits then generated 25 label-blind score files over 479
external slides with zero additional fits and observed six-worker inference;
the inference seal preceded every target outcome join. A separate append-only
continuation completed exactly two predeclared canonical fixed matched-WT
contrasts (codon and G12D-broad): 10 chains, 50 OOF fits, 10 refits, and 60
physical MIL fits at configured and observed peak six. Each is one conditional
single-draw contrast, not a consensus or performance ceiling. The other three
canonical controls and every repeated-WT draw remain pending."""


def _priority_6_setup() -> str:
    return f"""{verifier.PRIORITY_HEADINGS[5]}

Aim 4 inherits the k=32 prototype statistics and the completed human montage
read. The vocabulary was fit label-blind on a union that included target
cohorts, so this is legacy transductive interpretation rather than external
validation. Prototype associations and concordance are noncausal and based on
a single reader."""


def render_experimental_setup() -> str:
    return "\n\n".join(
        (
            "# FINAL-v13 experimental setup",
            "FINAL-v13 is an additive child of sealed FINAL-v12.1. This "
            "incremental snapshot is verified and explicitly unsealed: fine "
            "plus external scoring, two fixed controls, and two target-internal "
            "few-shot methods are complete while the remaining controls are pending.",
            _firewall_block(),
            _priority_1_setup(),
            _priority_2_setup(),
            _priority_3_setup(),
            _priority_4_setup(),
            _priority_5_setup(),
            _priority_6_setup(),
        )
    ) + "\n"


def _fine_table(fine: dict[str, Any]) -> str:
    rungs = fine["fine"]["rungs"]
    rows = []
    for rung in FINE_DISPLAY_ORDER:
        record = rungs[rung]
        estimate = float(record["five_seed_ensemble"]["estimate"])
        rows.append(
            f"| {FINE_DISPLAY_NAMES[rung]} | {record['patients']} "
            f"({record['positive']}/{record['negative']}) | {estimate:.4f} |"
        )
    return "\n".join(
        (
            "| Fine molecular task | Patients (positive/negative) | "
            "Five-seed patient-native-logit AUROC |",
            "|---|---:|---:|",
            *rows,
        )
    )


FINE_EXTERNAL_SCOPE_NAMES = {
    "cptac_primary": "CPTAC-primary",
    "orion_primary": "Orion-primary",
    "rih_primary": "RIH-primary",
    "rih_metastatic": "RIH-metastatic",
    "surgen_metastatic": "SurGen-metastatic",
    "primary_only": "Primary-only",
    "metastatic_only": "Met-only",
    "strict_disjoint_combined": "Combined (strict-disjoint)",
}


def _fine_external_internal_table(fine_external: dict[str, Any]) -> str:
    rows = []
    for task in FINE_DISPLAY_ORDER:
        block = fine_external["tasks"][task]["internal_oof"]
        ensemble = block["five_seed_oof_ensemble"]
        rows.append(
            f"| {FINE_DISPLAY_NAMES[task]} | {block['patients']} "
            f"({block['positive']}/{block['negative']}) | "
            f"{block['seed_auroc_mean']:.4f} ± "
            f"{block['seed_auroc_sample_sd']:.4f} | "
            f"{block['fold_auroc_mean']:.4f} ± "
            f"{block['fold_auroc_sample_sd']:.4f} | "
            f"{ensemble['auroc']:.4f} [{ensemble['ci95'][0]:.4f}, "
            f"{ensemble['ci95'][1]:.4f}] |"
        )
    return "\n".join(
        (
            "| Fine molecular task | Internal patients (positive/negative) | "
            "Five seed-specific pooled-OOF AUROCs, mean ± SD | Five held-out "
            "fold ensemble AUROCs, mean ± SD | Five-seed OOF ensemble AUROC "
            "[95% CI] |",
            "|---|---:|---:|---:|---:|",
            *rows,
        )
    )


def _fine_external_refit_table(fine_external: dict[str, Any]) -> str:
    rows = []
    for task in FINE_DISPLAY_ORDER:
        external = fine_external["tasks"][task]["external_refit"]
        for scope in verifier.AIM3_FINE_EXTERNAL_REPORT_SCOPES:
            block = (
                external["per_cohort"][scope]
                if scope in verifier.AIM3_FINE_EXTERNAL_TARGETS
                else external[scope]
            )
            ensemble = block["five_seed_refit_ensemble"]
            rows.append(
                f"| {FINE_DISPLAY_NAMES[task]} | {FINE_EXTERNAL_SCOPE_NAMES[scope]} | "
                f"{block['n_records']} ({block['positive']}/{block['negative']}) | "
                f"{block['seed_auroc_mean']:.4f} ± "
                f"{block['seed_auroc_sample_sd']:.4f} | "
                f"{ensemble['auroc']:.4f} [{ensemble['ci95'][0]:.4f}, "
                f"{ensemble['ci95'][1]:.4f}] | {block['support']} |"
            )
    return "\n".join(
        (
            "| Fine molecular task | Frozen external/test scope | Patients "
            "(positive/negative) | Five refit-seed AUROCs, mean ± SD | "
            "Five-refit mean-native-logit AUROC [pointwise 95% CI] | Support |",
            "|---|---|---:|---:|---:|---|",
            *rows,
        )
    )


def _fine_external_binding_comments(fine_external: dict[str, Any]) -> str:
    return "\n".join(
        "<!-- AIM3_SOURCE_VALUE "
        f"{binding['id']} {verifier._canonical_number(binding['value'])} -->"
        for binding in verifier._fine_external_report_bindings(fine_external)
    )


def _binding_comments(fine: dict[str, Any]) -> str:
    return "\n".join(
        "<!-- AIM3_SOURCE_VALUE "
        f"{binding['id']} {verifier._canonical_number(binding['value'])} -->"
        for binding in fine["report_bindings"]
    )


def _priority_1_results() -> str:
    return f"""{verifier.PRIORITY_HEADINGS[0]}

### Whole Aim-1 pipeline summary

| Aim-1 component | Source-compliant scope | Authenticated result or eligibility |
|---|---|---|
| E0 main/baseline | TCGA+SurGen-primary OOF, seeds 42--46 | UNI-v1 AUROC 0.6929 [0.6619, 0.7219], AUPRC 0.6007 [0.5636, 0.6413]; Virchow2-CLS AUROC 0.6657 [0.6345, 0.6957], AUPRC 0.5750 [0.5389, 0.6154] |
| Paired encoder sensitivity | Same 1,239 patients and shared bootstrap | Virchow2-minus-UNI AUROC -0.0272 [-0.0487, -0.0058]; AUPRC -0.0257 [-0.0557, 0.0038] |
| E1a molecular restriction | Fixed OOF scores; D = MSS/pMMR plus BRAF-WT | D AUROC 0.7261 [0.6947, 0.7579] UNI and 0.7079 [0.6751, 0.7412] Virchow2; A-minus-D -0.0332 [-0.0496, -0.0178] and -0.0422 [-0.0599, -0.0255] |
| E1a-S composition standardization | Common-support subcohort x site weights | Standardized D-minus-A +0.0280 [0.0112, 0.0457] UNI and +0.0477 [0.0298, 0.0665] Virchow2 |
| Why-D | Four zero-MIL-fit diagnostics | Observed D-minus-A-complete exceeded 10,000 matched random restrictions for both encoders (`p_ge_observed=9.999e-05`); explanatory, not causal |
| E1d clinical value | Cross-fitted age/sex/site/stage and fixed fusion | WSI-minus-clinical +0.1527 [0.1167, 0.1889] UNI and +0.1255 [0.0885, 0.1626] Virchow2; fixed fusion did not improve WSI |
| E1v and cap robustness | Named provenance | Inherited ALL-primary numeric rows are ineligible and omitted |
| E1e positive control | Named provenance | Inherited non-source-lineage numeric result is ineligible and omitted |
| Worklist and DCA | Named provenance | Inherited three-seed/ALL-development numeric rows are ineligible and omitted |
| Extended-RAS/MAPK/pathway-quiet | Named provenance | Inherited three-seed/all-primary numeric rows are ineligible and omitted |
| Conditional branches | E1b, E1c, E0b, all-primary-plus-Orion | E1b did not fire; E1c/E0b were not run; all-primary-plus-Orion is audit-only |

The eligible Aim-1 evidence supports modest primary-tumor gene-level ranking,
noncausal molecular-context dependence, and incremental value over routine
clinical variables. It does not authorize target-informed model development,
late-fusion superiority, a universal threshold, causal morphology, or
replacement of molecular testing."""


def _priority_2_results() -> str:
    # The full reconciler is independently hash-pinned by the production
    # verifier; reuse its immutable source-only tables without duplicating 20
    # long seed vectors here.
    from tools import final_v13_full_reconciler as full

    return f"""{verifier.PRIORITY_HEADINGS[1]}

### Frozen five-seed external/test performance

{full._external_native_table()}

These are frozen source-only deployment-refit, five-seed mean-native-logit
results. CPTAC-primary and RIH-primary are family-naive zero-shot validation;
RIH-metastatic is a family-naive retrospective sensitivity; SurGen-metastatic
is source-family exposed; Orion is retrospective cross-protocol evidence. No
target result selected an encoder, model, construction, calibration, threshold,
or hyperparameter.

### Post-outcome refit-versus-within-seed-fold5 robustness

{full._external_construction_table()}

Vectors are seeds 42--46; each summary is arithmetic mean ± sample SD
(`ddof=1`). SD is descriptive seed/partition dispersion, not a confidence
interval. The comparison is post-outcome robustness only: all inference was
label-blind and added zero fits, but it jointly changes source training fraction,
checkpoint selection, and ensembling. It cannot choose a deployment
construction or contribute main-model development feedback."""


def _fixed_table(fixed: dict[str, Any]) -> str:
    rows = []
    for rung in ("codon", "g12d_broad"):
        record = fixed["canonical_fixed_matched_wt"]["rungs"][rung]
        fine = record["fine_five_seed_ensemble_auroc"]
        control = record["control_five_seed_ensemble_auroc"]
        delta = record["delta_control_minus_fine_auroc"]
        rows.append(
            f"| {FINE_DISPLAY_NAMES[rung]} | {record['patients_per_arm']} "
            f"({record['positive_per_arm']}/{record['negative_per_arm']}) | "
            f"{record['canonical_fixed_wt_draw_seed']} | "
            f"{fine['estimate']:.4f} [{fine['ci95_two_sided'][0]:.4f}, {fine['ci95_two_sided'][1]:.4f}] | "
            f"{control['estimate']:.4f} [{control['ci95_two_sided'][0]:.4f}, {control['ci95_two_sided'][1]:.4f}] | "
            f"{delta['estimate']:+.4f} [{delta['ci95_two_sided'][0]:.4f}, {delta['ci95_two_sided'][1]:.4f}] |"
        )
    return "\n".join(
        (
            "| Task | Patients/arm (positive/negative) | Canonical draw seed | Fine AUROC [95% CI] | Matched-WT AUROC [95% CI] | Control - fine AUROC [95% CI] |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
        )
    )


def _fixed_binding_comments(bindings: list[dict[str, Any]]) -> str:
    return "\n".join(
        "<!-- AIM3_CONDITIONAL_FIXED_VALUE "
        f"{binding['id']} {verifier._canonical_number(binding['value'])} -->"
        for binding in bindings
    )


def _target_internal_table(
    pure: dict[str, Any], residual: dict[str, Any]
) -> str:
    rows = []
    for result, method, display in (
        (pure, "pure_ridge_linear_probe", "Pure ridge linear probe"),
        (
            residual,
            "source_anchored_residual_ridge",
            "Source-anchored residual ridge",
        ),
    ):
        for budget in ADAPTATION_BUDGETS:
            cell = result["few_shot"]["cells"][str(budget)]
            for scope in ADAPTATION_SCOPES:
                block = cell["scopes"][scope]
                method_block = block[method]
                gain = block[f"{method}_minus_native"]
                seed = cell["per_source_seed_expected_auroc"][method][scope]
                rows.append(
                    f"| {display} | {budget} | {ADAPTATION_SCOPE_NAMES[scope]} | "
                    f"{method_block['auroc']:.4f} ± {method_block['procedure_sample_sd']:.4f} | "
                    f"[{method_block['ci95'][0]:.4f}, {method_block['ci95'][1]:.4f}] | "
                    f"{seed['mean']:.4f} ± {seed['sample_sd']:.4f} | "
                    f"{gain['auroc_gain']:+.4f} [{gain['ci95'][0]:.4f}, {gain['ci95'][1]:.4f}] |"
                )
    return "\n".join(
        (
            "| Method | Support k/class total | Held-out scope | Mean procedure AUROC ± procedure SD | 95% CI | Five source-seed expected AUROCs, mean ± SD | Gain over frozen native [95% CI] |",
            "|---|---:|---|---:|---:|---:|---:|",
            *rows,
        )
    )


def _native_target_table(result: dict[str, Any]) -> str:
    rows = []
    for scope in ADAPTATION_SCOPES:
        block = result["native_zero_shot"]["scopes"][scope]
        rows.append(
            f"| {ADAPTATION_SCOPE_NAMES[scope]} | {block['census']['patients']} "
            f"({block['census']['mutant']}) | {block['auroc']:.4f} "
            f"[{block['ci95'][0]:.4f}, {block['ci95'][1]:.4f}] |"
        )
    return "\n".join(
        (
            "| Target-internal scope | Patients (mutant) | Frozen native AUROC [95% CI] |",
            "|---|---:|---:|",
            *rows,
        )
    )


def _target_accounting_table(
    pure: dict[str, Any], residual: dict[str, Any]
) -> str:
    rows = []
    for result, method, display in (
        (pure, "pure_ridge_linear_probe", "Pure ridge"),
        (residual, "source_anchored_residual_ridge", "Residual ridge"),
    ):
        values = {
            binding["id"].rsplit(".", 1)[1]: binding["value"]
            for binding in _adaptation_accounting_bindings(
                method=method, accounting=result["fit_and_solver_accounting"]
            )
        }
        rows.append(
            f"| {display} | {values['embedding_artifacts']} | "
            f"{values['support_fold_procedures']} | {values['final_heads']} | "
            f"{values['test_applications']} | {values['solver_calls']} | "
            f"{values['source_model_fits']} | {values['other_method_fits']} | "
            f"{values['local_mil_fits']} | {values['full_label_fits']} | "
            f"{values['platt_fits']} |"
        )
    return "\n".join(
        (
            "| Method | Embedding artifacts | Support-fold procedures | Final heads | Cohort applications | Solver calls | Source fits | Other-method fits | Local MIL | Full-label | Platt |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
        )
    )


def _target_binding_comments(bindings: list[dict[str, Any]]) -> str:
    return "\n".join(
        "<!-- AIM2_TARGET_INTERNAL_VALUE "
        f"{binding['id']} {verifier._canonical_number(binding['value'])} -->"
        for binding in bindings
    )


def render_results(
    fine: dict[str, Any],
    fine_external: dict[str, Any],
    fixed: dict[str, Any],
    pure: dict[str, Any],
    residual: dict[str, Any],
    fixed_bindings: list[dict[str, Any]],
    target_bindings: list[dict[str, Any]],
    *,
    k32: dict[str, str] | None = None,
    curated_from_sha256: str = verifier.EXPECTED_AIM4_CURATED_FROM_SHA256,
) -> str:
    k32_pins = verifier.EXPECTED_AIM4_K32_SHA256 if k32 is None else k32
    parts = [
        "# FINAL-v13 results",
        "This is a verified, explicitly unsealed incremental candidate. It is "
        "not a final release and cannot be sealed while four Phase-2 records remain pending.",
        _firewall_block(),
        _priority_1_results(),
        _priority_2_results(),
        f"""{verifier.PRIORITY_HEADINGS[2]}

{verifier.SECONDARY_MARKERS[0]}

No LOCO performance is reported in FINAL-v13 Results. Historical LOCO arms
develop models on multiple cohort families, including cohorts reserved here as
external tests, and are ineligible under the TCGA+SurGen-primary-only rule.
Their authenticated provenance and exclusion remain in Setup and Audit.""",
        f"""{verifier.PRIORITY_HEADINGS[3]}

{verifier.SECONDARY_MARKERS[1]}

### Source-anchored combined-met TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION

{_native_target_table(pure)}

{_target_internal_table(pure, residual)}

{_target_accounting_table(pure, residual)}

{_target_binding_comments(target_bindings)}

Both methods use combined RIH-M plus SurGen-M support, k=2/4/8/16 per class
total, balanced equally by cohort within class. Each cell averages 100
support/fold procedures without prediction averaging. Confidence intervals use
a paired cohort-wise ordinary patient bootstrap, with one-class draws rejected;
outcome-class counts are not fixed by resampling. All budgets and both methods
are reported without target-driven selection. Every adapted point is below its
frozen native comparator; no positive repair claim is supported. This is not a
theoretical ceiling: target labels make it {TARGET_INTERNAL_ROLE}, and
SurGen-M is source-family exposed.

Legacy non-source-anchored few-shot, local-training, and target-label ceiling
performance is not reported in FINAL-v13 Results; its provenance and exclusion
remain in Setup and Audit.""",
        f"""{verifier.PRIORITY_HEADINGS[4]}

### TCGA+SurGen-primary UNI-v1 five-seed fine tasks

{_fine_external_internal_table(fine_external)}

{_binding_comments(fine)}

### Frozen-refit fine-task external/test performance

{_fine_external_refit_table(fine_external)}

{_fine_external_binding_comments(fine_external)}

Aim-3 external scoring used only the five p75 refit seed models and their
across-seed mean-native-logit ensemble. It did not generate a within-seed
fold5 ensemble for Aim 3. Values are refit-seed AUROC mean ± sample SD and the
across-seed ensemble AUROC with pointwise 10,000-resample patient-bootstrap CI.
Combined excludes all eight dual-role RIH patients; Primary-only pools CPTAC,
Orion, and RIH-primary; Met-only pools both metastatic cohorts. Sparse rows are
descriptive. No external score selected a model, checkpoint, construction,
calibration, or threshold.

### Two conditional canonical fixed matched-WT contrasts

{_fixed_table(fixed)}

{_fixed_binding_comments(fixed_bindings)}

These are two predeclared, single canonical fixed matched-WT draws. The matched
comparator materially raises conditional AUROC for codon and G12D-broad,
showing that their weak fine-task AUROCs are comparator-sensitive: mutant-versus-
mutant negatives share KRAS biology and are heterogeneous. Each estimate is
conditional on its one draw; it is not repeated-WT consensus, a performance
ceiling, or a biological/theoretical bound.

### Remaining canonical fixed and repeated-WT controls — pending

FINAL_V13_INCREMENTAL_STATUS: CANDIDATE_UNSEALED; FINE_EXTERNAL_COMPLETE; TWO_FIXED_COMPLETE; SOURCE_ANCHORED_FEWSHOT_COMPLETE; REMAINING_CANONICAL_AND_REPEATED_CONTROLS_PENDING; DO_NOT_SEAL.

### Legacy all-valid-patient Aim 3 — isolated secondary

{verifier.SECONDARY_MARKERS[2]}

No all-valid-patient Aim-3 performance is reported in FINAL-v13 Results. That
legacy population includes CPTAC and RIH in model development and is ineligible
under the TCGA+SurGen-primary-only rule. Provenance and exclusion remain in
Setup and Audit.""",
        f"""{verifier.PRIORITY_HEADINGS[5]}

{verifier.AIM4_BOUNDARY_MARKER}

Prototype p17 was associated in Set A/D with AUROC 0.5605 (q=0.0010) and
0.5939 (q=6.30e-7); p28 with 0.5968 (q=3.18e-9) and 0.6012 (q=7.91e-8),
with p28 attention AUROC 0.6172 (q=1.54e-13). Human-machine montage review
yielded 5 agree, 5 partial, and 1 differ; p17 was robust in 9/9 vocabulary
variants and p28 in 8/9. Single-reader, montage-level, transductive, and
noncausal limitations apply.

reviews/k32/completed_review.md live SHA-256: `{k32_pins['completed_review.md']}`

reviews/k32/completed_review_structured.json curated-from SHA-256: `{curated_from_sha256}`""",
    ]
    return "\n\n".join(parts) + "\n"


def render_audit(manifest: dict[str, Any]) -> str:
    source_rows = "\n".join(
        f"| `{record['id']}` | `{record['sha256']}` |"
        for record in manifest["artifacts"]
    )
    pending_rows = "\n".join(
        f"| `{record['id']}` | PENDING_PHASE2 |"
        for record in manifest["pending_artifacts"]
    )
    priority_text = "\n\n".join(
        (
            f"{verifier.PRIORITY_HEADINGS[0]}\n\nInherited source-only Aim-1 evidence; non-source numeric branches excluded.",
            f"{verifier.PRIORITY_HEADINGS[1]}\n\nInherited frozen-score source-only target evidence, including refit/fold5 robustness.",
            f"{verifier.PRIORITY_HEADINGS[2]}\n\nInherited LOCO provenance, numerically excluded from Results.",
            f"{verifier.PRIORITY_HEADINGS[3]}\n\nNew pure/residual combined-met evidence, segregated as TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION.",
            f"{verifier.PRIORITY_HEADINGS[4]}\n\nFine plus frozen-refit external evidence and two conditional fixed controls; remaining controls pending.",
            f"{verifier.PRIORITY_HEADINGS[5]}\n\nInherited statistics plus exact raw reviews/k32 provenance.",
        )
    )
    return f"""# FINAL-v13 evidence audit

This incremental candidate inherits exactly 142 immutable FINAL-v12.1 records,
adds 83 directly rehashed records, and declares four absent Phase-2 records as
pending: 225 material scientific sources total. It has no publication receipt,
no reconciliation pins, and is not seal-ready.

{priority_text}

{verifier.FAILED_V1_AUDIT_MARKER}

The v2 root is controlling. The main external-data firewall is enforced by its
contract, frozen source lineage, preflight, and fine-phase completion receipt.
The fine-external seal proves zero new fits and outcome access only after frozen
scores. The append-only two-fixed continuation proves exactly ten selected
chains at peak six and excludes other fixed/repeated namespaces. Pure and
residual combined-met contracts, schedulers, OOF predictions, results, and
terminal completions are byte-pinned; point AUROCs, procedure SDs, source-seed
summaries, fixed-control CIs, and all displayed bindings are independently
replayed. Adaptation never modifies or selects the source model and is never
external validation. The remaining four full-control artifacts are absent.
The reviews/k32 Zone.Identifier sidecar is authenticated by the verifier but is
explicitly excluded from the scientific source ledger.

## Exact source index

| Source ID | SHA-256 |
|---|---|
{source_rows}

## Pending Phase-2 source index

| Source ID | State |
|---|---|
{pending_rows}
"""


def build_candidate_payload() -> dict[str, Any]:
    """Render the candidate from material evidence, without filesystem writes."""

    paths = candidate_paths()
    manifest = build_source_manifest()
    sources = manifest["artifacts"]
    fine_summary, _ = verifier._validate_aim3_fine_results(paths, sources)
    verifier._validate_fine_execution_evidence(paths, sources)
    external_summary, _ = verifier._validate_aim3_fine_external_results(
        paths, sources
    )
    incremental = validate_incremental_evidence(paths, sources)
    verifier._validate_aim4_k32(paths, sources)
    fine_record = next(
        record for record in sources if record["id"] == AIM3_FINE_RESULTS_SOURCE_ID
    )
    fine_path = verifier.parent._lexical_source_path(str(fine_record["path"]), paths)
    fine, _ = verifier._strict_stable_json(
        fine_path, label="Aim-3 fine-only results", repo=paths.repo
    )
    external_record = next(
        record
        for record in sources
        if record["id"] == AIM3_FINE_EXTERNAL_RESULTS_SOURCE_ID
    )
    external_path = verifier.parent._lexical_source_path(
        str(external_record["path"]), paths
    )
    fine_external, _ = verifier._strict_stable_json(
        external_path,
        label="Aim-3 fine-external results",
        repo=paths.repo,
    )
    if set(fine_summary) != {"fine_results"}:
        raise verifier.BundleVerificationError("fine-result validation return drift")
    if external_summary.get("numeric_binding_count") != 370:
        raise verifier.BundleVerificationError(
            "fine-external validation return drift"
        )
    return {
        "source_manifest": manifest,
        "documents": {
            "Experimental_Setup.md": render_experimental_setup(),
            "Results.md": render_results(
                fine,
                fine_external,
                incremental["two_fixed"],
                incremental["pure_ridge"],
                incremental["residual_ridge"],
                incremental["two_fixed_bindings"],
                incremental["target_internal_bindings"],
            ),
            "Audit.md": render_audit(manifest),
        },
    }


def ledger_status() -> dict[str, Any]:
    records = []
    for source in extension_sources():
        exists = source.path.is_file() and not source.path.is_symlink()
        record: dict[str, Any] = {
            "id": source.source_id,
            "path": _display(source.path),
            "phase": "pending_phase2" if source.pending_phase2 else "material_phase1",
            "exists": exists,
        }
        if exists:
            record.update(verifier._identity(source.path, display_path=_display(source.path)))
        records.append(record)
    return {
        "schema_version": 1,
        "status": "incremental_extension_ledger",
        "parent_source_count": 142,
        "extension_source_count": len(records),
        "material_phase1_source_count": sum(
            not source.pending_phase2 for source in extension_sources()
        ),
        "pending_phase2_source_count": sum(
            source.pending_phase2 for source in extension_sources()
        ),
        "records": records,
    }


def check_candidate_bundle() -> dict[str, Any]:
    """Byte-check the deterministic candidate before semantic verification."""

    verifier._trusted_reconciliation_generator_identities()
    paths = candidate_paths()
    payload = build_candidate_payload()
    verifier._require_exact_installed_bytes(
        paths.final_v13 / verifier.SOURCE_MANIFEST_NAME,
        verifier._canonical_json_bytes(payload["source_manifest"]),
        label="FINAL-v13 Phase-1 source manifest",
    )
    for name in verifier.REPORT_DOCUMENTS:
        text = payload["documents"].get(name)
        if not isinstance(text, str):
            raise verifier.BundleVerificationError(
                f"Phase-1 deterministic renderer returned non-text {name}"
            )
        verifier._require_exact_installed_bytes(
            paths.final_v13 / name,
            text.encode("utf-8"),
            label=f"FINAL-v13 Phase-1 {name}",
        )
    checked = verifier.check_bundle(paths)
    if (
        checked.get("status")
        != "INCREMENTAL_TWO_FIXED_AND_SOURCE_ANCHORED_FEWSHOT_CANDIDATE_VERIFIED_UNSEALED"
    ):
        raise verifier.BundleVerificationError(
            "production verifier did not preserve Phase-1 unsealed status"
        )
    return checked


def install_candidate_bundle(*, apply: bool) -> dict[str, Any]:
    """Atomically install deterministic unsealed bytes; never publish a receipt."""

    if not apply:
        raise verifier.BundleVerificationError("install requires explicit --apply")
    paths = candidate_paths()
    if paths.destination.exists() or paths.destination.is_symlink():
        raise verifier.BundleVerificationError(
            "incremental install refuses an existing FINAL-v13 receipt"
        )
    if paths.reconciliation_pins.exists() or paths.reconciliation_pins.is_symlink():
        raise verifier.BundleVerificationError(
            "incremental install refuses full reconciliation pins"
        )
    verifier.parent._reject_symlink_chain(
        paths.final_v13, context="FINAL-v13 incremental install root"
    )
    payload = build_candidate_payload()
    rendered = {
        paths.final_v13 / verifier.SOURCE_MANIFEST_NAME: verifier._canonical_json_bytes(
            payload["source_manifest"]
        ),
        **{
            paths.final_v13 / name: payload["documents"][name].encode("utf-8")
            for name in verifier.REPORT_DOCUMENTS
        },
    }
    paths.final_v13.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, content in rendered.items():
            if destination.is_symlink():
                raise verifier.BundleVerificationError(
                    f"incremental install refuses symlink destination: {destination}"
                )
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=paths.final_v13,
            )
            temporary_path = Path(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary_path, destination))
        for temporary_path, destination in staged:
            os.replace(temporary_path, destination)
    finally:
        for temporary_path, _ in staged:
            if temporary_path.exists():
                temporary_path.unlink()
    checked = check_candidate_bundle()
    return {
        "status": "INCREMENTAL_CANDIDATE_INSTALLED_VERIFIED_UNSEALED",
        "installed_files": {
            path.name: verifier._identity(path, display_path=_display(path))
            for path in rendered
        },
        "verification": checked,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("ledger", "render", "check", "install"),
        help="render/check are read-only; install requires --apply and never seals",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="required only for deterministic unsealed candidate installation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "ledger":
        value = ledger_status()
    elif args.action == "render":
        value = build_candidate_payload()
    elif args.action == "check":
        value = check_candidate_bundle()
    else:
        value = install_candidate_bundle(apply=args.apply)
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
