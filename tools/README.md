# Study tools and experiment entry points

The root experiment entries use `aim{1,2,3,4}_<purpose>.py` names. The complete
[entry-point tables below](#canonical-experiment-entry-points) link directly to
the published files and map their earlier names. Generic pipeline stages live
in [`scripts/`](../scripts/); reusable code lives in
[`src/oceanpath/`](../src/oceanpath/); experiment settings live in
[`configs/`](../configs/). This directory contains study preparation, campaign
support, extraction operations, and report/audit utilities.

This is a source publication containing `configs/`, `scripts/`, `src/`,
`tests/`, `tools/`, and the main experiment entries. Datasets, slide images,
feature stores, trained checkpoints, generated outputs, review packets,
reports, and archival directories are not distributed here. Existing data
locations and experiment identifiers in configurations must be matched to the
external resources required by each command.

Root setup files (`pyproject.toml`, `uv.lock`, `Makefile`, and
`.python-version`) are outside this publication's requested scope. This tree
therefore assumes an existing project environment. Campaigns that record
dependency metadata in source snapshots also require the matching project and
lock files before preparation can succeed.

Frozen receipts and source snapshots require their original artifacts and
original source bytes. The renamed publication does not replace a historical
snapshot or automatically satisfy its path and checksum checks. Supply the
original artifacts when replaying or verifying an existing experiment; follow
the relevant controller's preparation and provenance steps for a new run.

Plotting commands are grouped in [`visualization/`](visualization/README.md).
Supporting tools retain their existing filenames. Read a tool's CLI help and
protocol before starting a run; this page is a navigation guide, not an
installation guide or launch sequence.

## Preparation and training

| Tools | Purpose |
|---|---|
| [phase1_technical_audit.py](phase1_technical_audit.py) | per-slide technical descriptors |
| [phase2_manifests_splits.py](phase2_manifests_splits.py) | study manifests and frozen balanced folds |
| [phase2b_hparam_search.py](phase2b_hparam_search.py) | pre-baseline hyperparameter search |
| [study_train.py](study_train.py) | shared study training invocation |
| [build_codon_task_manifests.py](build_codon_task_manifests.py), [build_granularity_manifests.py](build_granularity_manifests.py), [build_crc_orion_labels.py](build_crc_orion_labels.py) | task/cohort manifest and label preparation |
| [derive_stage_crc_final_v4.py](derive_stage_crc_final_v4.py), [derive_sidedness_crc_final_v4.py](derive_sidedness_crc_final_v4.py) | clinical metadata derivations |
| [train_all_pipeline.py](train_all_pipeline.py), [run_all_others_oof.py](run_all_others_oof.py), [kras_initial_campaign.sh](kras_initial_campaign.sh) | existing campaign launchers |

## Aim-specific campaign support

| Tools | Purpose |
|---|---|
| [aim1_e0_five_seed_extension.py](aim1_e0_five_seed_extension.py) | E0 five-seed extension used by the root expansion controller |
| [aim1_source_cohort_five_seed_campaign.py](aim1_source_cohort_five_seed_campaign.py), [aim1_source_cohort_five_seed_analysis.py](aim1_source_cohort_five_seed_analysis.py) | source-cohort campaign and analysis |
| [aim1_tcga_surgen_two_encoder_campaign.py](aim1_tcga_surgen_two_encoder_campaign.py), `aim1_tcga_surgen_two_encoder_recovery*.py` | TCGA/SurGen two-encoder campaign and recovery variants |
| `aim1_tcga_surgen_full_pipeline_analysis*.py` | pipeline analyses, versioned revisions, and explicit errata |
| [aim2_e2a_five_seed_adjudication.py](aim2_e2a_five_seed_adjudication.py), [aim2_lineage_receipt.py](aim2_lineage_receipt.py), [aim2_exploratory_refresh.py](aim2_exploratory_refresh.py) | LOCO adjudication, lineage receipts, and exploratory updates |
| [build_v2_ladder.py](build_v2_ladder.py), [score_v2_ladder.py](score_v2_ladder.py), [verify_v2_ladder.py](verify_v2_ladder.py) | versioned molecular-resolution ladder support |
| [aim4_vocab_stability.py](aim4_vocab_stability.py) | Aim 4 vocabulary-stability analysis |

## Final v14 Aim 4 audit and tiling grid

These tools support the Aim 4 audit/grid workflow. Handoffs, logs, and generated
outputs belong to the external experiment artifacts and are not included in
this source publication. The table describes components; its order is not a
launch sequence.

| Tools | Purpose |
|---|---|
| [final_v14_grid_offset.py](final_v14_grid_offset.py), [final_v14_grid_offset_extract.py](final_v14_grid_offset_extract.py), [final_v14_grid_benchmark.py](final_v14_grid_benchmark.py) | grid-offset workflow, extraction, and benchmarking |
| [final_v14_job_resource_monitor.py](final_v14_job_resource_monitor.py) | grid-job resource monitoring |
| [audit_final_v14_grid_results.py](audit_final_v14_grid_results.py), [audit_final_v14_grid_legacy_receipts.py](audit_final_v14_grid_legacy_receipts.py) | grid-result and legacy-receipt audits |
| [final_v14_finish_after_grid.py](final_v14_finish_after_grid.py) | resume sealed audit/report packaging after the grid succeeds |
| `final_v14_context.py`, `final_v14_correspondence*.py`, `final_v14_variant_mapping.py` | context, correspondence, replay, and variant mapping |
| `final_v14_module1*.py`, `final_v14_module2*.py`, `final_v14_module3.py`, `final_v14_modeling*.py` | versioned analysis modules and modeling |
| `final_v14_pre_reader.py`, `final_v14_post_reader.py`, `final_v14_review_interpretation.py` | reader packets and interpretation |
| `audit_final_v14_*.py` | context, correspondence, module, reader, and mapping audits |
| [final_v14_bundle_receipt.py](final_v14_bundle_receipt.py), [final_v14_complete_report.py](final_v14_complete_report.py), [final_v14_report.py](final_v14_report.py), [final_v14_excel_handoff.py](final_v14_excel_handoff.py) | receipts, reports, and spreadsheet handoff |

## Reporting and verification

| Tools | Purpose |
|---|---|
| [results_verify.py](results_verify.py), [e2_audit.py](e2_audit.py), `verify_aim2_corrected.py`, `verify_corrected_aim2_aim3*.py` | reported-result, design, and corrected-lineage checks |
| [results_workbook.py](results_workbook.py) | result workbooks |
| [visualization/](visualization/README.md) | training curves, embedding maps, and cohort/encoder UMAPs |
| `build_reviews_v*.py`, [analyze_reviews_v5.py](analyze_reviews_v5.py) | versioned review packets and analysis |
| `build_crc_final_v*.py`, `final_v3_*.py` through `final_v13_*.py`, `crc_final_v6_*.py` | earlier versioned analyses, report builders, receipts, and reconciliation |

Version suffixes identify distinct scientific lineages or revisions. Consult
their reports and receipts before choosing an implementation; the highest
version number alone does not establish which inputs a tool should consume.

## Extraction and operational support

| Tools | Purpose |
|---|---|
| [download_crc_orion_slides.py](download_crc_orion_slides.py) | Orion slide download |
| [run_cptac_priority_encoding.py](run_cptac_priority_encoding.py), [resume_virchow2_after_cptac.py](resume_virchow2_after_cptac.py) | coordinated encoding and continuation |
| [pack_orion_expanded.py](pack_orion_expanded.py), [pack_virchow2_after_encoding.py](pack_virchow2_after_encoding.py) | cohort/encoder feature packing |
| [assert_cuda_idle.sh](assert_cuda_idle.sh) | GPU-idle guard for existing launch workflows |
| [migrate_colon_receipts_v2.py](migrate_colon_receipts_v2.py) | historical receipt migration |

For generic extraction, streaming, and packing entry points, see
[`scripts/extract_features.py`](../scripts/extract_features.py),
[`scripts/watch_colon_encoding.py`](../scripts/watch_colon_encoding.py), and
[`scripts/pack_features.py`](../scripts/pack_features.py).

## Canonical experiment entry points

Each linked entry below is a regular Python file. Earlier aliases have been
consolidated into these canonical entries; legacy root filenames and symlinks
are not included. Imports, launch commands, and references to current source
files use the canonical names. Historical data and output identifiers retain
their experiment labels. Aim 1 includes the former E0 baseline and expansion
entries; the MIL expansion controller also coordinates Aim 2 and Aim 3 work.

### Aim 1 — baseline and source-cohort experiments

| Canonical root entry point | Earlier root name(s) |
|---|---|
| [aim1_all_primary_orion_experiment.py](../aim1_all_primary_orion_experiment.py) | `e0_all_primary_orion_experiment.py` |
| [aim1_all_primary_orion_parallel_v2.py](../aim1_all_primary_orion_parallel_v2.py) | `e0_all_primary_orion_parallel_v2.py` |
| [aim1_challenge_sets.py](../aim1_challenge_sets.py) | `e1a_challenge_sets.py` |
| [aim1_clinical_baseline.py](../aim1_clinical_baseline.py) | `e1d_clinical_baseline.py` |
| [aim1_locked_baseline.py](../aim1_locked_baseline.py) | `e0_locked_baseline.py` |
| [aim1_mil_five_seed_expansion.py](../aim1_mil_five_seed_expansion.py) | `e0_mil_five_seed_expansion.py`, `five_seed_mil_expansion_campaign.py` |
| [aim1_msi_positive_control.py](../aim1_msi_positive_control.py) | `e1e_msi_positive_control.py` |
| [aim1_s_composition_standardized.py](../aim1_s_composition_standardized.py) | `e1a_s_composition_standardized.py` |
| [aim1_virchow2_encoder_arm.py](../aim1_virchow2_encoder_arm.py) | `e1v_virchow2_encoder_arm.py` |

### Aim 2 — transport and adaptation experiments

| Canonical root entry point | Earlier root name(s) |
|---|---|
| [aim2_a_orion_residual_adaptation.py](../aim2_a_orion_residual_adaptation.py) | `e2cpht_a_orion_residual_adaptation.py` |
| [aim2_confirmatory_transfer.py](../aim2_confirmatory_transfer.py) | `e2cpht_confirmatory_transfer.py` |
| [aim2_cross_protocol_transfer.py](../aim2_cross_protocol_transfer.py) | `e2cpht_cross_protocol_transfer.py` |
| [aim2_cross_protocol_transfer_verify.py](../aim2_cross_protocol_transfer_verify.py) | `e2cpht_cross_protocol_transfer_verify.py` |
| [aim2_head_adaptation_base.py](../aim2_head_adaptation_base.py) | `e2c_head_adaptation_base.py` |
| [aim2_loco_five_seed_extension.py](../aim2_loco_five_seed_extension.py) | `e2a_loco_five_seed_extension.py` |
| [aim2_loco_transport.py](../aim2_loco_transport.py) | `e2a_loco_transport.py` |
| [aim2_metastatic_indomain_bound.py](../aim2_metastatic_indomain_bound.py) | `e2e_metastatic_indomain_bound.py` |
| [aim2_metastatic_site.py](../aim2_metastatic_site.py) | `e2d1_metastatic_site.py` |
| [aim2_metastatic_transport.py](../aim2_metastatic_transport.py) | `e2b_metastatic_transport.py` |
| [aim2_paired_specimen_concordance.py](../aim2_paired_specimen_concordance.py) | `e2d5_paired_specimen_concordance.py` |
| [aim2_peritoneal_stability.py](../aim2_peritoneal_stability.py) | `e2d2_peritoneal_stability.py` |
| [aim2_primary_to_metastatic_transfer.py](../aim2_primary_to_metastatic_transfer.py) | `e2met_primary_to_metastatic_transfer.py` |
| [aim2_residual_adaptation.py](../aim2_residual_adaptation.py) | `e2c_residual_adaptation.py` |
| [aim2_rih_acquisition_regime.py](../aim2_rih_acquisition_regime.py) | `e2d6_rih_acquisition_regime.py` |
| [aim2_setd_role_contrast.py](../aim2_setd_role_contrast.py) | `e2d3_setd_role_contrast.py` |
| [aim2_sibling_loco.py](../aim2_sibling_loco.py) | `e2ad_sibling_loco.py` |
| [aim2_surgen_subcohort_gap.py](../aim2_surgen_subcohort_gap.py) | `e2d4_surgen_subcohort_gap.py` |
| [aim2_tcga_surgen_pure_ridge_combined_fewshot_campaign.py](../aim2_tcga_surgen_pure_ridge_combined_fewshot_campaign.py) | `e2c_tcga_surgen_pure_ridge_combined_fewshot.py` |
| [aim2_tcga_surgen_residual_combined_fewshot_campaign.py](../aim2_tcga_surgen_residual_combined_fewshot_campaign.py) | `e2c_tcga_surgen_residual_combined_fewshot.py` |
| [aim2_tcga_surgen_source_anchored_met_adaptation_campaign.py](../aim2_tcga_surgen_source_anchored_met_adaptation_campaign.py) | `e2c_tcga_surgen_source_anchored_met_adaptation.py` |
| [aim2_v3_fulllabel_residual_adaptation.py](../aim2_v3_fulllabel_residual_adaptation.py) | `e2f_v3_fulllabel_residual_adaptation.py` |
| [aim2_v3_fulllabel_residual_verify.py](../aim2_v3_fulllabel_residual_verify.py) | `e2f_v3_fulllabel_residual_verify.py` |

### Aim 3 — molecular-resolution and control experiments

| Canonical root entry point | Earlier root name(s) |
|---|---|
| [aim3_fixed_control_analysis.py](../aim3_fixed_control_analysis.py) | `e3_fixed_control_analysis.py` |
| [aim3_ladders_five_seed_extension.py](../aim3_ladders_five_seed_extension.py) | `e3_ladders_five_seed_extension.py` |
| [aim3_repeated_control_campaign.py](../aim3_repeated_control_campaign.py) | `e3_repeated_control_campaign.py` |
| [aim3_resolution_ladder.py](../aim3_resolution_ladder.py) | `e3_resolution_ladder.py` |
| [aim3_tcga_surgen_primary_fine_external_campaign.py](../aim3_tcga_surgen_primary_fine_external_campaign.py) | `e3_tcga_surgen_primary_fine_external.py` |
| [aim3_tcga_surgen_primary_five_seed_campaign.py](../aim3_tcga_surgen_primary_five_seed_campaign.py) | `e3_tcga_surgen_primary_five_seed.py` |
| [aim3_tcga_surgen_primary_two_fixed_controls_campaign.py](../aim3_tcga_surgen_primary_two_fixed_controls_campaign.py) | `e3_tcga_surgen_primary_two_fixed_controls.py` |
| [aim3_virchow2_replication.py](../aim3_virchow2_replication.py) | `e3v_virchow2_replication.py` |

### Aim 4 — morphologic atlas experiments

| Canonical root entry point | Earlier root name(s) |
|---|---|
| [aim4_morphologic_atlas.py](../aim4_morphologic_atlas.py) | `e4_morphologic_atlas.py` |
| [aim4_morphologic_atlas_base.py](../aim4_morphologic_atlas_base.py) | `e4_morphologic_atlas_base.py` |
