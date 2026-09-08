#!/usr/bin/env python3
"""Render or check the complete, seal-ready FINAL-v13 reconciliation.

This helper is deliberately read-only.  ``render`` returns the exact 229-row
scientific source manifest, three report documents, and strict pre-seal
reconciliation pins.  ``check`` byte-compares those renderings with installed
files and delegates to the production FINAL-v13 verifier.  Neither action
writes reports, publishes a receipt, or touches the frozen Aim-3 campaign.

The pin file is pre-seal metadata, not scientific evidence.  This exclusion is
necessary because the scientific Audit includes the verifier hash: embedding
the document hashes back into that verifier would create a cryptographic
self-hash cycle.  The final receipt binds the pin file's byte identity instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import final_v13_bundle_receipt as verifier  # noqa: E402
from tools import final_v13_phase1_candidate as phase1  # noqa: E402

ExtensionSource = phase1.ExtensionSource
FULL_PARENT_SOURCE_COUNT = 142
FULL_EXTENSION_SOURCE_COUNT = 87
FULL_SOURCE_COUNT = 229

UNFROZEN_TERMINAL_PHASE2_SHA256 = "UNFROZEN_TERMINAL_PHASE2_SHA256"
TERMINAL_PHASE2_SOURCE_IDS = phase1.PENDING_PHASE2_SOURCE_IDS
# Frozen only after the campaign-native terminal verifier passed all six stages.
# Any later byte drift makes production render, pin generation, and sealing fail
# closed.
EXPECTED_TERMINAL_PHASE2_IDENTITIES: dict[str, tuple[int | None, str]] = {
    "aim3-source-primary-controls-scheduler": (
        27_563,
        "1c9086338a049e519572b44b9adabf76fd52ef3d74d7628e1943709b8cc540c9",
    ),
    "aim3-source-primary-results": (
        53_695,
        "9a8d1366eb00ea01aa85d246eb8aeddc2377e99650a8897deb3a547aa5de3c7e",
    ),
    "aim3-source-primary-scheduler": (
        28_065,
        "f7e380a88b98ad585f7f77fddf73f7fc04995e67268d252490b05435216d5b7f",
    ),
    "aim3-source-primary-training-completion": (
        37_004,
        "a66793097877a9d40fa0164c96706fa603b07aee23a08aabc87eeb5d92e4ea86",
    ),
}

# Authenticated FINAL-v12.1 target rows.  The vectors are ordered by model
# seeds 42, 43, 44, 45, and 46; SD is sample seed dispersion, not a CI.
EXTERNAL_CONSTRUCTION_ROWS = (
    (
        "UNI-v1",
        "CPTAC-primary",
        "94 (33)",
        "0.7250, 0.6853, 0.7084, 0.6672, 0.6749",
        "0.6922 ± 0.0241",
        "0.6607, 0.6890, 0.6706, 0.6642, 0.6970",
        "0.6763 ± 0.0159",
        "family-naive primary",
    ),
    (
        "UNI-v1",
        "Orion",
        "40 (15)",
        "0.7227, 0.7747, 0.6507, 0.6933, 0.7360",
        "0.7155 ± 0.0465",
        "0.6907, 0.7520, 0.7253, 0.7013, 0.7013",
        "0.7141 ± 0.0247",
        "retrospective cross-protocol",
    ),
    (
        "UNI-v1",
        "RIH-primary",
        "153 (70)",
        "0.7217, 0.7224, 0.7281, 0.7325, 0.7380",
        "0.7286 ± 0.0069",
        "0.7213, 0.7201, 0.7286, 0.7398, 0.7353",
        "0.7290 ± 0.0086",
        "family-naive primary",
    ),
    (
        "UNI-v1",
        "RIH-metastatic",
        "85 (37)",
        "0.6104, 0.6408, 0.6470, 0.5864, 0.5932",
        "0.6155 ± 0.0274",
        "0.6458, 0.6289, 0.6408, 0.6295, 0.6289",
        "0.6348 ± 0.0080",
        "family-naive retrospective",
    ),
    (
        "UNI-v1",
        "SurGen-metastatic (SR1482-M)",
        "74 (30)",
        "0.5674, 0.5534, 0.5780, 0.5898, 0.5561",
        "0.5689 ± 0.0152",
        "0.6235, 0.5667, 0.5871, 0.5788, 0.6114",
        "0.5935 ± 0.0234",
        "source-family-exposed retrospective",
    ),
    (
        "Virchow2-CLS",
        "CPTAC-primary",
        "94 (33)",
        "0.6833, 0.6744, 0.6413, 0.6947, 0.7094",
        "0.6806 ± 0.0256",
        "0.6925, 0.6930, 0.7129, 0.6557, 0.7168",
        "0.6942 ± 0.0242",
        "family-naive primary",
    ),
    (
        "Virchow2-CLS",
        "Orion",
        "40 (15)",
        "0.8160, 0.7773, 0.7640, 0.7453, 0.7947",
        "0.7795 ± 0.0273",
        "0.7200, 0.7387, 0.7387, 0.7360, 0.7760",
        "0.7419 ± 0.0206",
        "retrospective cross-protocol",
    ),
    (
        "Virchow2-CLS",
        "RIH-primary",
        "153 (70)",
        "0.6816, 0.6855, 0.6926, 0.6867, 0.6998",
        "0.6892 ± 0.0071",
        "0.6997, 0.7164, 0.7317, 0.7303, 0.7213",
        "0.7199 ± 0.0130",
        "family-naive primary",
    ),
    (
        "Virchow2-CLS",
        "RIH-metastatic",
        "85 (37)",
        "0.6160, 0.6329, 0.6208, 0.6540, 0.5954",
        "0.6238 ± 0.0216",
        "0.6943, 0.6374, 0.6385, 0.6858, 0.6616",
        "0.6635 ± 0.0262",
        "family-naive retrospective",
    ),
    (
        "Virchow2-CLS",
        "SurGen-metastatic (SR1482-M)",
        "74 (30)",
        "0.5848, 0.5538, 0.6098, 0.5708, 0.5648",
        "0.5768 ± 0.0216",
        "0.5606, 0.5856, 0.6023, 0.5417, 0.6076",
        "0.5795 ± 0.0280",
        "source-family-exposed retrospective",
    ),
)

EXTERNAL_NATIVE_ROWS = (
    ("UNI-v1", "CPTAC-primary", "94 (33)", "0.7059 [0.5946, 0.8077]"),
    ("UNI-v1", "Orion", "40 (15)", "0.7120 [0.5440, 0.8613]"),
    ("UNI-v1", "RIH-primary", "153 (70)", "0.7467 [0.6683, 0.8220]"),
    ("UNI-v1", "RIH-metastatic", "85 (37)", "0.6256 [0.4977, 0.7483]"),
    (
        "UNI-v1",
        "SurGen-metastatic (SR1482-M)",
        "74 (30)",
        "0.5864 [0.4500, 0.7242]",
    ),
    (
        "Virchow2-CLS",
        "CPTAC-primary",
        "94 (33)",
        "0.6846 [0.5609, 0.7998]",
    ),
    ("Virchow2-CLS", "Orion", "40 (15)", "0.7947 [0.6373, 0.9227]"),
    (
        "Virchow2-CLS",
        "RIH-primary",
        "153 (70)",
        "0.7053 [0.6244, 0.7830]",
    ),
    (
        "Virchow2-CLS",
        "RIH-metastatic",
        "85 (37)",
        "0.6427 [0.5214, 0.7607]",
    ),
    (
        "Virchow2-CLS",
        "SurGen-metastatic (SR1482-M)",
        "74 (30)",
        "0.5955 [0.4602, 0.7231]",
    ),
)


def extension_sources() -> tuple[ExtensionSource, ...]:
    """Promote the exact incremental 87-record ledger to full material status."""

    promoted = tuple(
        dataclasses.replace(source, pending_phase2=False) for source in phase1.extension_sources()
    )
    ids = tuple(source.source_id for source in promoted)
    paths = tuple(str(source.path.absolute()) for source in promoted)
    if (
        len(promoted) != FULL_EXTENSION_SOURCE_COUNT
        or ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or len(paths) != len(set(paths))
    ):
        raise verifier.BundleVerificationError(
            "FINAL-v13 full extension ledger is not exactly 87 sorted unique records"
        )
    return promoted


def reconciliation_paths() -> verifier.BundlePaths:
    """Return production paths with the exact semantic extension IDs bound."""

    return phase1.candidate_paths()


def _display(path: Path, repo: Path) -> str:
    absolute = path.absolute()
    try:
        return absolute.relative_to(repo.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _material_record(source: ExtensionSource, *, repo: Path) -> dict[str, Any]:
    identity = verifier._identity(
        source.path,
        display_path=_display(source.path, repo),
    )
    return {
        "id": source.source_id,
        "aims": list(source.aims),
        "experiments": list(source.experiments),
        "role": source.role,
        **identity,
    }


def _require_terminal_phase2_identities(
    paths: verifier.BundlePaths,
    extensions: Sequence[ExtensionSource],
) -> dict[str, dict[str, Any]]:
    """Require all four Phase-2 outputs and immutable post-verify identities."""

    required_ids = (
        paths.aim3_controls_scheduler_source_id,
        paths.aim3_results_source_id,
        paths.aim3_scheduler_source_id,
        paths.aim3_training_source_id,
    )
    by_id = {source.source_id: source for source in extensions}
    if set(required_ids) - set(by_id):
        raise verifier.BundleVerificationError(
            "FINAL-v13 terminal Phase-2 source roster is incomplete"
        )
    production = verifier._is_production_bundle(paths)
    pins = EXPECTED_TERMINAL_PHASE2_IDENTITIES
    if set(pins) != set(TERMINAL_PHASE2_SOURCE_IDS):
        raise verifier.BundleVerificationError(
            "FINAL-v13 terminal Phase-2 identity pin roster drift"
        )
    if production and required_ids != TERMINAL_PHASE2_SOURCE_IDS:
        raise verifier.BundleVerificationError(
            "FINAL-v13 production terminal Phase-2 semantic ID drift"
        )
    if production and any(
        size is None or digest == UNFROZEN_TERMINAL_PHASE2_SHA256 for size, digest in pins.values()
    ):
        raise verifier.BundleVerificationError(
            "FINAL-v13 terminal Phase-2 identity pins are UNFROZEN"
        )

    identities: dict[str, dict[str, Any]] = {}
    for source_id in required_ids:
        source = by_id[source_id]
        path = source.path
        verifier.parent._reject_symlink_chain(
            path, context=f"FINAL-v13 terminal Phase-2 source {source_id}"
        )
        if not path.is_file() or path.is_symlink():
            raise verifier.BundleVerificationError(
                f"FINAL-v13 terminal Phase-2 source is absent: {source_id}"
            )
        identity = verifier._identity(path, display_path=_display(path, paths.repo))
        expected_size, expected_digest = (
            pins[source_id] if production else (identity["size_bytes"], identity["sha256"])
        )
        if (
            not isinstance(expected_size, int)
            or expected_size <= 0
            or not isinstance(expected_digest, str)
            or verifier._SHA256_RE.fullmatch(expected_digest) is None
            or identity["size_bytes"] != expected_size
            or identity["sha256"] != expected_digest
        ):
            raise verifier.BundleVerificationError(
                f"FINAL-v13 terminal Phase-2 identity drift: {source_id}"
            )
        identities[source_id] = identity
    return identities


def _validate_extension_roster(
    paths: verifier.BundlePaths,
    extensions: Sequence[ExtensionSource],
) -> tuple[ExtensionSource, ...]:
    ordered = tuple(extensions)
    ids = tuple(source.source_id for source in ordered)
    source_paths = tuple(str(source.path.absolute()) for source in ordered)
    if (
        len(ordered) != FULL_EXTENSION_SOURCE_COUNT
        or ids != tuple(sorted(ids))
        or len(ids) != len(set(ids))
        or len(source_paths) != len(set(source_paths))
        or any(source.pending_phase2 for source in ordered)
    ):
        raise verifier.BundleVerificationError(
            "FINAL-v13 full extension roster must be 87 sorted, unique, material records"
        )
    configured = tuple(paths.expected_extension_source_ids)
    if configured != (verifier.UNFROZEN,) and configured != ids:
        raise verifier.BundleVerificationError(
            "FINAL-v13 reconciler extension roster disagrees with verifier paths"
        )
    return ordered


def build_source_manifest(
    paths: verifier.BundlePaths | None = None,
    extensions: Sequence[ExtensionSource] | None = None,
) -> dict[str, Any]:
    """Build the exact 142-parent + 87-extension scientific ledger."""

    selected = reconciliation_paths() if paths is None else paths
    selected_extensions = _validate_extension_roster(
        selected,
        extension_sources() if extensions is None else extensions,
    )
    _require_terminal_phase2_identities(selected, selected_extensions)
    if selected.expected_parent_source_count != FULL_PARENT_SOURCE_COUNT:
        raise verifier.BundleVerificationError(
            "FINAL-v13 full reconciliation requires exactly 142 parent sources"
        )
    if verifier.sha256_file(selected.parent_manifest) != selected.expected_parent_manifest_sha256:
        raise verifier.BundleVerificationError("sealed FINAL-v12.1 manifest SHA-256 drift")
    parent_manifest, _ = verifier._strict_stable_json(
        selected.parent_manifest,
        label="sealed FINAL-v12.1 source manifest",
        repo=selected.repo,
    )
    parent_sources = parent_manifest.get("artifacts")
    if (
        parent_manifest.get("schema_version") != 2
        or parent_manifest.get("bundle") != "final_v12_1"
        or parent_manifest.get("status") != selected.expected_parent_manifest_status
        or parent_manifest.get("pending_artifacts") != []
        or not isinstance(parent_sources, list)
        or len(parent_sources) != FULL_PARENT_SOURCE_COUNT
    ):
        raise verifier.BundleVerificationError(
            "sealed FINAL-v12.1 source manifest is not the exact 142-row parent"
        )
    material = [_material_record(source, repo=selected.repo) for source in selected_extensions]
    parent_ids = {str(record["id"]) for record in parent_sources}
    extension_ids = {record["id"] for record in material}
    if parent_ids.intersection(extension_ids):
        raise verifier.BundleVerificationError(
            "FINAL-v13 extension source ID collides with sealed parent"
        )
    pin_path = selected.reconciliation_pins.absolute()
    if any(source.path.absolute() == pin_path for source in selected_extensions):
        raise verifier.BundleVerificationError(
            "reconciliation pins must remain outside the scientific source ledger"
        )
    artifacts = sorted([*parent_sources, *material], key=lambda record: record["id"])
    if len(artifacts) != FULL_SOURCE_COUNT:
        raise verifier.BundleVerificationError(
            "FINAL-v13 full scientific source count is not exactly 229"
        )
    return {
        "schema_version": 2,
        "bundle": "final_v13",
        "status": verifier.FINAL_MANIFEST_STATUS,
        "artifacts": artifacts,
        "pending_artifacts": [],
    }


def _source_document(
    paths: verifier.BundlePaths,
    sources: Sequence[Mapping[str, Any]],
    source_id: str,
    *,
    label: str,
) -> dict[str, Any]:
    matches = [record for record in sources if record["id"] == source_id]
    if len(matches) != 1:
        raise verifier.BundleVerificationError(f"required source is not unique: {source_id}")
    source_path = verifier.parent._lexical_source_path(str(matches[0]["path"]), paths)
    value, _ = verifier._strict_stable_json(
        source_path,
        label=label,
        repo=paths.repo,
    )
    return value


def _require_fine_full_consistency(fine: Mapping[str, Any], full: Mapping[str, Any]) -> None:
    fine_rungs = fine["fine"]["rungs"]
    fixed_rungs = full["fixed"]["rungs"]
    for rung in verifier.AIM3_RUNG_CONTROL:
        fine_record = fine_rungs[rung]
        fixed_record = fixed_rungs[rung]
        if fine_record["per_seed_auroc"] != fixed_record["fine_per_seed"]:
            raise verifier.BundleVerificationError(f"Aim-3 fine/full per-seed source drift: {rung}")
        phase1_estimate = float(fine_record["five_seed_ensemble"]["estimate"])
        full_estimate = float(fixed_record["fine"]["estimate"])
        if not math.isclose(
            phase1_estimate,
            full_estimate,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise verifier.BundleVerificationError(
                f"Aim-3 fine/full five-seed estimate drift: {rung}"
            )


def _firewall_block() -> str:
    return phase1._firewall_block()


def _priority_1_setup() -> str:
    return f"""{verifier.PRIORITY_HEADINGS[0]}

The FINAL-v13 main development population is exactly 1,239 conventional-primary
patients (501 KRAS-mutant, 738 wild type; 1,389 slides) from TCGA-COAD,
TCGA-READ, SR386, and SR1482. The main UNI-v1 and encoder-sensitivity
Virchow2-CLS ABMIL arms use the same patient-stratified five-fold splits and
model seeds 42--46. OOF native logits define development performance; one
source-only p75-epoch refit per seed and encoder exists only for frozen target
scoring. Inherited ALL-primary models that used CPTAC or RIH remain historical
continuity evidence and cannot develop or select the FINAL-v13 main model.

The whole Aim-1 pipeline is retained: E0 gene-level KRAS OOF ranking and paired
encoder contrasts; E1a fixed-score restrictions over molecular, site, stage,
and sidedness populations; E1a-S common-support acquisition/composition
standardization; the four zero-MIL-fit Why-D matched-restriction, pairwise,
molecular-context, and adjusted-association diagnostics; E1d cross-fitted
routine-clinical and fixed late-fusion comparators; E1v encoder and bag-cap
sensitivities; E1e MSI/dMMR positive control; retrospective fixed-capacity
worklist enrichment and decision curves; and frozen-score extended-RAS, MAPK,
and pathway-quiet relabelings. E1b did not fire; E1c and E0b were not run; the
all-primary-plus-Orion campaign is audit-only. None of these branches may use
external-target outcomes to choose the main model, encoder, construction,
calibration, or threshold."""


def _priority_2_setup() -> str:
    return f"""{verifier.PRIORITY_HEADINGS[1]}

Five full-source TCGA+SurGen-primary refits per encoder were frozen before
outcome joins, then scored CPTAC-primary, Orion, RIH-primary, RIH-metastatic,
and SurGen-metastatic in a 2-encoder x 5-seed x 5-target grid: 50 label-blind
score files and 4,790 slide rows. All five rosters are test-only under the main
model firewall. CPTAC and RIH-primary are family-naive primary validation;
RIH-metastatic is family-naive retrospective; SurGen-metastatic is
source-family exposed because SR1482 primary trained the model; Orion is a
retrospective cross-protocol test because study-level outcomes were previously
accessible. Those qualifications never authorize target feedback.

The inherited construction robustness analysis compares, within every encoder
and seed, the prespecified all-source p75-budget refit with the equal-native-
logit ensemble of that seed's five source-CV best checkpoints. It added zero
fits and reused 250 label-blind checkpoint-target inference passes in 50 files
containing 23,950 checkpoint-slide rows. Seed-specific patient AUROCs are
reported in seed order 42--46, followed by arithmetic mean and sample SD
(`ddof=1`); the SD is seed/partition dispersion, not a confidence interval.
The comparison jointly changes source training fraction, epoch/checkpoint
selection, and ensembling, was derived after study-level outcome access, and
therefore cannot select a deployment construction or supersede the frozen
five-seed mean-native-logit target estimand."""


def render_experimental_setup() -> str:
    priority_5 = f"""{verifier.PRIORITY_HEADINGS[4]}

The controlling population is TCGA+SurGen primaries only and the encoder is
UNI-v1. Five fine tasks, their canonical matched-WT controls, and three
prespecified repeated-WT draws each use model seeds 42--46. The completed
campaign comprises 125 chains: 25 fine, 25 canonical-control, and 75 repeated-
control chains; equivalently 625 OOF folds, 125 p75 refits, and 750 physical
MIL fits. Six chains ran concurrently. The earlier append-only two-control
checkpoint is retained as authenticated provenance but does not alter these
controlling full-campaign counts. Before controls, the 25 frozen fine-task
refits generated the separately sealed, zero-fit, 25-job external score grid
over CPTAC, Orion, both RIH roles, and SurGen-metastatic; target outcomes were
joined only after inference sealing. Consensus requires all three repeated
draws and is replayed from the draw-level gate verdicts.

{verifier.AIM3_CEILING_SCOPE_MARKER}

{verifier.AIM3_FIXED_NONOVERRIDE_MARKER}"""
    return (
        "\n\n".join(
            (
                "# FINAL-v13 experimental setup",
                "FINAL-v13 is an additive child of sealed FINAL-v12.1. The complete "
                "bundle retains the parent's exact 142-source ledger and adds 87 "
                "material sources. It has zero pending governed source artifacts; "
                "deliberately not-run/ineligible components are disclosed explicitly.",
                _firewall_block(),
                phase1._priority_1_setup(),
                phase1._priority_2_setup(),
                phase1._priority_3_setup(),
                phase1._priority_4_setup()
                + "\n\n"
                + verifier.THEORETICAL_CEILING_NOT_RUN_MARKER
                + ". No source-anchored theoretical-ceiling experiment was run. "
                "The pure/residual procedures estimate adaptation, not a theoretical-"
                "ceiling estimand. Their target-label use separately makes them "
                "target-internal and not external validation; a source-anchored "
                "target-label ceiling would likewise be ineligible as external "
                "validation and is unreported.",
                priority_5,
                phase1._priority_6_setup(),
            )
        )
        + "\n"
    )


def _summary_display(summary: Mapping[str, Any]) -> str:
    estimate = float(summary["estimate"])
    interval = summary.get("ci95_two_sided")
    if (
        isinstance(interval, list)
        and len(interval) == 2
        and all(isinstance(value, (int, float)) for value in interval)
    ):
        return f"{estimate:.4f} [{float(interval[0]):.4f}, {float(interval[1]):.4f}]"
    return f"{estimate:.4f}"


def _fixed_table(fine: Mapping[str, Any], full: Mapping[str, Any]) -> str:
    census = fine["fine"]["rungs"]
    fixed = full["fixed"]["rungs"]
    rows = []
    for rung in phase1.FINE_DISPLAY_ORDER:
        record = fixed[rung]
        gate = record.get("gate")
        verdict = gate.get("verdict", "—") if isinstance(gate, dict) else "—"
        rows.append(
            f"| {phase1.FINE_DISPLAY_NAMES[rung]} | "
            f"{census[rung]['patients']} ({census[rung]['positive']}/"
            f"{census[rung]['negative']}) | {_summary_display(record['fine'])} | "
            f"{_summary_display(record['control'])} | "
            f"{_summary_display(record['delta_control_minus_fine'])} | {verdict} |"
        )
    return "\n".join(
        (
            "| Task | Patients (positive/negative) | Fine AUROC | Canonical "
            "matched-WT AUROC | Control - fine AUROC | Gate verdict |",
            "|---|---:|---:|---:|---:|---|",
            *rows,
        )
    )


def _repeated_table(full: Mapping[str, Any]) -> str:
    repeated = full["repeated"]["rungs"]
    rows = []
    for rung in phase1.FINE_DISPLAY_ORDER:
        for draw_seed in verifier.WT_DRAW_SEEDS:
            draw = repeated[rung]["draws"][str(draw_seed)]
            rows.append(
                f"| {phase1.FINE_DISPLAY_NAMES[rung]} | {draw_seed} | "
                f"{_summary_display(draw['fine'])} | "
                f"{_summary_display(draw['control'])} | "
                f"{_summary_display(draw['delta_control_minus_fine'])} | "
                f"{draw['gate']['verdict']} |"
            )
    return "\n".join(
        (
            "| Task | WT draw seed | Fine AUROC | Repeated matched-WT AUROC | "
            "Control - fine AUROC | Draw verdict |",
            "|---|---:|---:|---:|---:|---|",
            *rows,
        )
    )


def _consensus_table(full: Mapping[str, Any]) -> str:
    repeated = full["repeated"]["rungs"]
    rows = [
        f"| {phase1.FINE_DISPLAY_NAMES[rung]} | {repeated[rung]['consensus_verdict']} |"
        for rung in phase1.FINE_DISPLAY_ORDER
    ]
    return "\n".join(
        (
            "| Task | Three-draw WT-control consensus |",
            "|---|---|",
            *rows,
        )
    )


def _binding_comments(full: Mapping[str, Any]) -> str:
    return "\n".join(
        f"<!-- AIM3_SOURCE_VALUE {binding['id']} {verifier._canonical_number(binding['value'])} -->"
        for binding in full["report_bindings"]
        if binding["metric"] != "consensus_verdict"
    )


def _consensus_comments(full: Mapping[str, Any]) -> str:
    repeated = full["repeated"]["rungs"]
    return "\n".join(
        f"<!-- AIM3_CONSENSUS {rung} {repeated[rung]['consensus_verdict']} -->"
        for rung in sorted(repeated)
    )


def _external_native_table() -> str:
    rows = [
        f"| {encoder} | {target} | {patients} | {auroc} |"
        for encoder, target, patients, auroc in EXTERNAL_NATIVE_ROWS
    ]
    return "\n".join(
        (
            "| Encoder | Frozen test population | Patients (mutant) | "
            "Five-seed mean-native-logit AUROC [95% CI] |",
            "|---|---|---:|---:|",
            *rows,
        )
    )


def _external_construction_table() -> str:
    rows = [
        f"| {encoder} | {target} | {patients} | {refit_values} | {refit_summary} | "
        f"{fold_values} | {fold_summary} | {role} |"
        for (
            encoder,
            target,
            patients,
            refit_values,
            refit_summary,
            fold_values,
            fold_summary,
            role,
        ) in EXTERNAL_CONSTRUCTION_ROWS
    ]
    return "\n".join(
        (
            "| Encoder | Frozen test population | Patients (mutant) | Refit "
            "seed AUROCs (42--46) | Refit mean ± SD | Within-seed fold5 "
            "AUROCs (42--46) | Fold5 mean ± SD | Qualification |",
            "|---|---|---:|---|---:|---|---:|---|",
            *rows,
        )
    )


def _priority_1_results() -> str:
    return f"""{verifier.PRIORITY_HEADINGS[0]}

### Whole Aim-1 pipeline summary

| Aim-1 component | FINAL-v13 scope | Authenticated inherited result |
|---|---|---|
| E0 main/baseline | TCGA+SurGen-primary OOF, seeds 42--46 | UNI-v1 AUROC 0.6929 [0.6619, 0.7219], AUPRC 0.6007 [0.5636, 0.6413]; Virchow2-CLS AUROC 0.6657 [0.6345, 0.6957], AUPRC 0.5750 [0.5389, 0.6154] |
| Paired encoder sensitivity | Same 1,239 patients and shared bootstrap | Virchow2-minus-UNI AUROC -0.0272 [-0.0487, -0.0058]; AUPRC -0.0257 [-0.0557, 0.0038] |
| E1a molecular restriction | Fixed OOF scores, D = MSS/pMMR plus BRAF-WT | D AUROC 0.7261 [0.6947, 0.7579] UNI and 0.7079 [0.6751, 0.7412] Virchow2; governed A-minus-D AUROC -0.0332 [-0.0496, -0.0178] and -0.0422 [-0.0599, -0.0255] |
| E1a-S composition standardization | Shared common-support subcohort x site weights | Standardized D-minus-A +0.0280 [0.0112, 0.0457] UNI and +0.0477 [0.0298, 0.0665] Virchow2 |
| Why-D | Matched restriction, pair decomposition, molecular-context distributions, adjusted OLS | Observed D-minus-A-complete lay beyond 10,000 matched random restrictions for both encoders (`p_ge_observed=9.999e-05`); explanatory association, not causality |
| E1d clinical value | Cross-fitted age/sex/site/stage comparator and fixed fusion | WSI-minus-clinical AUROC +0.1527 [0.1167, 0.1889] UNI and +0.1255 [0.0885, 0.1626] Virchow2; fixed fusion did not improve WSI |
| E1v and cap robustness | Encoder replication and 4,096/8,192-patch sensitivities | Source-primary-developed sensitivity evidence is retained; historical ALL-primary model performance is ineligible and not numerically reported |
| E1e positive control | MSI/dMMR endpoint | AUROC 0.9111 [0.8826, 0.9354]; pipeline positive control, not a KRAS effect |
| Worklist and DCA | Inherited frozen three-seed scores; retrospective | At top-30% capacity, development capture 0.445 [0.417, 0.476] and equal-target capture 0.466 [0.421, 0.508]; DCA bands were cohort-specific, with no universal threshold |
| Extended-RAS/MAPK/pathway-quiet | Frozen-score relabelings | Extended-RAS AUROC 0.6571 [0.6289, 0.6853], MAPK 0.6579 [0.6284, 0.6856], and KRAS versus pathway-quiet WT 0.6851 [0.6548, 0.7143]; pathway context, not treatment selection |
| Conditional/noncontrolling branches | E1b, E1c, E0b, all-primary-plus-Orion | E1b did not fire; E1c/E0b were not run; all-primary-plus-Orion is audit-only and supplies no selectable estimate |

Thus, Aim 1 supports a modest primary-tumor gene-level ranking signal, a
noncausal molecular-context dependency that survives composition
standardization, and incremental ranking/probability value over routine
clinical variables. It does not support target-informed model development,
late-fusion superiority, a universal operating threshold, causal morphology,
or replacement of molecular testing. Historical ALL-primary branches that
trained with CPTAC or RIH are isolated continuity evidence, not FINAL-v13 main
model development."""


def _aim2_primary_pool_binding_comments(
    primary_pool: Mapping[str, Any] | None,
) -> str:
    if primary_pool is None:
        return ""
    return "\n".join(
        "<!-- AIM2_PRIMARY_POOL_VALUE "
        f"{binding['id']} {verifier._canonical_number(binding['value'])} -->"
        for binding in primary_pool["bindings"]
    )


def _priority_2_results(primary_pool: Mapping[str, Any] | None) -> str:
    if primary_pool is None:
        headline = (
            "This nonproduction synthetic reconciliation does not contain the "
            "inherited governed patient-native-logit artifact, so it emits no "
            "conventional-primary pooled headline or numeric binding."
        )
        binding_comments = ""
    else:
        if (
            primary_pool.get("patients") != 247
            or primary_pool.get("mutant") != 103
            or primary_pool.get("wild_type") != 144
            or not math.isclose(
                float(primary_pool.get("auroc", math.nan)),
                verifier.EXPECTED_AIM2_PRIMARY_POOL_AUROC,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or primary_pool.get("cohort_auroc_averaging") is not False
            or primary_pool.get("governed_pooled_ci_available") is not False
        ):
            raise verifier.BundleVerificationError(
                "Aim-2 conventional-primary pooled render input drift"
            )
        headline = """| Encoder | Conventional external-primary patient pool | Patients (mutant/wild type) | Patient-pooled AUROC |
|---|---|---:|---:|
| UNI-v1 | CPTAC-primary + RIH-primary | 247 (103/144) | 0.745 |

The headline is one patient-pooled AUROC computed after concatenating the 247
patients and ranking their five-refit mean native logits. It is not an average
of the CPTAC-primary and RIH-primary cohort AUROCs. No governed pooled
confidence interval exists, so none is reported."""
        binding_comments = _aim2_primary_pool_binding_comments(primary_pool)
    return f"""{verifier.PRIORITY_HEADINGS[1]}

### Frozen five-seed external/test performance

{headline}

{binding_comments}

### Orion retrospective processing sensitivity

| Encoder | Retrospective cross-protocol population | Patients (mutant/wild type) | AUROC [95% CI] |
|---|---|---:|---:|
| UNI-v1 | Orion | 40 (15/25) | 0.7120 [0.5440, 0.8613] |

Orion remains separate from the conventional external-primary headline. Its
H&E slides followed multiple immunofluorescence staining cycles, so this is a
retrospective processing-transfer sensitivity with imprecise estimation, not
part of the CPTAC-plus-RIH-primary patient pool.

### Audit/supplementary cohort detail

{_external_native_table()}

These are the frozen deployment-refit, five-seed mean-native-logit results.
CPTAC-primary and RIH-primary are family-naive zero-shot validation targets.
RIH-metastatic is a family-naive retrospective sensitivity;
SurGen-metastatic is a source-family-exposed test because SR1482-primary enters
source training; Orion is retrospective cross-protocol evidence. Every roster
remains test-only for the main model and none selected an encoder, refit,
checkpoint construction, calibration, threshold, or hyperparameter.

### Post-outcome refit-versus-within-seed-fold5 robustness

{_external_construction_table()}

The vectors are ordered by seeds 42, 43, 44, 45, and 46. Each summary is
the arithmetic mean ± sample SD (`ddof=1`) of five seed-specific patient
AUROCs; SD is descriptive seed/partition dispersion, not a confidence
interval. Fold5 was not a uniform winner: it was lower for UNI-v1 CPTAC and
Virchow2-CLS Orion, nearly unchanged in some rows, and higher in others. This
authenticated table is a post-outcome robustness sensitivity only. Although
every inference artifact was generated label-blind, the comparison was
specified after study-level outcome access and jointly changes source training
fraction, epoch/checkpoint selection, and ensembling. It cannot justify
choosing a deployment construction after target outcomes, cannot supersede
the frozen five-seed target results above, and contributes no main-model
development feedback."""


def _priority_4_results(retained: Mapping[str, Any] | None) -> str:
    heading = f"""{verifier.PRIORITY_HEADINGS[3]}

{verifier.SECONDARY_MARKERS[1]}"""
    if retained is None:
        return f"""{heading}

No retained source-anchored target-internal fixture is present in this
nonproduction reconciliation.

{verifier.THEORETICAL_CEILING_NOT_RUN_MARKER}. No source-anchored theoretical-
ceiling experiment is represented, and no ceiling performance is reported.

Legacy non-source-anchored few-shot, local-training, and target-label ceiling
performance is not reported in FINAL-v13 Results; its provenance and exclusion
remain in Setup and Audit."""
    pure = retained["pure_ridge"]
    residual = retained["residual_ridge"]
    bindings = retained["target_internal_bindings"]
    return f"""{heading}

### Source-anchored combined-met TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION

{phase1._native_target_table(pure)}

{phase1._target_internal_table(pure, residual)}

{phase1._target_accounting_table(pure, residual)}

{phase1._target_binding_comments(bindings)}

Both methods use combined RIH-M plus SurGen-M support, k=2/4/8/16 per class
total, balanced equally by cohort within class. Each cell averages 100
support/fold procedures without prediction averaging. Confidence intervals use
a paired cohort-wise ordinary patient bootstrap, with one-class draws rejected;
outcome-class counts are not fixed by resampling. All budgets and both methods
are reported without target-driven selection. Every adapted point is below its
frozen native comparator; no positive repair claim is supported. This is not a
theoretical-ceiling estimand. Independently, target-label use makes every row
{phase1.TARGET_INTERNAL_ROLE}; SurGen-M is source-family exposed.

{verifier.THEORETICAL_CEILING_NOT_RUN_MARKER}. No source-anchored theoretical-
ceiling experiment was run, and the pure/residual few-shot analyses above are
not a ceiling. Such a target-label ceiling would be ineligible as external
validation under the firewall, and no ceiling performance is reported.

Legacy non-source-anchored few-shot, local-training, and target-label ceiling
performance is not reported in FINAL-v13 Results; its provenance and exclusion
remain in Setup and Audit."""


def render_results(
    fine: Mapping[str, Any],
    full: Mapping[str, Any],
    fine_external: Mapping[str, Any],
    retained_incremental: Mapping[str, Any] | None,
    *,
    aim2_primary_pool: Mapping[str, Any] | None = None,
    k32: Mapping[str, str],
    curated_from_sha256: str,
) -> str:
    parts = [
        "# FINAL-v13 results",
        "FINAL-v13 has zero pending governed source artifacts; deliberately "
        "not-run/ineligible components are disclosed explicitly. Aim-3 values "
        "below are rendered from authenticated primitive estimates; hidden "
        "comments provide exact machine-replay bindings.",
        _firewall_block(),
        phase1._priority_1_results(),
        _priority_2_results(aim2_primary_pool),
        f"""{verifier.PRIORITY_HEADINGS[2]}

{verifier.SECONDARY_MARKERS[0]}

No LOCO performance is reported in FINAL-v13 Results. Those historical arms
develop models on multiple cohort families, including cohorts reserved here as
external tests, and are therefore ineligible under the TCGA+SurGen-primary-only
model-development rule. Their authenticated provenance and exclusion remain in
the setup and audit only.""",
        _priority_4_results(retained_incremental),
        f"""{verifier.PRIORITY_HEADINGS[4]}

### Controlling TCGA+SurGen-primary UNI-v1 five-seed ladder

{_fixed_table(fine, full)}

{_binding_comments(full)}

### Frozen-refit fine-task external/test performance

#### Internal five-fold OOF and five-seed ensemble

{phase1._fine_external_internal_table(dict(fine_external))}

#### External refit-seed dispersion and five-refit ensemble

{phase1._fine_external_refit_table(dict(fine_external))}

{phase1._fine_external_binding_comments(dict(fine_external))}

All five external rosters were label-blind score-only targets and generated no
new fits. Mean ± SD is the arithmetic mean and sample SD (`ddof=1`) of five
fixed refit-seed patient AUROCs; the ensemble is the five-refit mean-native-
logit AUROC with a pointwise 10,000-resample patient-bootstrap CI. `Combined`
excludes all eight dual-role RIH patients from both roles. Primary-only pools
CPTAC, Orion, and RIH-primary; Met-only pools RIH-metastatic and SurGen-
metastatic. Sparse rows, especially `<10` in a class, are descriptive. No
external result selected a model, checkpoint, construction, calibration,
threshold, or any matched-WT experiment.

### Repeated three-draw WT-control consensus

{_repeated_table(full)}

{_consensus_table(full)}

{_consensus_comments(full)}

{verifier.AIM3_GATE_QUALIFICATION_MARKER}

{verifier.AIM3_CEILING_SCOPE_MARKER}

{verifier.AIM3_FIXED_NONOVERRIDE_MARKER}

This distinction is outcome-relevant in two audit traps. Allele-2 draw
20260824 remains `INCONCLUSIVE`: its descriptive 95% delta interval lower bound
is +0.0118, but the governed one-sided 99% FWER lower bound is -0.0015. G12C
draw 20260825 remains `UNDERPOWERED`: its descriptive 95% control interval
lower bound is 0.5056, but the governed one-sided 99% FWER lower bound is
0.4923. Neither verdict may be inferred from the displayed 95% interval.

### Legacy all-valid-patient Aim 3 — isolated secondary

{verifier.SECONDARY_MARKERS[2]}

No all-valid-patient Aim-3 performance is reported in FINAL-v13 Results. That
legacy population includes CPTAC and RIH in model development and is therefore
ineligible under the TCGA+SurGen-primary-only rule. Its authenticated
provenance and exclusion remain in the setup and audit only.""",
        f"""{verifier.PRIORITY_HEADINGS[5]}

{verifier.AIM4_BOUNDARY_MARKER}

Prototype p17 was associated in Set A/D with AUROC 0.5605 (q=0.0010) and
0.5939 (q=6.30e-7); p28 with 0.5968 (q=3.18e-9) and 0.6012 (q=7.91e-8),
with p28 attention AUROC 0.6172 (q=1.54e-13). Human-machine montage review
yielded 5 agree, 5 partial, and 1 differ; p17 was robust in 9/9 vocabulary
variants and p28 in 8/9. Single-reader, montage-level, transductive, and
noncausal limitations apply.

reviews/k32/completed_review.md live SHA-256: `{k32["completed_review.md"]}`

reviews/k32/completed_review_structured.json curated-from SHA-256: `{curated_from_sha256}`""",
    ]
    return "\n\n".join(parts) + "\n"


def render_audit(manifest: Mapping[str, Any]) -> str:
    source_rows = "\n".join(
        f"| `{record['id']}` | `{record['sha256']}` |" for record in manifest["artifacts"]
    )
    priority_text = "\n\n".join(
        (
            f"{verifier.PRIORITY_HEADINGS[0]}\n\nInherited sealed source-only Aim-1 evidence.",
            f"{verifier.PRIORITY_HEADINGS[1]}\n\nInherited frozen-score target evidence.",
            f"{verifier.PRIORITY_HEADINGS[2]}\n\nInherited, isolated LOCO evidence.",
            f"{verifier.PRIORITY_HEADINGS[3]}\n\nPure/residual combined-met evidence retained as TARGET_INTERNAL_NOT_EXTERNAL_VALIDATION; legacy non-source-anchored methods excluded numerically. {verifier.THEORETICAL_CEILING_NOT_RUN_MARKER}; the few-shot analyses are not a ceiling.",
            f"{verifier.PRIORITY_HEADINGS[4]}\n\nCompleted fine, frozen-refit external, canonical-control, and three-draw repeated-control evidence.\n\n{verifier.AIM3_CEILING_SCOPE_MARKER}\n\n{verifier.AIM3_FIXED_NONOVERRIDE_MARKER}",
            f"{verifier.PRIORITY_HEADINGS[5]}\n\nInherited statistics plus exact raw reviews/k32 provenance.",
        )
    )
    return f"""# FINAL-v13 evidence audit

FINAL-v13 inherits exactly 142 immutable FINAL-v12.1 records and adds exactly
87 directly rehashed material records: 229 scientific sources. It has zero
pending governed source artifacts; deliberately not-run/ineligible components
are disclosed explicitly.

{priority_text}

{verifier.FAILED_V1_AUDIT_MARKER}

{verifier.RECONCILIATION_PINS_BOUNDARY_MARKER}

`{verifier.RECONCILIATION_PINS_NAME}` is strict pre-seal metadata, deliberately
excluded from the scientific manifest and source index to break the verifier ->
document -> verifier self-hash cycle. The final receipt binds its byte identity.

{verifier.AIM3_TERMINAL_ANALYSIS_COMPLETION_MARKER}

The v2 root is controlling. The main external-data firewall is enforced by its
contract, frozen source lineage, preflight, and terminal completion receipts.
Pure and residual combined-met contracts, schedulers, predictions, terminal
results, and exact target-internal bindings remain authenticated and explicitly
non-external; SurGen-M remains source-family exposed and no adaptation result
can modify or select the source model.

The reviews/k32 Zone.Identifier sidecar is authenticated but excluded from the
scientific ledger.

## Exact source index

| Source ID | SHA-256 |
|---|---|
{source_rows}

## Pending source index

None. FINAL-v13 has zero pending governed source artifacts. Deliberately
not-run/ineligible components are disclosed above and are not pending evidence.
"""


def build_reconciliation_pins(
    paths: verifier.BundlePaths,
    documents: Mapping[str, str],
    extension_ids: Sequence[str],
) -> dict[str, Any]:
    document_hashes = {
        name: _sha256_bytes(documents[name].encode("utf-8")) for name in verifier.REPORT_DOCUMENTS
    }
    return {
        "schema_version": 1,
        "bundle": "final_v13",
        "status": verifier.RECONCILIATION_PINS_STATUS,
        "parent_final_v12_1": {
            "status": paths.expected_parent_status,
            "receipt_sha256": paths.expected_parent_receipt_sha256,
            "source_manifest_sha256": paths.expected_parent_manifest_sha256,
        },
        "extension_source_count": FULL_EXTENSION_SOURCE_COUNT,
        "extension_source_ids": list(extension_ids),
        "aim3_source_ids": {
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
        },
        "documents": document_hashes,
        "verifier": {"sha256": verifier.sha256_file(paths.verifier_code)},
        "verifier_test": {"sha256": verifier.sha256_file(paths.verifier_test)},
        "scientific_source_manifest_membership": (verifier.RECONCILIATION_PINS_MEMBERSHIP),
    }


def _patch_guidance(pins: Mapping[str, Any]) -> dict[str, Any]:
    aim3 = pins["aim3_source_ids"]
    return {
        "self_hash_cycle_resolution": (
            "Install reconciliation_pins.json; do not embed result-dependent document "
            "hashes in final_v13_bundle_receipt.py. The receipt binds pin-file bytes."
        ),
        "terminal_phase2_identity_freeze": {
            source_id: {
                "size_bytes": EXPECTED_TERMINAL_PHASE2_IDENTITIES[source_id][0],
                "sha256": EXPECTED_TERMINAL_PHASE2_IDENTITIES[source_id][1],
            }
            for source_id in TERMINAL_PHASE2_SOURCE_IDS
        },
        "install_targets": {
            verifier.SOURCE_MANIFEST_NAME: (f"reports/final_v13/{verifier.SOURCE_MANIFEST_NAME}"),
            **{name: f"reports/final_v13/{name}" for name in verifier.REPORT_DOCUMENTS},
            verifier.RECONCILIATION_PINS_NAME: (
                f"reports/final_v13/{verifier.RECONCILIATION_PINS_NAME}"
            ),
        },
        "resolved_verifier_expectations": {
            "EXPECTED_FINAL_DOCUMENT_SHA256": pins["documents"],
            "EXPECTED_EXTENSION_SOURCE_IDS": pins["extension_source_ids"],
            "EXPECTED_AIM3_RESULTS_SOURCE_ID": aim3["results"],
            "EXPECTED_AIM3_SCHEDULER_SOURCE_ID": aim3["scheduler"],
            "EXPECTED_AIM3_TRAINING_SOURCE_ID": aim3["training_completion"],
            "EXPECTED_AIM3_FINE_RESULTS_SOURCE_ID": aim3["fine_results"],
            "EXPECTED_AIM3_FINE_SCHEDULER_SOURCE_ID": aim3["fine_scheduler"],
            "EXPECTED_AIM3_FINE_TRAINING_SOURCE_ID": aim3["fine_training_completion"],
            "EXPECTED_AIM3_CONTROLS_SCHEDULER_SOURCE_ID": aim3["controls_scheduler"],
            "EXPECTED_AIM3_FINE_EXTERNAL_RESULTS_SOURCE_ID": aim3["fine_external_results"],
            "EXPECTED_AIM3_FINE_EXTERNAL_COMPLETION_SOURCE_ID": aim3[
                "fine_external_analysis_completion"
            ],
            "EXPECTED_AIM3_FINE_EXTERNAL_INFERENCE_SEAL_SOURCE_ID": aim3[
                "fine_external_inference_seal"
            ],
            "EXPECTED_AIM3_FINE_EXTERNAL_SCORING_SOURCE_ID": aim3[
                "fine_external_scoring_completion"
            ],
        },
        "verification_order": [
            "wait for all four terminal Phase-2 outputs and campaign-native verification",
            "freeze their exact size/SHA-256 identities in EXPECTED_TERMINAL_PHASE2_IDENTITIES",
            "install the five rendered files without creating a receipt",
            "run focused verifier/reconciler tests",
            "python tools/final_v13_full_reconciler.py check",
            "python tools/final_v13_bundle_receipt.py --check",
            "only then run python tools/final_v13_bundle_receipt.py --seal exactly once",
            "run python tools/final_v13_bundle_receipt.py --verify-published",
        ],
    }


def build_full_payload(
    paths: verifier.BundlePaths | None = None,
    extensions: Sequence[ExtensionSource] | None = None,
) -> dict[str, Any]:
    """Validate completed evidence and render every full reconciliation byte."""

    selected = reconciliation_paths() if paths is None else paths
    selected_extensions = extension_sources() if extensions is None else tuple(extensions)
    manifest = build_source_manifest(selected, selected_extensions)
    sources = manifest["artifacts"]
    verifier._validate_aim3_fine_results(selected, sources)
    verifier._validate_fine_execution_evidence(selected, sources)
    verifier._validate_aim3_fine_external_results(selected, sources)
    verifier._validate_aim3_results(selected, sources)
    verifier._validate_execution_evidence(selected, sources)
    source_ids = {str(record["id"]) for record in sources}
    aim2_primary_pool = (
        verifier._validate_aim2_conventional_primary_pool(selected, sources)
        if (
            verifier._is_production_bundle(selected)
            or verifier.EXPECTED_AIM2_PRIMARY_POOL_SOURCE_ID in source_ids
        )
        else None
    )
    retained_incremental = (
        verifier._validate_retained_incremental_evidence(selected, sources)
        if set(verifier.EXPECTED_INCREMENTAL_ADDITION_SOURCE_IDS).issubset(source_ids)
        else None
    )
    aim4 = verifier._validate_aim4_k32(selected, sources)
    fine = _source_document(
        selected,
        sources,
        selected.aim3_fine_results_source_id,
        label="Aim-3 fine-only results",
    )
    full = _source_document(
        selected,
        sources,
        selected.aim3_results_source_id,
        label="Aim-3 complete results",
    )
    fine_external = _source_document(
        selected,
        sources,
        selected.aim3_fine_external_results_source_id,
        label="Aim-3 fine-external results",
    )
    _require_fine_full_consistency(fine, full)
    documents = {
        "Experimental_Setup.md": render_experimental_setup(),
        "Results.md": render_results(
            fine,
            full,
            fine_external,
            retained_incremental,
            aim2_primary_pool=aim2_primary_pool,
            k32=dict(selected.expected_aim4_k32_sha256),
            curated_from_sha256=selected.expected_aim4_curated_from_sha256,
        ),
        "Audit.md": render_audit(manifest),
    }
    extension_ids = tuple(source.source_id for source in selected_extensions)
    pins = build_reconciliation_pins(selected, documents, extension_ids)
    if aim4["boundary"] != verifier.AIM4_BOUNDARY_MARKER:
        raise verifier.BundleVerificationError("Aim-4 provenance boundary drift")
    return {
        "schema_version": 1,
        "status": "FULL_RECONCILIATION_RENDERED_READ_ONLY",
        "scientific_source_count": FULL_SOURCE_COUNT,
        "parent_source_count": FULL_PARENT_SOURCE_COUNT,
        "extension_source_count": FULL_EXTENSION_SOURCE_COUNT,
        "pending_source_count": 0,
        "retained_target_internal_binding_count": (
            len(retained_incremental["target_internal_bindings"])
            if retained_incremental is not None
            else 0
        ),
        "aim2_conventional_primary_pool_binding_count": (
            len(aim2_primary_pool["bindings"]) if aim2_primary_pool is not None else 0
        ),
        "source_manifest": manifest,
        "documents": documents,
        "reconciliation_pins": pins,
        "patch_guidance": _patch_guidance(pins),
    }


def _require_exact_installed_bytes(
    path: Path,
    expected: bytes,
    *,
    label: str,
) -> None:
    verifier.parent._reject_symlink_chain(path, context=label)
    if not path.is_file() or path.read_bytes() != expected:
        raise verifier.BundleVerificationError(
            f"installed {label} differs from deterministic full reconciliation"
        )


def check_full_bundle(
    paths: verifier.BundlePaths | None = None,
    extensions: Sequence[ExtensionSource] | None = None,
) -> dict[str, Any]:
    """Byte-check installed renderings, then run the production verifier."""

    selected = reconciliation_paths() if paths is None else paths
    payload = build_full_payload(selected, extensions)
    _require_exact_installed_bytes(
        selected.final_v13 / verifier.SOURCE_MANIFEST_NAME,
        _json_bytes(payload["source_manifest"]),
        label="FINAL-v13 source manifest",
    )
    for name, text in payload["documents"].items():
        _require_exact_installed_bytes(
            selected.final_v13 / name,
            text.encode("utf-8"),
            label=f"FINAL-v13 {name}",
        )
    _require_exact_installed_bytes(
        selected.reconciliation_pins,
        _json_bytes(payload["reconciliation_pins"]),
        label="FINAL-v13 reconciliation pins",
    )
    checked = verifier.check_bundle(selected)
    if checked.get("status") != "READY_TO_SEAL" or checked.get("source_count") != 229:
        raise verifier.BundleVerificationError(
            "production verifier did not return exact 229-source READY_TO_SEAL"
        )
    return {
        "schema_version": 1,
        "status": "FULL_RECONCILIATION_VERIFIED_READ_ONLY",
        "seal_ready": True,
        "source_count": 229,
        "pending_source_count": 0,
        "reconciliation_pins": checked["reconciliation_pins"],
        "production_verifier": checked,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("render", "check"),
        help="read-only action; this helper never writes or seals",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    value = build_full_payload() if args.action == "render" else check_full_bundle()
    print(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
