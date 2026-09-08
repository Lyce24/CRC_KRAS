#!/usr/bin/env python3
"""Verify and seal the clean, aim-focused final-v9 report bundle.

The default command is read-only: it verifies the published receipt against
the current report documents, source manifest, authoritative source files,
and verifier implementation.  ``--seal`` publishes the receipt exactly once
using an atomic hard-link operation.  The verifier deliberately does not
follow nested receipts or assume append-only inheritance from an older report.

``reports/final_v9/source_manifest.json`` is the explicit upstream boundary::

    {
      "schema_version": 1,
      "bundle": "final_v9",
      "artifacts": [{
        "id": "aim1-sealed-replay",
        "aims": ["Aim 1"],
        "experiments": ["E0"],
        "role": "controlling_results",
        "path": "reports/reruns/.../results.json",
        "size_bytes": 123,
        "sha256": "..."
      }]
    }

Repository-relative and absolute paths are accepted.  Every listed identity is
rehashed directly, while receipt-shaped sources are treated as ordinary files;
their potentially large artifact trees are not traversed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FINAL_V9 = REPO / "reports" / "final_v9"
REPORT_DOCUMENTS = ("Experimental_Setup.md", "Results.md", "Audit.md")
SOURCE_MANIFEST_NAME = "source_manifest.json"
FINAL_RECEIPT_NAME = "report_bundle_receipt.json"
SEALED_STATUS = "SEALED_COMPLETED_RESULTS_WITH_DECLARED_NOT_RUN_ARM"
OFFICIAL_CPHT_NAME = (
    "Cross-Protocol H&E Transfer (E2-CPHT): conventional primary colorectal H&E "
    "→ same-section H&E after multiplex immunofluorescence (Orion cohort)"
)
REQUIRED_AIMS = ("Aim 1", "Aim 2", "Aim 3", "Aim 4")
MODEL_SEEDS = [42, 43, 44, 45, 46]
ADOPTED_MODEL_SEEDS = [42, 43, 44]
NEW_MODEL_SEEDS = [45, 46]
FIVE_SEED_CAMPAIGN_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/final_v9_mil_5seed_expansion_v1_20260823"
)
FIVE_SEED_ADJUDICATION_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/final_v9_adjudication/aim2_e2a_five_seed"
)

# These sources retain the parts of the study that were deliberately not
# expanded.  Requiring them prevents a nominally five-seed manifest from
# silently dropping the three-seed and otherwise unaffected scope.
REQUIRED_MIXED_SCOPE_SOURCE_IDS = frozenset(
    {
        "aim1-sealed-replay-results",
        "aim1-encoder-and-control-results",
        "aim1-worklist-results",
        "aim1-ras-composite-results",
        "aim1-decision-curve-results",
        "aim2-between-slide-results",
        "aim2-e2met-results",
        "aim2-e2e-results",
        "aim2-e2f-v3-results",
        "aim2-e2cpht-results",
        "aim2-e2cpht-a-v2-results",
    }
)

# Once their five-seed replacements are present, these historical result files
# are adoption/continuity evidence rather than the controlling result for the
# upgraded experiment.  A manifest role must make that precedence explicit.
SUPERSEDED_BY_FIVE_SEED_SOURCE_IDS = frozenset(
    {
        "aim1-sealed-replay-results",
        "aim1-encoder-and-control-results",
        "aim2-e2a-family-results",
        "aim2-e2ad-results",
        "aim2-e2met-results",
        "aim3-fixed-control-results",
        "aim3-repeated-control-results",
        "aim3-e3v-results",
    }
)
_NONCONTROLLING_ROLE_MARKERS = (
    "adopted",
    "continuity",
    "legacy",
    "mixed_scope",
    "superseded",
    "unchanged",
)

_SIBLING_ARMS = (
    "sibling_sr386",
    "sibling_sr1482",
    "sibling_tcga_coad",
    "sibling_tcga_read",
)
AIM2_CONTROLLING_ARMS = (
    "family_cptac",
    "family_rih",
    "family_surgen",
    "family_tcga",
    *_SIBLING_ARMS,
)
AIM2_SENSITIVITY_ARM = "family_rih_sm"
AIM2_ALL_ARMS = (
    "family_cptac",
    "family_rih",
    "family_surgen",
    "family_tcga",
    AIM2_SENSITIVITY_ARM,
    *_SIBLING_ARMS,
)
AIM3_FIXED_TASKS = (
    "codon",
    "g12d_broad",
    "allele1",
    "allele2",
    "g12c",
    "ctrl_codon",
    "ctrl_g12d_broad",
    "ctrl_allele1",
    "ctrl_allele2",
    "ctrl_g12c",
)
AIM3_REPEATED_TASKS = (
    "ctrl_codon",
    "ctrl_g12d_broad",
    "ctrl_allele1",
    "ctrl_allele2",
    "ctrl_g12c",
)
AIM3_REPEATED_DRAW_SEEDS = (20260823, 20260824, 20260825)
AIM3_E3V_TASKS = (
    "codon",
    "ctrl_codon",
    "g12d_broad",
    "ctrl_g12d_broad",
    "allele1",
    "ctrl_allele1",
)
AIM3_E1V_TASKS = ("gene",)
_SIBLING_SLUGS = ("sr386", "sr1482", "tcga_coad", "tcga_read")
_PAIRED_SOURCE_KEYS = (
    "sibling_sr386_minus_family_surgen_SR386",
    "sibling_sr1482_minus_family_surgen_SR1482",
    "sibling_tcga_coad_minus_family_tcga_TCGA-COAD",
    "sibling_tcga_read_minus_family_tcga_TCGA-READ",
    "family_rih_sm_minus_family_rih",
)
SUPERSEDED_ADJUDICATION_POINTERS = (
    "/family_loco_standardized_macro/directional_gate",
    "/family_loco_standardized_macro/claim_family_loco_transport",
    "/sibling_loco_directional_gate",
    *(f"/primary/{arm}/auroc_ci95" for arm in _SIBLING_ARMS),
)
NONAUTHORITATIVE_PAIRED_CI_POINTERS = tuple(
    f"/paired_sibling_and_size_matched_contrasts/{key}/{field}"
    for key in _PAIRED_SOURCE_KEYS
    for field in ("ci_low", "ci_high")
)
RETAINED_PAIRED_POINT_POINTERS = tuple(
    f"/paired_sibling_and_size_matched_contrasts/{key}/delta_auroc" for key in _PAIRED_SOURCE_KEYS
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_AIM_HEADING_RE = re.compile(r"^##\s+Aim\s+([1-4])\b", re.MULTILINE)
_ORION_EXCLUSION_RE = re.compile(
    r"Orion\s+is\s+excluded\s+from\s+(?:the\s+)?canonical\s+Aim\s*1",
    re.IGNORECASE,
)
_CPHT_R_NOT_RUN_RE = re.compile(r"CPHT-R.{0,200}\bNOT[ _-]?RUN\b", re.IGNORECASE | re.DOTALL)
_WHOLE_SECTION_UNREAD_RE = re.compile(
    r"whole-section\s+pathology\s+validation.{0,240}\bGENERATED_UNREAD\b",
    re.IGNORECASE | re.DOTALL,
)
_PLACEHOLDER_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])(?:TODO|TBD|TK|FIXME)(?![A-Za-z0-9_])", re.IGNORECASE),
    re.compile(r"<\s*(?:PLACEHOLDER|INSERT(?:\s+[^>]*)?|FILL(?:\s+[^>]*)?)\s*>", re.IGNORECASE),
    re.compile(r"\{\{[^{}]+\}\}"),
    re.compile(r"\[\[\s*(?:PLACEHOLDER|TODO|TBD|TK)[^\]]*\]\]", re.IGNORECASE),
    re.compile(
        r"\b(?:REPLACE_ME|FILL_ME_IN|SHA256_HERE|HASH_HERE|RESULT_HERE|"
        r"INSERT_(?:VALUE|HASH|RESULT|TEXT))\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![0-9a-fA-F])0{64}(?![0-9a-fA-F])"),
)


class BundleVerificationError(RuntimeError):
    """A fail-closed final-v9 report, source, or receipt verification error."""


@dataclass(frozen=True)
class BundlePaths:
    """Filesystem locations used by the final-v9 verifier."""

    repo: Path
    final_v9: Path
    destination: Path
    verifier_code: Path
    verifier_test: Path
    campaign_root: Path = FIVE_SEED_CAMPAIGN_ROOT
    adjudication_root: Path = FIVE_SEED_ADJUDICATION_ROOT


def default_paths() -> BundlePaths:
    """Return production final-v9 paths."""
    return BundlePaths(
        repo=REPO,
        final_v9=FINAL_V9,
        destination=FINAL_V9 / FINAL_RECEIPT_NAME,
        verifier_code=Path(__file__).resolve(),
        verifier_test=REPO / "tests" / "test_final_v9_bundle_receipt.py",
        campaign_root=FIVE_SEED_CAMPAIGN_ROOT,
        adjudication_root=FIVE_SEED_ADJUDICATION_ROOT,
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    """Return the path, byte size, and SHA-256 identity of a regular file."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise BundleVerificationError(f"missing required file: {resolved}")
    return {
        "path": display_path if display_path is not None else str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise BundleVerificationError(f"duplicate JSON key in {label} {path}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except BundleVerificationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleVerificationError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleVerificationError(f"{label} must contain a JSON object: {path}")
    return value


def _relative_display(path: Path, repo: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo.resolve()))
    except ValueError:
        return str(resolved)


def _validate_report_document(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise BundleVerificationError(f"cannot read report document {path}: {exc}") from exc
    if not text.strip():
        raise BundleVerificationError(f"empty report document: {path}")

    headings = [f"Aim {match}" for match in _AIM_HEADING_RE.findall(text)]
    if headings != list(REQUIRED_AIMS):
        raise BundleVerificationError(
            f"{path.name} must contain exactly one ordered level-two heading for each of "
            f"{', '.join(REQUIRED_AIMS)}; found {headings}"
        )
    if OFFICIAL_CPHT_NAME not in text:
        raise BundleVerificationError(f"{path.name} is missing the exact official E2-CPHT name")
    if _CPHT_R_NOT_RUN_RE.search(text) is None:
        raise BundleVerificationError(f"{path.name} must declare CPHT-R as NOT RUN")
    if _WHOLE_SECTION_UNREAD_RE.search(text) is None:
        raise BundleVerificationError(
            f"{path.name} must declare whole-section pathology validation GENERATED_UNREAD"
        )
    for pattern in _PLACEHOLDER_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            raise BundleVerificationError(
                f"placeholder-like text in {path.name}: {match.group(0)!r}"
            )


def _validate_report_contract(paths: BundlePaths) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for filename in REPORT_DOCUMENTS:
        path = paths.final_v9 / filename
        _validate_report_document(path)
        documents[filename] = identity(path, display_path=_relative_display(path, paths.repo))

    setup = (paths.final_v9 / "Experimental_Setup.md").read_text(encoding="utf-8")
    if "max_concurrent_gpu_trainers=6" not in setup:
        raise BundleVerificationError(
            "Experimental_Setup.md must declare max_concurrent_gpu_trainers=6"
        )
    if _ORION_EXCLUSION_RE.search(setup) is None:
        raise BundleVerificationError(
            "Experimental_Setup.md must state that Orion is excluded from canonical Aim 1"
        )
    _validate_five_seed_audit_scope(paths.final_v9 / "Audit.md")
    return documents


def _audit_scope_paragraph(text: str, heading: str) -> str:
    normalized = text.replace("\u2013", "-").replace("\u2014", "-")
    match = re.search(
        rf"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?{re.escape(heading)}(?:\*\*)?\s*:\s*(.+)$",
        normalized,
    )
    if match is None:
        raise BundleVerificationError(f"Audit.md must declare {heading!r}")
    return match.group(1).strip()


def _require_scope_tokens(paragraph: str, tokens: tuple[str, ...], *, label: str) -> None:
    missing = [token for token in tokens if token.casefold() not in paragraph.casefold()]
    if missing:
        raise BundleVerificationError(f"Audit.md {label} declaration is missing {missing}")


def _validate_five_seed_audit_scope(path: Path) -> None:
    """Require one explicit, machine-checkable mixed-scope and fit ledger."""

    text = path.read_text(encoding="utf-8")
    five_seed = _audit_scope_paragraph(text, "Five-seed scope")
    _require_scope_tokens(
        five_seed,
        (
            "E0",
            "E1v",
            "all nine LOCO primary",
            "E2-MET",
            "Orion LOCO sensitivity",
            "fixed",
            "repeated",
            "E3v",
        ),
        label="five-seed scope",
    )
    three_seed = _audit_scope_paragraph(text, "Three-seed unchanged scope")
    _require_scope_tokens(
        three_seed,
        (
            "raw E2-CPHT",
            "E2-CPHT-A",
            "15-fold Orion sensitivity",
            "E2e",
            "E2f-v3",
            "between-slide",
            "detailed E2-MET role/organ",
            "unaffected Aim 1",
        ),
        label="three-seed unchanged scope",
    )
    census = _audit_scope_paragraph(text, "Study-wide MIL census")
    compact = census.casefold().replace(",", "")
    _require_scope_tokens(
        compact,
        (
            "1225",
            "735 adopted",
            "490 new",
            "102",
            "42, 43, 44, 45, 46".replace(",", ""),
            "patient/slide folds unchanged",
            "maximum concurrency 6",
            "observed peak 6",
        ),
        label="MIL census",
    )


def _require_string_list(value: Any, *, location: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise BundleVerificationError(f"{location} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise BundleVerificationError(f"{location} entries must be non-empty strings")
    if len(value) != len(set(value)):
        raise BundleVerificationError(f"{location} contains duplicate entries")
    return value


def _resolve_source_path(raw_path: str, paths: BundlePaths) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    resolved = (paths.repo / candidate).resolve()
    try:
        resolved.relative_to(paths.repo.resolve())
    except ValueError as exc:
        raise BundleVerificationError(
            f"relative source path escapes the repository: {raw_path}"
        ) from exc
    return resolved


def _required_five_seed_sources(paths: BundlePaths) -> dict[str, tuple[Path, str]]:
    campaign = paths.campaign_root
    aim1 = campaign / "aim1_e0"
    aim2 = campaign / "aim2_loco"
    aim3 = campaign / "aim3_ladders"
    adjudication = paths.adjudication_root
    return {
        "final-v9-five-seed-campaign-contract": (
            campaign / "campaign/experiment_contract.json",
            "Shared",
        ),
        "final-v9-five-seed-campaign-deep-preflight": (
            campaign / "campaign/receipts/deep_preflight.json",
            "Shared",
        ),
        "final-v9-five-seed-campaign-training-completion": (
            campaign / "campaign/receipts/training_complete.json",
            "Shared",
        ),
        "final-v9-five-seed-campaign-results-completion": (
            campaign / "campaign/receipts/five_seed_results_complete.json",
            "Shared",
        ),
        "aim1-e0-five-seed-contract": (
            aim1 / "contract/campaign_contract.json",
            "Aim 1",
        ),
        "aim1-e0-five-seed-training-validation": (
            aim1 / "receipts/five_seed_training_validation.json",
            "Aim 1",
        ),
        "aim1-e0-five-seed-results": (aim1 / "analysis/five_seed_results.json", "Aim 1"),
        "aim1-e0-five-seed-patient-logits": (
            aim1 / "analysis/five_seed_patient_native_logits.parquet",
            "Aim 1",
        ),
        "aim1-e0-five-seed-analysis-receipt": (aim1 / "analysis/receipt.json", "Aim 1"),
        "aim2-loco-five-seed-contract": (aim2 / "contract.json", "Aim 2"),
        "aim2-loco-five-seed-inference-seal": (
            aim2 / "inference/inference_seal.json",
            "Aim 2",
        ),
        "aim2-loco-five-seed-source-oof": (
            aim2 / "analysis/source_oof_five_seed.parquet",
            "Aim 2",
        ),
        "aim2-loco-five-seed-calibrators": (
            aim2 / "analysis/source_calibrators_five_seed.json",
            "Aim 2",
        ),
        "aim2-loco-five-seed-primary-patients": (
            aim2 / "analysis/primary_patient_scores_five_seed.parquet",
            "Aim 2",
        ),
        "aim2-loco-five-seed-metastatic-patients": (
            aim2 / "analysis/e2met_patient_scores_five_seed.parquet",
            "Aim 2",
        ),
        "aim2-loco-five-seed-orion-patients": (
            aim2 / "analysis/e2cpht_orion_patient_scores_five_seed.parquet",
            "Aim 2",
        ),
        "aim2-loco-five-seed-results": (aim2 / "analysis/results_five_seed.json", "Aim 2"),
        "aim2-loco-five-seed-table": (aim2 / "analysis/results_five_seed.csv", "Aim 2"),
        "aim2-loco-five-seed-report-receipt": (
            aim2 / "analysis/results_five_seed.receipt.json",
            "Aim 2",
        ),
        "aim2-e2a-five-seed-adjudication-result": (adjudication / "result.json", "Aim 2"),
        "aim2-e2a-five-seed-adjudication-bootstrap": (
            adjudication / "bootstrap_distributions.npz",
            "Aim 2",
        ),
        "aim2-e2a-five-seed-adjudication-receipt": (
            adjudication / "receipt.json",
            "Aim 2",
        ),
        "aim2-e2a-five-seed-adjudication-implementation": (
            paths.repo / "tools/aim2_e2a_five_seed_adjudication.py",
            "Aim 2",
        ),
        "aim2-e2a-five-seed-adjudication-test": (
            paths.repo / "tests/test_aim2_e2a_five_seed_adjudication.py",
            "Aim 2",
        ),
        "aim3-ladders-five-seed-contract": (
            aim3 / "inputs/experiment_contract.json",
            "Aim 3",
        ),
        "aim3-ladders-five-seed-results": (
            aim3 / "analysis/five_seed_results.json",
            "Aim 3",
        ),
        "aim3-ladders-five-seed-bootstrap": (
            aim3 / "analysis/bootstrap_distributions.npz",
            "Aim 3",
        ),
        "aim3-ladders-five-seed-analysis-audit": (
            aim3 / "analysis/analysis_audit.json",
            "Aim 3",
        ),
        "aim3-ladders-five-seed-completion": (
            aim3 / "receipts/extension_complete.json",
            "Aim 3",
        ),
    }


def _source_map(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(source["id"]): source for source in sources}


def _source_path(source: dict[str, Any], paths: BundlePaths) -> Path:
    return _resolve_source_path(str(source["path"]), paths)


def _source_json(
    sources: dict[str, dict[str, Any]], source_id: str, paths: BundlePaths
) -> dict[str, Any]:
    return _load_json(_source_path(sources[source_id], paths), label=f"source {source_id}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleVerificationError(message)


def _assert_source_identity(
    record: Any,
    source: dict[str, Any],
    paths: BundlePaths,
    *,
    context: str,
) -> None:
    if not isinstance(record, dict) or not {"path", "size_bytes", "sha256"}.issubset(record):
        raise BundleVerificationError(f"{context} lacks an artifact identity")
    try:
        record_path = _resolve_source_path(str(record["path"]), paths)
    except (TypeError, BundleVerificationError) as exc:
        raise BundleVerificationError(f"{context} has an invalid artifact path") from exc
    expected_path = _source_path(source, paths)
    if (
        record_path != expected_path
        or record.get("size_bytes") != source["size_bytes"]
        or record.get("sha256") != source["sha256"]
    ):
        raise BundleVerificationError(f"{context} does not bind the declared source artifact")


def _assert_source_id(
    record: Any,
    sources: dict[str, dict[str, Any]],
    source_id: str,
    paths: BundlePaths,
    *,
    context: str,
) -> None:
    _assert_source_identity(record, sources[source_id], paths, context=context)


def _validate_source_manifest(
    paths: BundlePaths,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = paths.final_v9 / SOURCE_MANIFEST_NAME
    manifest = _load_json(manifest_path, label="source manifest")
    if set(manifest) != {"schema_version", "bundle", "artifacts"}:
        raise BundleVerificationError(
            "source manifest keys must be exactly schema_version, bundle, artifacts"
        )
    if manifest["schema_version"] != 1 or manifest["bundle"] != "final_v9":
        raise BundleVerificationError(
            "source manifest must declare schema_version=1 and bundle=final_v9"
        )
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise BundleVerificationError("source manifest artifacts must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    represented_aims: set[str] = set()
    expected_keys = {
        "id",
        "aims",
        "experiments",
        "role",
        "path",
        "size_bytes",
        "sha256",
    }
    for index, artifact in enumerate(artifacts):
        location = f"source manifest artifact {index}"
        if not isinstance(artifact, dict) or set(artifact) != expected_keys:
            raise BundleVerificationError(
                f"{location} keys must be exactly {', '.join(sorted(expected_keys))}"
            )
        source_id = artifact["id"]
        if not isinstance(source_id, str) or _SOURCE_ID_RE.fullmatch(source_id) is None:
            raise BundleVerificationError(f"{location}.id is not a lowercase source ID")
        if source_id in seen_ids:
            raise BundleVerificationError(f"duplicate source manifest ID: {source_id}")
        seen_ids.add(source_id)

        aims = _require_string_list(artifact["aims"], location=f"{location}.aims")
        invalid_aims = set(aims) - set(REQUIRED_AIMS) - {"Shared"}
        if invalid_aims:
            raise BundleVerificationError(
                f"{location}.aims contains invalid values: {sorted(invalid_aims)}"
            )
        represented_aims.update(set(aims) & set(REQUIRED_AIMS))
        experiments = _require_string_list(
            artifact["experiments"], location=f"{location}.experiments"
        )
        role = artifact["role"]
        raw_path = artifact["path"]
        size_bytes = artifact["size_bytes"]
        sha256 = artifact["sha256"]
        if not isinstance(role, str) or not role.strip():
            raise BundleVerificationError(f"{location}.role must be a non-empty string")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise BundleVerificationError(f"{location}.path must be a non-empty string")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise BundleVerificationError(f"{location}.size_bytes must be a non-negative integer")
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise BundleVerificationError(f"{location}.sha256 must be lowercase SHA-256")

        resolved = _resolve_source_path(raw_path, paths)
        try:
            resolved.relative_to(paths.final_v9.resolve())
        except ValueError:
            pass
        else:
            raise BundleVerificationError(
                f"authoritative source must be upstream of final_v9: {raw_path}"
            )
        if resolved in seen_paths:
            raise BundleVerificationError(
                f"source file listed more than once; combine its aims/experiments: {raw_path}"
            )
        seen_paths.add(resolved)
        actual = identity(resolved, display_path=raw_path)
        if actual["size_bytes"] != size_bytes or actual["sha256"] != sha256:
            raise BundleVerificationError(
                f"authoritative source identity drift for {source_id}: {raw_path}"
            )
        normalized.append(
            {
                "id": source_id,
                "aims": aims,
                "experiments": experiments,
                "role": role,
                **actual,
            }
        )

    missing_aims = set(REQUIRED_AIMS) - represented_aims
    if missing_aims:
        raise BundleVerificationError(
            f"source manifest has no authoritative artifact for {sorted(missing_aims)}"
        )
    normalized.sort(key=lambda item: item["id"])
    _validate_five_seed_source_inventory(paths, normalized)
    return (
        identity(
            manifest_path,
            display_path=_relative_display(manifest_path, paths.repo),
        ),
        normalized,
    )


def _validate_five_seed_source_inventory(
    paths: BundlePaths, normalized: list[dict[str, Any]]
) -> None:
    sources = _source_map(normalized)
    requirements = _required_five_seed_sources(paths)
    missing = sorted(set(requirements) - set(sources))
    if missing:
        raise BundleVerificationError(f"source manifest lacks required five-seed IDs: {missing}")
    missing_mixed = sorted(REQUIRED_MIXED_SCOPE_SOURCE_IDS - set(sources))
    if missing_mixed:
        raise BundleVerificationError(
            f"source manifest lacks required unchanged/mixed-scope IDs: {missing_mixed}"
        )

    for source_id, (expected_path, expected_aim) in requirements.items():
        source = sources[source_id]
        if _source_path(source, paths) != expected_path.resolve(strict=False):
            raise BundleVerificationError(
                f"required five-seed source path mismatch for {source_id}: {source['path']}"
            )
        if source["aims"] != [expected_aim]:
            raise BundleVerificationError(
                f"required five-seed source {source_id} must declare aims=[{expected_aim!r}]"
            )

    for source_id in SUPERSEDED_BY_FIVE_SEED_SOURCE_IDS & set(sources):
        role = str(sources[source_id]["role"]).casefold()
        if "controlling" in role and not any(
            marker in role for marker in _NONCONTROLLING_ROLE_MARKERS
        ):
            raise BundleVerificationError(
                f"historical three-seed source {source_id} is still labeled controlling; "
                "declare its adopted/continuity/mixed-scope precedence"
            )

    for source_id in (
        "aim1-e0-five-seed-results",
        "aim2-loco-five-seed-results",
        "aim2-e2a-five-seed-adjudication-result",
        "aim3-ladders-five-seed-results",
    ):
        if "controlling" not in str(sources[source_id]["role"]).casefold():
            raise BundleVerificationError(
                f"new five-seed result {source_id} must have an explicit controlling role"
            )

    _validate_five_seed_source_semantics(paths, sources)


def _validate_five_seed_source_semantics(
    paths: BundlePaths, sources: dict[str, dict[str, Any]]
) -> None:
    try:
        campaign = _validate_campaign_sources(paths, sources)
        aim1_counts = _validate_aim1_sources(paths, sources)
        aim2_counts = _validate_aim2_sources(paths, sources)
        aim3_counts = _validate_aim3_sources(paths, sources)
        _validate_adjudication_sources(paths, sources)

        adopted = aim1_counts["adopted"] + aim2_counts["adopted"] + aim3_counts["adopted"]
        new = aim1_counts["new"] + aim2_counts["new"] + aim3_counts["new"]
        complete = aim1_counts["complete"] + aim2_counts["complete"] + aim3_counts["complete"]
        _require(
            (adopted, new, complete) == (735, 490, 1225),
            "study-wide MIL census must be 1,225 = 735 adopted + 490 new fits",
        )
        _require(campaign["new"] == new, "campaign and component new-fit census disagree")
    except BundleVerificationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise BundleVerificationError(f"malformed five-seed semantic source graph: {exc}") from exc


def _safe_campaign_job_name(job_id: str) -> str:
    return job_id.replace("/", "__").replace(":", "_")


def _aim3_logical_roster() -> set[tuple[str, str, int | None]]:
    return {
        *(("fixed", task, None) for task in AIM3_FIXED_TASKS),
        *(
            ("repeated", task, draw_seed)
            for task in AIM3_REPEATED_TASKS
            for draw_seed in AIM3_REPEATED_DRAW_SEEDS
        ),
        *(("e3v", task, None) for task in AIM3_E3V_TASKS),
        *(("e1v", task, None) for task in AIM3_E1V_TASKS),
    }


def _aim3_job_key(stage: str, task: str, draw_seed: Any, seed: int) -> str:
    draw = f"__wt{draw_seed}" if stage == "repeated" else ""
    return f"{stage}__{task}{draw}__seed{seed}"


def _expected_aim3_jobs(seeds: list[int]) -> set[tuple[str, str, int | None, int]]:
    return {
        (stage, task, draw_seed, seed)
        for stage, task, draw_seed in _aim3_logical_roster()
        for seed in seeds
    }


def _validate_campaign_job_inventory(jobs: Any, declared: Any) -> list[dict[str, Any]]:
    _require(isinstance(jobs, list) and len(jobs) == 102, "campaign must contain 102 jobs")
    normalized: list[dict[str, Any]] = []
    identifiers: list[str] = []
    by_component: dict[str, dict[str, int]] = {}
    stage_seed_counts: dict[tuple[str, str, int], int] = {}
    aim2_jobs: set[tuple[str, str, int]] = set()
    aim3_jobs: set[tuple[str, str, int | None, int]] = set()
    identities_bound = True
    for index, job in enumerate(jobs):
        _require(isinstance(job, dict), f"campaign job {index} must be an object")
        job_id = job.get("job_id")
        component = job.get("component")
        stage = job.get("stage")
        seed = job.get("seed")
        fit_count = job.get("fit_count")
        command = job.get("command")
        _require(
            isinstance(job_id, str)
            and bool(job_id)
            and component in {"aim1_e0", "aim2_loco", "aim3_ladders"}
            and isinstance(stage, str)
            and bool(stage)
            and seed in NEW_MODEL_SEEDS
            and isinstance(seed, int)
            and not isinstance(seed, bool)
            and isinstance(fit_count, int)
            and not isinstance(fit_count, bool)
            and fit_count > 0
            and isinstance(command, list)
            and bool(command)
            and all(isinstance(item, str) and bool(item) for item in command),
            f"campaign job {index} has invalid identity/component/stage/seed/fit count",
        )
        identifiers.append(job_id)
        summary = by_component.setdefault(component, {"jobs": 0, "new_fits": 0})
        summary["jobs"] += 1
        summary["new_fits"] += fit_count
        stage_seed_counts[(component, stage, seed)] = (
            stage_seed_counts.get((component, stage, seed), 0) + 1
        )
        if component == "aim1_e0":
            component_job_key = f"aim1_e0_seed{seed}"
            _require(
                stage == "baseline" and fit_count == 6,
                "Aim-1 top-level jobs changed",
            )
            identities_bound &= (
                job_id == f"aim1.{component_job_key}"
                and job.get("component_job_key") == component_job_key
            )
        elif component == "aim2_loco":
            arm = job.get("arm")
            component_job_key = f"aim2.{stage}.{arm}.seed{seed}"
            _require(
                isinstance(arm, str)
                and bool(arm)
                and (
                    (stage == "source_cv" and fit_count == 5)
                    or (stage == "refit" and fit_count == 1)
                ),
                f"Aim-2 top-level job changed: {job_id}",
            )
            identities_bound &= (
                job_id == component_job_key and job.get("component_job_key") == component_job_key
            )
            aim2_jobs.add((arm, stage, seed))
        else:
            task = job.get("task")
            draw_seed = job.get("draw_seed")
            component_job_key = _aim3_job_key(stage, str(task), draw_seed, seed)
            _require(
                stage in {"fixed", "repeated", "e3v", "e1v"}
                and isinstance(task, str)
                and bool(task)
                and fit_count == (5 if stage in {"e3v", "e1v"} else 6),
                f"Aim-3 top-level job changed: {job_id}",
            )
            identities_bound &= (
                job_id == f"aim3.{component_job_key}"
                and job.get("component_job_key") == component_job_key
            )
            aim3_jobs.add((stage, task, draw_seed, seed))
        normalized.append(job)

    _require(len(set(identifiers)) == 102, "campaign job IDs must be unique")
    expected_by_component = {
        "aim1_e0": {"jobs": 2, "new_fits": 12},
        "aim2_loco": {"jobs": 36, "new_fits": 108},
        "aim3_ladders": {"jobs": 64, "new_fits": 370},
    }
    _require(
        by_component == expected_by_component and declared == expected_by_component,
        "campaign declared and derived 2/36/64 component census disagree",
    )
    _require(
        identities_bound,
        "campaign job IDs and component job keys must bind exact job semantics",
    )
    expected_stage_seed = {
        ("aim1_e0", "baseline", 45): 1,
        ("aim1_e0", "baseline", 46): 1,
        **{
            ("aim2_loco", stage, seed): 9
            for stage in ("source_cv", "refit")
            for seed in NEW_MODEL_SEEDS
        },
        **{
            ("aim3_ladders", stage, seed): count
            for stage, count in {"fixed": 10, "repeated": 15, "e3v": 6, "e1v": 1}.items()
            for seed in NEW_MODEL_SEEDS
        },
    }
    _require(
        stage_seed_counts == expected_stage_seed,
        "campaign stage-by-seed distribution changed",
    )
    expected_aim2_jobs = {
        (arm, stage, seed)
        for arm in AIM2_ALL_ARMS
        for stage in ("source_cv", "refit")
        for seed in NEW_MODEL_SEEDS
    }
    _require(aim2_jobs == expected_aim2_jobs, "Aim-2 exact scheduler job roster changed")
    _require(
        aim3_jobs == _expected_aim3_jobs(NEW_MODEL_SEEDS),
        "Aim-3 exact scheduler task/draw/seed roster changed",
    )
    return normalized


def _validate_aim2_arm_roster(contract: dict[str, Any]) -> None:
    scope = contract.get("scope", {})
    arms = contract.get("arms")
    _require(
        scope.get("controlling_arms") == list(AIM2_CONTROLLING_ARMS)
        and scope.get("secondary_sensitivity") == [AIM2_SENSITIVITY_ARM],
        "Aim-2 exact eight-arm controlling roster plus RIH-SM sensitivity changed",
    )
    _require(
        isinstance(arms, dict)
        and set(arms) == set(AIM2_ALL_ARMS)
        and all(
            isinstance(arms[arm], dict) and arms[arm].get("name") == arm for arm in AIM2_ALL_ARMS
        ),
        "Aim-2 exact nine-arm component roster changed",
    )


def _validate_campaign_component_job_rosters(
    paths: BundlePaths,
    sources: dict[str, dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> None:
    aim2_contract = _source_json(sources, "aim2-loco-five-seed-contract", paths)
    _validate_aim2_arm_roster(aim2_contract)
    expected_aim2 = {
        (arm, stage, seed)
        for arm in aim2_contract["arms"]
        for stage in ("source_cv", "refit")
        for seed in NEW_MODEL_SEEDS
    }
    scheduler_aim2 = {
        (job["arm"], job["stage"], job["seed"]) for job in jobs if job["component"] == "aim2_loco"
    }
    _require(
        scheduler_aim2 == expected_aim2,
        "campaign Aim-2 scheduler jobs do not match the component contract roster",
    )

    aim3_contract = _source_json(sources, "aim3-ladders-five-seed-contract", paths)
    new_jobs = aim3_contract.get("new_jobs")
    _aim3_job_census(new_jobs, adopted=False)
    component_aim3 = {
        (job["component"], job["task"], job.get("draw_seed"), job["model_seed"]) for job in new_jobs
    }
    scheduler_aim3 = {
        (job["stage"], job["task"], job.get("draw_seed"), job["seed"])
        for job in jobs
        if job["component"] == "aim3_ladders"
    }
    _require(
        component_aim3 == scheduler_aim3 == _expected_aim3_jobs(NEW_MODEL_SEEDS),
        "campaign Aim-3 scheduler jobs do not match the exact component task roster",
    )


def _validate_campaign_job_receipts(
    paths: BundlePaths, jobs: list[dict[str, Any]], records: Any
) -> None:
    _require(isinstance(records, dict), "campaign training job_receipts must be an object")
    expected_ids = {str(job["job_id"]) for job in jobs}
    _require(
        set(records) == expected_ids,
        "campaign training job_receipts keys must exactly equal contract job IDs",
    )
    receipt_root = paths.campaign_root / "campaign/receipts/jobs"
    for job_id in sorted(expected_ids):
        record = records[job_id]
        expected_path = receipt_root / f"{_safe_campaign_job_name(job_id)}.json"
        _require(
            expected_path.is_file() and not expected_path.is_symlink(),
            f"missing regular campaign job receipt: {job_id}",
        )
        actual = identity(expected_path, display_path=str(expected_path.resolve()))
        _require(
            isinstance(record, dict)
            and set(record) == {"path", "size_bytes", "sha256"}
            and Path(str(record.get("path", ""))).resolve(strict=False)
            == expected_path.resolve(strict=False)
            and record.get("size_bytes") == actual["size_bytes"]
            and record.get("sha256") == actual["sha256"],
            f"campaign job receipt identity is not bound to live bytes: {job_id}",
        )
        payload = _load_json(expected_path, label=f"campaign job receipt {job_id}")
        contract_path = paths.campaign_root / "campaign/experiment_contract.json"
        contract_identity = identity(contract_path, display_path=str(contract_path.resolve()))
        receipt_contract = payload.get("contract")
        _require(
            payload.get("status") == "completed_rc0"
            and payload.get("job_id") == job_id
            and payload.get("returncode") == 0
            and payload.get("command")
            == next(job["command"] for job in jobs if job["job_id"] == job_id)
            and isinstance(receipt_contract, dict)
            and Path(str(receipt_contract.get("path", ""))).resolve(strict=False)
            == contract_path.resolve(strict=False)
            and receipt_contract.get("size_bytes") == contract_identity["size_bytes"]
            and receipt_contract.get("sha256") == contract_identity["sha256"],
            f"campaign job receipt semantics do not bind its contract job: {job_id}",
        )


def _validate_campaign_sources(
    paths: BundlePaths, sources: dict[str, dict[str, Any]]
) -> dict[str, int]:
    contract = _source_json(sources, "final-v9-five-seed-campaign-contract", paths)
    _require(
        contract.get("status") == "sealed_before_new_fit"
        and contract.get("experiment") == "final-v9 study-wide MIL five-seed expansion",
        "five-seed campaign contract has the wrong status or experiment",
    )
    _require(
        contract.get("model_seeds")
        == {"adopted": ADOPTED_MODEL_SEEDS, "new": NEW_MODEL_SEEDS, "complete": MODEL_SEEDS},
        "five-seed campaign model-seed roster changed",
    )
    split = contract.get("split_policy", {})
    _require(
        split.get("outer_and_inner_membership_changes_across_model_seeds") is False
        and "frozen patient/slide fold manifest" in str(split.get("rule", "")),
        "campaign must reuse unchanged patient/slide folds across all model seeds",
    )
    execution = contract.get("execution", {})
    _require(
        execution.get("max_concurrent_gpu_trainers") == 6
        and execution.get("fresh_process_per_chain") is True,
        "campaign execution contract must use six independent trainer processes",
    )
    counts = contract.get("counts", {})
    _require(
        counts.get("jobs") == 102
        and counts.get("new_mil_fits") == 490
        and isinstance(counts.get("by_component"), dict),
        "campaign job/fit census is not 102 jobs / 490 new MIL fits",
    )
    jobs = _validate_campaign_job_inventory(contract.get("jobs"), counts.get("by_component"))
    component_contracts = contract.get("component_contracts", {})
    for component, source_id in {
        "aim1_e0": "aim1-e0-five-seed-contract",
        "aim2_loco": "aim2-loco-five-seed-contract",
        "aim3_ladders": "aim3-ladders-five-seed-contract",
    }.items():
        _assert_source_id(
            component_contracts.get(component),
            sources,
            source_id,
            paths,
            context=f"campaign component contract {component}",
        )
    _validate_campaign_component_job_rosters(paths, sources, jobs)

    preflight = _source_json(sources, "final-v9-five-seed-campaign-deep-preflight", paths)
    _require(
        preflight.get("status") == "deep_preflight_passed"
        and preflight.get("maximum_concurrent_gpu_trainers") == 6
        and preflight.get("aim1_live_deep_contract_and_pack_authentication") is True,
        "campaign deep-preflight receipt is incomplete",
    )
    _assert_source_id(
        preflight.get("contract"),
        sources,
        "final-v9-five-seed-campaign-contract",
        paths,
        context="campaign preflight contract",
    )

    training = _source_json(sources, "final-v9-five-seed-campaign-training-completion", paths)
    _require(
        training.get("status") == "490 new MIL fits completed and certified"
        and training.get("new_mil_fits") == 490
        and training.get("new_training_jobs") == 102
        and training.get("maximum_concurrent_gpu_trainers") == 6
        and training.get("observed_peak_concurrent_gpu_trainers") == 6
        and training.get("reconstructed_peak_from_job_exit_intervals") == 6
        and isinstance(training.get("job_receipts"), dict),
        "campaign training completion must certify 102 jobs/490 fits and an observed peak of six",
    )
    _assert_source_id(
        training.get("contract"),
        sources,
        "final-v9-five-seed-campaign-contract",
        paths,
        context="campaign training contract",
    )
    _assert_source_id(
        training.get("preflight"),
        sources,
        "final-v9-five-seed-campaign-deep-preflight",
        paths,
        context="campaign training preflight",
    )
    _validate_campaign_job_receipts(paths, jobs, training.get("job_receipts"))

    final = _source_json(sources, "final-v9-five-seed-campaign-results-completion", paths)
    _require(
        final.get("status") == "five_seed_results_complete"
        and final.get("seeds") == MODEL_SEEDS
        and final.get("model_seeds_are_not_inference_units") is True,
        "campaign final receipt is not a complete five-seed result seal",
    )
    _assert_source_id(
        final.get("contract"),
        sources,
        "final-v9-five-seed-campaign-contract",
        paths,
        context="campaign final contract",
    )
    _assert_source_id(
        final.get("training"),
        sources,
        "final-v9-five-seed-campaign-training-completion",
        paths,
        context="campaign final training receipt",
    )
    component_sources = {
        "aim1_e0": {
            "training": "aim1-e0-five-seed-training-validation",
            "results": "aim1-e0-five-seed-results",
            "analysis_receipt": "aim1-e0-five-seed-analysis-receipt",
        },
        "aim2_loco": {
            "inference_seal": "aim2-loco-five-seed-inference-seal",
            "analysis_source_oof": "aim2-loco-five-seed-source-oof",
            "analysis_calibrators": "aim2-loco-five-seed-calibrators",
            "analysis_primary_patients": "aim2-loco-five-seed-primary-patients",
            "analysis_met_patients": "aim2-loco-five-seed-metastatic-patients",
            "analysis_orion_patients": "aim2-loco-five-seed-orion-patients",
            "analysis_results": "aim2-loco-five-seed-results",
            "analysis_table": "aim2-loco-five-seed-table",
            "analysis_receipt": "aim2-loco-five-seed-report-receipt",
        },
        "aim3_ladders": {
            "results": "aim3-ladders-five-seed-results",
            "bootstrap_distributions": "aim3-ladders-five-seed-bootstrap",
            "analysis_audit": "aim3-ladders-five-seed-analysis-audit",
            "completion": "aim3-ladders-five-seed-completion",
        },
    }
    components = final.get("components", {})
    _require(set(components) == set(component_sources), "campaign final component roster changed")
    for component, expected in component_sources.items():
        observed = components.get(component, {})
        _require(
            set(observed) == set(expected), f"campaign final {component} artifact roster changed"
        )
        for key, source_id in expected.items():
            _assert_source_id(
                observed[key],
                sources,
                source_id,
                paths,
                context=f"campaign final {component}/{key}",
            )
    return {"new": 490}


def _validate_aim1_sources(
    paths: BundlePaths, sources: dict[str, dict[str, Any]]
) -> dict[str, int]:
    contract = _source_json(sources, "aim1-e0-five-seed-contract", paths)
    scientific_change = contract.get("scientific_change", {})
    _require(
        contract.get("status") == "sealed before extension training"
        and contract.get("experiment") == "Aim1 E0 canonical five-seed extension"
        and scientific_change.get("old_model_seeds") == ADOPTED_MODEL_SEEDS
        and scientific_change.get("new_model_seeds") == MODEL_SEEDS
        and scientific_change.get("added_model_seeds") == NEW_MODEL_SEEDS
        and scientific_change.get("split_layout_changed") is False
        and scientific_change.get("model_recipe_changed") is False
        and scientific_change.get("inherited_artifacts_modified") is False,
        "Aim-1 E0 contract does not preserve the frozen three-seed lineage and split layout",
    )
    fit_census = contract.get("fit_census", {})
    _require(
        fit_census
        == {
            "inherited_folds": 15,
            "inherited_p75_refits": 3,
            "new_folds": 10,
            "new_p75_refits": 2,
            "final_folds": 25,
            "final_p75_refits": 5,
            "new_fits": 12,
            "final_fits": 30,
        },
        "Aim-1 E0 fit census is not 18 adopted + 12 new = 30",
    )
    bootstrap = contract.get("analysis_contract", {}).get("bootstrap", {})
    _require(
        bootstrap.get("unit") == "patient"
        and bootstrap.get("shared_indices_across_model_seeds") is True,
        "Aim-1 inference must resample patients with shared draws, not model seeds",
    )

    training = _source_json(sources, "aim1-e0-five-seed-training-validation", paths)
    _require(
        training.get("status") == "complete"
        and training.get("model_seeds") == MODEL_SEEDS
        and training.get("fit_census") == fit_census
        and training.get("fold_layout_shared_across_all_seeds") is True
        and training.get("p75_refit_authenticated_for_all_seeds") is True,
        "Aim-1 five-seed training validation is incomplete or its fold layout changed",
    )
    _assert_source_id(
        training.get("campaign_contract"),
        sources,
        "aim1-e0-five-seed-contract",
        paths,
        context="Aim-1 training contract",
    )

    results = _source_json(sources, "aim1-e0-five-seed-results", paths)
    inference = results.get("inference", {})
    _require(
        results.get("experiment") == "Aim1 E0 canonical five-seed extension"
        and results.get("model_seeds") == MODEL_SEEDS
        and inference.get("unit") == "patient"
        and inference.get("shared_indices_across_model_seeds") is True
        and inference.get("model_seeds_are_inferential_units") is False,
        "Aim-1 result is not the governed five-seed patient-level result",
    )
    for key, source_id in {
        "campaign_contract": "aim1-e0-five-seed-contract",
        "training_validation": "aim1-e0-five-seed-training-validation",
        "patient_native_logits": "aim1-e0-five-seed-patient-logits",
    }.items():
        _assert_source_id(
            results.get("inputs", {}).get(key),
            sources,
            source_id,
            paths,
            context=f"Aim-1 results input {key}",
        )

    analysis = _source_json(sources, "aim1-e0-five-seed-analysis-receipt", paths)
    _require(
        analysis.get("status") == "complete"
        and analysis.get("probability_roundtrip_used") is False,
        "Aim-1 analysis receipt is incomplete or used a probability roundtrip",
    )
    for key, source_id in {
        "results": "aim1-e0-five-seed-results",
        "patient_native_logits": "aim1-e0-five-seed-patient-logits",
        "training_validation": "aim1-e0-five-seed-training-validation",
    }.items():
        _assert_source_id(
            analysis.get(key),
            sources,
            source_id,
            paths,
            context=f"Aim-1 analysis receipt {key}",
        )
    return {"adopted": 18, "new": 12, "complete": 30}


def _validate_aim2_sources(
    paths: BundlePaths, sources: dict[str, dict[str, Any]]
) -> dict[str, int]:
    contract = _source_json(sources, "aim2-loco-five-seed-contract", paths)
    _require(
        contract.get("status") == "sealed_extension_contract"
        and contract.get("experiment") == "Aim 2 complete LOCO MIL five-seed extension"
        and contract.get("seeds")
        == {"adopted": ADOPTED_MODEL_SEEDS, "new": NEW_MODEL_SEEDS, "complete": MODEL_SEEDS},
        "Aim-2 LOCO contract has the wrong status or seed roster",
    )
    split = contract.get("split_verdict", {})
    _require(
        split.get("patient_and_slide_folds_change_across_model_seeds") is False
        and split.get("model_seed_changes_only_stochastic_training") is True,
        "Aim-2 patient/slide folds must be identical for seeds 42-46",
    )
    _require(
        contract.get("statistics", {}).get("model_seeds_are_not_inference_units") is True,
        "Aim-2 contract incorrectly treats model seeds as inference units",
    )
    _validate_aim2_arm_roster(contract)
    scope = contract.get("scope", {})
    _require(
        scope.get("raw_all_conventional_cpht_seed_scope") == ADOPTED_MODEL_SEEDS
        and scope.get("cpht_a_seed_scope") == ADOPTED_MODEL_SEEDS,
        "Aim-2 nine-arm and three-seed CPHT mixed scope changed",
    )
    counts = contract.get("counts", {})
    _require(
        counts
        == {
            "arms": 9,
            "controlling_arms": 8,
            "adopted_fits": 162,
            "new_fits": 108,
            "complete_fits": 270,
            "training_chain_jobs": 36,
            "adopted_score_artifacts": 99,
            "new_score_artifacts": 66,
            "complete_score_artifacts": 165,
        },
        "Aim-2 LOCO census is not nine arms / 270 fits / 165 score artifacts",
    )
    mixed_references = contract.get("mixed_seed_scope_references", {})
    _assert_source_id(
        mixed_references.get("all_conventional_cpht_three_seed"),
        sources,
        "aim2-e2cpht-results",
        paths,
        context="Aim-2 raw CPHT three-seed reference",
    )
    _assert_source_id(
        mixed_references.get("cpht_a_three_seed"),
        sources,
        "aim2-e2cpht-a-v2-results",
        paths,
        context="Aim-2 CPHT-A three-seed reference",
    )

    inference_seal = _source_json(sources, "aim2-loco-five-seed-inference-seal", paths)
    _require(
        inference_seal.get("status") == "sealed_before_outcome_join"
        and inference_seal.get("target_outcomes_present") is False
        and inference_seal.get("adopted_score_count") == 99
        and inference_seal.get("new_score_count") == 66
        and inference_seal.get("complete_score_count") == 165
        and inference_seal.get("five_seed_loco_complete") is True
        and inference_seal.get("all_conventional_cpht_scope") == ADOPTED_MODEL_SEEDS
        and inference_seal.get("cpht_a_scope") == ADOPTED_MODEL_SEEDS,
        "Aim-2 inference seal does not preserve the five-/three-seed boundary",
    )
    _assert_source_id(
        inference_seal.get("contract"),
        sources,
        "aim2-loco-five-seed-contract",
        paths,
        context="Aim-2 inference contract",
    )

    results = _source_json(sources, "aim2-loco-five-seed-results", paths)
    expected_mixed_scope = {
        "five_seed": [
            "all LOCO primary",
            "E2-MET complete LOCO matrix",
            "Orion LOCO sensitivity",
        ],
        "three_seed_unchanged": [
            "raw all-conventional CPHT",
            "CPHT-A residual adaptation",
        ],
        "references": mixed_references,
    }
    _require(
        results.get("experiment") == "Aim 2 complete LOCO MIL five-seed results"
        and results.get("seeds") == MODEL_SEEDS
        and results.get("ensemble_rule") == "mean native logits across seeds, then one sigmoid"
        and results.get("mixed_seed_scope") == expected_mixed_scope
        and set(results.get("primary", {}))
        == {
            "family_cptac",
            "family_rih",
            "family_rih_sm",
            "family_surgen",
            "family_tcga",
            *_SIBLING_ARMS,
        },
        "Aim-2 result is not the complete nine-arm five-seed result with frozen mixed scope",
    )

    report = _source_json(sources, "aim2-loco-five-seed-report-receipt", paths)
    _require(
        report.get("status") == "sealed_five_seed_results"
        and report.get("target_outcomes_opened_only_after_inference_seal") is True
        and report.get("bootstrap_unit") == "patient"
        and report.get("model_seeds_are_not_inference_units") is True,
        "Aim-2 report receipt lacks its label-blind/patient-inference guarantees",
    )
    _assert_source_id(
        report.get("contract"),
        sources,
        "aim2-loco-five-seed-contract",
        paths,
        context="Aim-2 report contract",
    )
    _assert_source_id(
        report.get("inference_seal"),
        sources,
        "aim2-loco-five-seed-inference-seal",
        paths,
        context="Aim-2 report inference seal",
    )
    for key, source_id in {
        "source_oof": "aim2-loco-five-seed-source-oof",
        "calibrators": "aim2-loco-five-seed-calibrators",
        "primary_patients": "aim2-loco-five-seed-primary-patients",
        "met_patients": "aim2-loco-five-seed-metastatic-patients",
        "orion_patients": "aim2-loco-five-seed-orion-patients",
        "results": "aim2-loco-five-seed-results",
        "table": "aim2-loco-five-seed-table",
    }.items():
        _assert_source_id(
            report.get("artifacts", {}).get(key),
            sources,
            source_id,
            paths,
            context=f"Aim-2 report artifact {key}",
        )
    return {"adopted": 162, "new": 108, "complete": 270}


def _aim3_job_census(records: Any, *, adopted: bool) -> tuple[int, dict[str, int], set[int]]:
    _require(isinstance(records, list), "Aim-3 job inventory must be a list")
    component_counts: dict[str, int] = {}
    seeds: set[int] = set()
    component_seed_counts: dict[tuple[str, int], int] = {}
    logical_jobs: dict[tuple[str, str, Any], list[int]] = {}
    observed_jobs: set[tuple[str, str, Any, int]] = set()
    fits = 0
    keys_bound = True
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"Aim-3 job record {index} must be an object")
        job = record.get("job") if adopted else record
        _require(isinstance(job, dict), f"Aim-3 job record {index} lacks job semantics")
        component = job.get("component")
        seed = job.get("model_seed")
        task = job.get("task")
        draw_seed = job.get("draw_seed")
        _require(
            component in {"fixed", "repeated", "e3v", "e1v"}
            and isinstance(seed, int)
            and not isinstance(seed, bool)
            and isinstance(task, str)
            and bool(task),
            f"Aim-3 job record {index} has an invalid component, task, or seed",
        )
        expected_key = _aim3_job_key(component, task, draw_seed, seed)
        expected_fits = 5 if component in {"e3v", "e1v"} else 6
        if adopted:
            keys_bound &= record.get("job_key") == expected_key
        else:
            keys_bound &= (
                record.get("key") == expected_key and record.get("actual_fits") == expected_fits
            )
        component_counts[component] = component_counts.get(component, 0) + 1
        seeds.add(seed)
        component_seed_counts[(component, seed)] = (
            component_seed_counts.get((component, seed), 0) + 1
        )
        logical_jobs.setdefault((component, task, draw_seed), []).append(seed)
        observed_jobs.add((component, task, draw_seed, seed))
        fits += expected_fits
    expected_seeds = ADOPTED_MODEL_SEEDS if adopted else NEW_MODEL_SEEDS
    expected_per_seed = {"fixed": 10, "repeated": 15, "e3v": 6, "e1v": 1}
    _require(
        component_seed_counts
        == {
            (component, seed): count
            for component, count in expected_per_seed.items()
            for seed in expected_seeds
        },
        "Aim-3 component-by-seed distribution changed",
    )
    _require(
        all(sorted(group_seeds) == expected_seeds for group_seeds in logical_jobs.values())
        and {
            component: sum(key[0] == component for key in logical_jobs)
            for component in expected_per_seed
        }
        == expected_per_seed,
        "Aim-3 logical jobs must have one chain for every governed model seed",
    )
    _require(
        observed_jobs == _expected_aim3_jobs(expected_seeds),
        "Aim-3 exact fixed/repeated/E3v/E1v task/draw/seed roster changed",
    )
    _require(keys_bound, "Aim-3 logical job keys and fit counts changed")
    return fits, component_counts, seeds


def _validate_aim3_sources(
    paths: BundlePaths, sources: dict[str, dict[str, Any]]
) -> dict[str, int]:
    contract = _source_json(sources, "aim3-ladders-five-seed-contract", paths)
    protocol = contract.get("protocol", {})
    _require(
        contract.get("status") == "prepared"
        and protocol.get("old_seeds") == ADOPTED_MODEL_SEEDS
        and protocol.get("new_seeds") == NEW_MODEL_SEEDS
        and protocol.get("all_seeds") == MODEL_SEEDS
        and protocol.get("max_parallel_chains") == 6,
        "Aim-3 ladder contract has the wrong seed roster or concurrency",
    )
    _require(
        contract.get("new_counts")
        == {"chains": 64, "oof_folds": 320, "refits": 50, "actual_mil_fits": 370},
        "Aim-3 new-job census is not 64 chains / 370 fits",
    )
    adopted_fits, adopted_components, adopted_seeds = _aim3_job_census(
        contract.get("adopted_old_chains"), adopted=True
    )
    new_fits, new_components, new_seeds = _aim3_job_census(contract.get("new_jobs"), adopted=False)
    _require(
        adopted_fits == 555
        and adopted_components == {"fixed": 30, "repeated": 45, "e3v": 18, "e1v": 3}
        and adopted_seeds == set(ADOPTED_MODEL_SEEDS),
        "Aim-3 adopted inventory is not the governed 555-fit three-seed lineage",
    )
    _require(
        new_fits == 370
        and new_components == {"fixed": 20, "repeated": 30, "e3v": 12, "e1v": 2}
        and new_seeds == set(NEW_MODEL_SEEDS),
        "Aim-3 extension inventory is not the governed 370-fit two-seed addition",
    )

    results = _source_json(sources, "aim3-ladders-five-seed-results", paths)
    _require(
        results.get("status") == "complete"
        and results.get("estimand")
        == "five-seed patient native-logit ensemble; seeds are not inference units"
        and {
            "three_seed_replay_before_extension",
            "e0_gene_reference",
            "fixed_univ1",
            "repeated_univ1",
            "e3v_virchow2_cls",
            "e1v_virchow2_cls_gene_reference",
        }.issubset(results),
        "Aim-3 result does not cover fixed, repeated, E3v, and E1v five-seed analyses",
    )

    audit = _source_json(sources, "aim3-ladders-five-seed-analysis-audit", paths)
    _require(audit.get("status") == "PASS", "Aim-3 analysis audit did not pass")
    _assert_source_id(
        audit.get("contract"),
        sources,
        "aim3-ladders-five-seed-contract",
        paths,
        context="Aim-3 analysis contract",
    )
    _assert_source_id(
        audit.get("report"),
        sources,
        "aim3-ladders-five-seed-results",
        paths,
        context="Aim-3 analysis report",
    )
    _assert_source_id(
        audit.get("bootstrap_distributions"),
        sources,
        "aim3-ladders-five-seed-bootstrap",
        paths,
        context="Aim-3 bootstrap archive",
    )

    completion = _source_json(sources, "aim3-ladders-five-seed-completion", paths)
    _require(
        completion.get("status") == "completed"
        and completion.get("counts")
        == {"chains": 64, "oof_folds": 320, "refits": 50, "actual_mil_fits": 370},
        "Aim-3 extension completion census changed",
    )
    for key, source_id in {
        "five_seed_results.json": "aim3-ladders-five-seed-results",
        "bootstrap_distributions.npz": "aim3-ladders-five-seed-bootstrap",
        "analysis_audit.json": "aim3-ladders-five-seed-analysis-audit",
    }.items():
        _assert_source_id(
            completion.get("artifacts", {}).get(key),
            sources,
            source_id,
            paths,
            context=f"Aim-3 completion artifact {key}",
        )
    return {"adopted": 555, "new": 370, "complete": 925}


def _validate_auroc_block(block: Any, *, context: str) -> tuple[float, list[float]]:
    _require(isinstance(block, dict), f"{context} must be an object")
    point = block.get("auroc")
    interval = block.get("auroc_ci95")
    _require(
        isinstance(point, (int, float))
        and not isinstance(point, bool)
        and isinstance(interval, list)
        and len(interval) == 2
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in interval
        ),
        f"{context} lacks a numeric AUROC point and interval",
    )
    numeric = [float(interval[0]), float(point), float(interval[1])]
    _require(
        all(math.isfinite(value) for value in numeric)
        and 0.0 <= numeric[0] <= numeric[1] <= numeric[2] <= 1.0,
        f"{context} AUROC interval must be finite, ordered, in [0,1], and contain its point",
    )
    return float(point), [float(interval[0]), float(interval[1])]


def _validate_directional_gate(block: Any, *, context: str) -> bool:
    _point, interval = _validate_auroc_block(block, context=context)
    passed = bool(float(interval[0]) > 0.50)
    gate = block.get("directional_gate")
    _require(
        isinstance(gate, dict)
        and gate.get("rule") == "patient-bootstrap AUROC lower 95% bound > 0.50"
        and gate.get("lower_ci_above_0p5") is passed
        and gate.get("passes") is passed,
        f"{context} gate must be determined only by its lower 95% AUROC bound",
    )
    return passed


def _validate_adjudication_sources(paths: BundlePaths, sources: dict[str, dict[str, Any]]) -> None:
    result = _source_json(sources, "aim2-e2a-five-seed-adjudication-result", paths)
    _require(
        result.get("status") == "governed_five_seed_adjudication"
        and result.get("experiment") == "Aim 2 E2a five-seed governed adjudication"
        and result.get("model_seeds") == MODEL_SEEDS
        and result.get("ensemble_rule") == "mean native logits across five model seeds",
        "Aim-2 adjudication has the wrong status or model-seed scope",
    )
    inference = result.get("inference", {})
    _require(
        inference.get("unit") == "patient"
        and inference.get("stratification") == "target_x_KRAS"
        and inference.get("n_bootstrap") == 10_000
        and inference.get("bootstrap_seed") == 20260817
        and inference.get("paired_scorer_indices_shared") is True
        and inference.get("macro_recomputed_each_draw") is True
        and inference.get("folds_and_model_seeds_are_not_resampling_units") is True,
        "Aim-2 adjudication inference topology changed",
    )

    precedence = result.get("precedence", {})
    _require(
        precedence.get("superseded_source_report_json_pointers")
        == list(SUPERSEDED_ADJUDICATION_POINTERS)
        and precedence.get("non_authoritative_source_paired_ci_json_pointers")
        == list(NONAUTHORITATIVE_PAIRED_CI_POINTERS)
        and precedence.get("retained_source_paired_delta_point_json_pointers")
        == list(RETAINED_PAIRED_POINT_POINTERS)
        and precedence.get("authoritative_replacements")
        == {
            "E2a-F gates and claim": "/e2a_f/adjudication",
            "E2a-D gates and claim": "/e2a_d/adjudication",
            "sibling paired intervals": "/e2a_d/paired_sibling_minus_family",
            "RIH size-matched paired interval": "/e2a_d/size_matched_rih_sensitivity",
        }
        and precedence.get("superseded_sibling_primary_ci_replacements")
        == {
            f"/primary/{arm}/auroc_ci95": f"/e2a_d/targets/{slug}/auroc_ci95"
            for arm, slug in zip(_SIBLING_ARMS, _SIBLING_SLUGS, strict=True)
        }
        and precedence.get("confirmed_unchanged_source_report_json_pointers")
        == [
            "/family_loco_standardized_macro/directional_results",
            "/family_loco_standardized_macro/macro_auroc",
            "/family_loco_standardized_macro/macro_auroc_ci95",
            "/e2met_confirmatory",
        ]
        and precedence.get("supersession_is_field_scoped") is True
        and precedence.get("all_unlisted_source_report_fields_remain_authoritative") is True
        and precedence.get("macro_rescue_prohibited") is True,
        "Aim-2 adjudication pointer inventory or precedence boundary changed",
    )
    _assert_source_id(
        precedence.get("source_report"),
        sources,
        "aim2-loco-five-seed-results",
        paths,
        context="Aim-2 adjudication source report",
    )

    e2a_f = result.get("e2a_f", {})
    family_names = (
        "CPTAC",
        "RIH",
        "SR386_given_whole_SurGen_holdout",
        "SR1482_given_whole_SurGen_holdout",
        "TCGA_pooled_COAD_READ",
    )
    directions = e2a_f.get("directions", {})
    _require(
        isinstance(directions, dict) and set(directions) == set(family_names),
        "E2a-F adjudication must contain the exact five directional targets",
    )
    passed_family = [
        name
        for name in family_names
        if _validate_directional_gate(directions[name], context=f"E2a-F/{name}")
    ]
    f_adjudication = e2a_f.get("adjudication", {})
    family_macro = e2a_f.get("nested_four_family_macro", {})
    _validate_auroc_block(family_macro, context="E2a-F nested macro")
    _require(
        f_adjudication.get("passed_directions") == passed_family
        and f_adjudication.get("required_directions") == list(family_names)
        and f_adjudication.get("all_directional_lower_bounds_above_0p5")
        is (len(passed_family) == 5)
        and f_adjudication.get("claim_family_loco_transport") is (len(passed_family) == 5)
        and f_adjudication.get("rule")
        == (
            "all five target-specific patient-bootstrap AUROC lower 95% bounds "
            "must exceed 0.50; the macro cannot rescue a failed direction"
        )
        and family_macro.get("role") == "panel summary; cannot rescue a failed direction",
        "E2a-F claim must be the intersection of all five lower-CI gates",
    )

    e2a_d = result.get("e2a_d", {})
    targets = e2a_d.get("targets", {})
    _require(
        isinstance(targets, dict) and set(targets) == set(_SIBLING_SLUGS),
        "E2a-D adjudication must contain the exact four sibling targets",
    )
    passed_siblings = [
        name
        for name in _SIBLING_SLUGS
        if _validate_directional_gate(targets[name], context=f"E2a-D/{name}")
    ]
    d_adjudication = e2a_d.get("adjudication", {})
    macros = e2a_d.get("macros", {})
    _require(
        isinstance(macros, dict)
        and set(macros)
        == {
            "secondary_five_acquisition_domain",
            "descriptive_equal_six_stratum",
        },
        "E2a-D macro inventory changed",
    )
    secondary_macro = macros["secondary_five_acquisition_domain"]
    descriptive_macro = macros["descriptive_equal_six_stratum"]
    _validate_auroc_block(secondary_macro, context="E2a-D five-acquisition macro")
    _validate_auroc_block(descriptive_macro, context="E2a-D six-stratum macro")
    _require(
        d_adjudication.get("passed_directions") == passed_siblings
        and d_adjudication.get("required_directions") == list(_SIBLING_SLUGS)
        and d_adjudication.get("all_directional_lower_bounds_above_0p5")
        is (len(passed_siblings) == 4)
        and d_adjudication.get("claim_sibling_stratum_transport") is (len(passed_siblings) == 4)
        and d_adjudication.get("rule")
        == (
            "all four sibling-stratum patient-bootstrap AUROC lower 95% bounds "
            "must exceed 0.50; neither macro can rescue a failed direction"
        )
        and secondary_macro.get("role")
        == "secondary panel summary; cannot rescue a failed direction"
        and descriptive_macro.get("role") == "descriptive; gives TCGA and SurGen two votes each",
        "E2a-D claim must be the intersection of all four lower-CI gates without macro rescue",
    )

    scope = result.get("scope_boundary", {})
    _require(
        scope.get("raw_cpht", {}).get("model_seeds") == ADOPTED_MODEL_SEEDS
        and scope.get("raw_cpht", {}).get("five_seed_adjudication_applied") is False
        and scope.get("cpht_a", {}).get("model_seeds") == ADOPTED_MODEL_SEEDS
        and scope.get("orion_loco_sensitivity", {}).get("is_confirmatory_e2_cpht") is False
        and scope.get("five_seed_e2met_role_and_organ_analyses", {}).get(
            "required_to_confirm_existing_e2met_gate"
        )
        is False,
        "Aim-2 adjudication mixed-scope boundary changed",
    )
    _assert_source_id(
        scope.get("raw_cpht", {}).get("result"),
        sources,
        "aim2-e2cpht-results",
        paths,
        context="Aim-2 adjudication raw CPHT reference",
    )

    receipt = _source_json(sources, "aim2-e2a-five-seed-adjudication-receipt", paths)
    _require(
        receipt.get("status") == "sealed_governed_five_seed_adjudication"
        and receipt.get("experiment") == "Aim 2 E2a five-seed governed adjudication"
        and receipt.get("bootstrap_inventory", {}).get("draws_per_array") == 10_000
        and receipt.get("bootstrap_inventory", {}).get("dtype") == "float64"
        and receipt.get("verification_contract")
        == {
            "deterministic_full_recomputation": True,
            "verify_is_read_only": True,
            "receipt_written_last": True,
            "source_campaign_never_written": True,
        },
        "Aim-2 adjudication receipt or deterministic replay contract changed",
    )
    for key, source_id in {
        "result": "aim2-e2a-five-seed-adjudication-result",
        "bootstrap_distributions": "aim2-e2a-five-seed-adjudication-bootstrap",
    }.items():
        _assert_source_id(
            receipt.get("outputs", {}).get(key),
            sources,
            source_id,
            paths,
            context=f"Aim-2 adjudication output {key}",
        )
    for key, source_id in {
        "tool": "aim2-e2a-five-seed-adjudication-implementation",
        "focused_test": "aim2-e2a-five-seed-adjudication-test",
    }.items():
        _assert_source_id(
            receipt.get("implementation", {}).get(key),
            sources,
            source_id,
            paths,
            context=f"Aim-2 adjudication implementation {key}",
        )


def _verification_identities(paths: BundlePaths) -> dict[str, dict[str, Any]]:
    return {
        "verifier": identity(
            paths.verifier_code,
            display_path=_relative_display(paths.verifier_code, paths.repo),
        ),
        "tests": identity(
            paths.verifier_test,
            display_path=_relative_display(paths.verifier_test, paths.repo),
        ),
    }


def build_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Validate the candidate bundle and return receipt content."""
    selected = default_paths() if paths is None else paths
    documents = _validate_report_contract(selected)
    source_manifest, authoritative_sources = _validate_source_manifest(selected)
    return {
        "schema_version": 1,
        "bundle": "reports/final_v9",
        "status": SEALED_STATUS,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "organization": "aim_focused_clean_rewrite",
        "post_outcome_official_specification": True,
        "not_preregistration": True,
        "preregistration": False,
        "append_only_inheritance_required": False,
        "execution_policy": {
            "max_concurrent_gpu_trainers": 6,
            "unit": "independent_training_subprocesses",
        },
        "declared_scientific_states": {
            "canonical_aim1_orion_included": False,
            "cpht_r": "NOT_RUN",
            "aim4_whole_section_pathology_validation": "GENERATED_UNREAD",
        },
        "model_seed_scope": {
            "adopted": ADOPTED_MODEL_SEEDS,
            "new": NEW_MODEL_SEEDS,
            "complete": MODEL_SEEDS,
            "patient_and_slide_fold_membership_changes_across_model_seeds": False,
            "model_seeds_are_inference_units": False,
        },
        "study_wide_mil_census": {
            "adopted_fits": 735,
            "new_fits": 490,
            "complete_fits": 1225,
            "new_training_chain_jobs": 102,
            "maximum_concurrent_gpu_trainers": 6,
            "observed_peak_concurrent_gpu_trainers": 6,
        },
        "mixed_seed_scope": {
            "five_seed": [
                "Aim 1 E0",
                "Aim 1 E1v",
                "Aim 2 all nine LOCO primary arms",
                "Aim 2 E2-MET LOCO sensitivity",
                "Aim 2 Orion LOCO sensitivity",
                "Aim 3 fixed controls",
                "Aim 3 repeated controls",
                "Aim 3 E3v",
            ],
            "three_seed_unchanged": [
                "raw E2-CPHT",
                "E2-CPHT-A",
                "15-fold Orion sensitivity",
                "E2e",
                "E2f-v3",
                "between-slide analysis",
                "detailed E2-MET role/organ analyses",
                "unaffected Aim 1 analyses",
            ],
        },
        "documents": documents,
        "source_manifest": source_manifest,
        "authoritative_sources": authoritative_sources,
        "verification": _verification_identities(selected),
        "checks": {
            "aim_headings_ordered_and_unique": "PASS",
            "official_e2_cpht_name": "PASS",
            "six_parallel_trainers_default": "PASS",
            "canonical_aim1_orion_exclusion": "PASS",
            "declared_incomplete_components": "PASS",
            "placeholder_scan": "PASS",
            "source_identity_validation": "PASS",
            "required_five_seed_source_inventory": "PASS",
            "five_seed_semantic_cross_links": "PASS",
            "mixed_three_five_seed_scope": "PASS",
            "adjudication_pointer_and_lower_ci_gates": "PASS",
            "study_wide_fit_and_concurrency_census": "PASS",
        },
    }


def _stable_receipt_fields(receipt: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "created_utc"}


def verify_published_receipt(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Verify the published receipt and all files it seals."""
    selected = default_paths() if paths is None else paths
    published = _load_json(selected.destination, label="final-v9 bundle receipt")
    created = published.get("created_utc")
    if not isinstance(created, str):
        raise BundleVerificationError("published receipt has no created_utc timestamp")
    try:
        parsed = dt.datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BundleVerificationError("published receipt created_utc is invalid") from exc
    if parsed.tzinfo is None:
        raise BundleVerificationError("published receipt created_utc must be timezone-aware")

    current = build_receipt(selected)
    if _stable_receipt_fields(published) != _stable_receipt_fields(current):
        raise BundleVerificationError(
            "published final-v9 receipt does not match the current documents, sources, or verifier"
        )
    return published


def _receipt_bytes(receipt: dict[str, Any]) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def seal(paths: BundlePaths | None = None) -> dict[str, Any]:
    """Validate and atomically publish a final-v9 receipt exactly once."""
    selected = default_paths() if paths is None else paths
    if selected.destination.exists():
        raise BundleVerificationError(
            f"refusing to overwrite existing final-v9 receipt: {selected.destination}"
        )
    selected.destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = build_receipt(selected)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{selected.destination.name}.",
        suffix=".tmp",
        dir=selected.destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(_receipt_bytes(receipt))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, selected.destination)
        except FileExistsError as exc:
            raise BundleVerificationError(
                f"refusing to overwrite existing final-v9 receipt: {selected.destination}"
            ) from exc
        directory_fd = os.open(selected.destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return verify_published_receipt(selected)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seal",
        action="store_true",
        help="validate and publish report_bundle_receipt.json exactly once",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = seal() if args.seal else verify_published_receipt()
    except BundleVerificationError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        f"PASS: {receipt['status']} ({len(receipt['authoritative_sources'])} authoritative sources)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
