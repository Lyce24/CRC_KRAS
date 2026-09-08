#!/usr/bin/env python3
"""Additive FINAL-v11 continuation in the isolated ``downstream_v2`` namespace.

The first downstream prepare completed before the training recovery acquired a
scope-aware terminal.  Its seven label-blind artifacts and the controller that
created them are immutable evidence.  This continuation therefore never edits
``downstream/``, ``recovery_v1/``, ``recovery_v2/``, the legacy controller, or
the legacy controller test.  It authenticates those inputs, adopts the scoped
recovery-v2 terminal, and places every new contract, score, seal, receipt, and
analysis artifact below ``downstream_v2/``.

The statistical implementation is delegated to the frozen legacy controller.
Delegation is explicit and receipted: a scoped runtime binding redirects every
mutable path, every internal scoring subprocess, the training gate, and the
implementation graph.  The five outcome-bearing analysis artifacts live at
``downstream_v2/analysis`` and cannot be produced before the v2 inference seal.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import aim1_tcga_surgen_full_pipeline_analysis as legacy  # noqa: E402

SCHEMA_VERSION = 1
EXPERIMENT = "final_v11_tcga_surgen_two_encoder_full_pipeline_downstream_v2"
DEFAULT_CAMPAIGN_ROOT = legacy.DEFAULT_CAMPAIGN_ROOT
NAMESPACE = "downstream_v2"
LEGACY_NAMESPACE = "downstream"
RECOVERY_V2_TERMINAL = Path("recovery_v2/receipts/training_complete_scoped.json")
RECOVERY_V2_STATUS = "complete_and_certified_via_scoped_census_erratum"
RECOVERY_V2_TERMINAL_SHA256 = "fa41798f4266ebcb6bfde7ccae7ad4990e8de965c2cdacca4be1276989b43d08"
RECOVERY_V2_TERMINAL_SIZE = 6_267
RECOVERY_V2_TERMINAL_FIELDS = {
    "schema_version",
    "recovery",
    "status",
    "created_utc",
    "base_campaign",
    "scope_contract",
    "scope_adjudication",
    "recovery_implementation",
    "predecessor_v1_terminal",
    "training_scoped_census",
    "prepared_downstream_baseline",
    "namespace_policy",
    "fit_accounting",
    "execution_accounting",
    "concurrency",
    "certification_boundary",
}
RECOVERY_V2_CONTROLLER = REPO / "tools/aim1_tcga_surgen_two_encoder_recovery_v2.py"
RECOVERY_V2_TEST = REPO / "tests/test_aim1_tcga_surgen_two_encoder_recovery_v2.py"
RECOVERY_V2_CONTROLLER_SHA256 = "e474781f955a932cef575e6f8452f7eb824e178e1fbbe9254432ef7008b4687d"
RECOVERY_V2_CONTROLLER_SIZE = 42_182
RECOVERY_V2_TEST_SHA256 = "2dc7b5362edbb55bfdd84d53cb3ff61ce5b30b0fc12c0a1a1fa5b9f763d52891"
RECOVERY_V2_TEST_SIZE = 18_916
ANALYSIS_V2_TEST = REPO / "tests/test_aim1_tcga_surgen_full_pipeline_analysis_v2.py"
LEGACY_RECOVERY_TERMINAL = Path(legacy.TRAINING_RECOVERY_TERMINAL)

LEGACY_ANALYSIS_CONTROLLER_SHA256 = (
    "9f86156da96a8fb2620d45a692c222f46cceef9fa546afd8dd4ca35e09178292"
)
LEGACY_ANALYSIS_CONTROLLER_SIZE = 185_328
LEGACY_ANALYSIS_TEST_SHA256 = "ac971cca54cac60bda914e7729bb817f47e9513f74e6a5cce55ee4db5dfd0219"
LEGACY_ANALYSIS_TEST_SIZE = 45_668
LEGACY_RECOVERY_CONTROLLER_SHA256 = legacy.TRAINING_RECOVERY_CONTROLLER_SHA256
LEGACY_RECOVERY_CONTROLLER_SIZE = legacy.TRAINING_RECOVERY_CONTROLLER_SIZE
LEGACY_RECOVERY_TEST_SHA256 = legacy.TRAINING_RECOVERY_TEST_SHA256
LEGACY_RECOVERY_TEST_SIZE = legacy.TRAINING_RECOVERY_TEST_SIZE
LEGACY_RECOVERY_TERMINAL_IDENTITY = {
    "path": str((DEFAULT_CAMPAIGN_ROOT / LEGACY_RECOVERY_TERMINAL).resolve()),
    "sha256": "4c3a2f4626b8a66e37b37afd20b211bb2de8fa32901db49a86d123a572b13c86",
    "size_bytes": 23_089,
}
LEGACY_PREPARED_CONTRACT_IDENTITY = {
    "path": str((DEFAULT_CAMPAIGN_ROOT / "downstream/contract.json").resolve()),
    "sha256": "a3a7329c7ae57c6d3c5b17cd275d196539559a01dab1e307f3cfe65dfd5361a9",
    "size_bytes": 27_484,
}
LEGACY_PREPARED_JOB_PLAN_IDENTITY = {
    "path": str((DEFAULT_CAMPAIGN_ROOT / "downstream/jobs/score_jobs.json").resolve()),
    "sha256": "2385c72511069f6c1db7da2cd08a3b9b941500fc0cb2aa0232d922019d8bc7d9",
    "size_bytes": 35_839,
}
LEGACY_PREPARED_FILES: tuple[tuple[str, str, int], ...] = (
    (
        "downstream/contract.json",
        "a3a7329c7ae57c6d3c5b17cd275d196539559a01dab1e307f3cfe65dfd5361a9",
        27_484,
    ),
    (
        "downstream/inputs/label_blind/cptac_primary.csv",
        "135a454edbd94d2c27889946187faedf68312be0cbb6f93786987beae970eb5b",
        8_203,
    ),
    (
        "downstream/inputs/label_blind/orion_cpht.csv",
        "6acb9699c53944c1fa87b90a42e773eed830ef71594589de5d71005a1a31d024",
        3_059,
    ),
    (
        "downstream/inputs/label_blind/rih_metastatic.csv",
        "dec443d99e437e7cb3fc3297a4f98302081ba4455386e3b7d930ff142feb1e5c",
        5_177,
    ),
    (
        "downstream/inputs/label_blind/rih_primary.csv",
        "0c33a3562da87a9843a41a34e6693e37825538aa5be0c31971798943f010a06e",
        9_268,
    ),
    (
        "downstream/inputs/label_blind/sr1482_metastatic.csv",
        "90eb00ef0baf6a33fc4b423affbce33dde1d38068add59cc136f870f13cf6d15",
        6_772,
    ),
    (
        "downstream/jobs/score_jobs.json",
        "2385c72511069f6c1db7da2cd08a3b9b941500fc0cb2aa0232d922019d8bc7d9",
        35_839,
    ),
)
LEGACY_PREPARED_ARTIFACT_COUNT = 7
LEGACY_PREPARED_TOTAL_SIZE = 95_802
LEGACY_PREPARED_TREE_SHA256 = "d6400d6004ad2c1a850c022db1cb694f08b0fac445cdbbc3d94400f1fc995621"

GovernanceError = legacy.GovernanceError
SEEDS = legacy.SEEDS
ENCODERS = legacy.ENCODERS
TARGET_ORDER = legacy.TARGET_ORDER
TARGETS = legacy.TARGETS
DEFAULT_MAX_WORKERS = legacy.DEFAULT_MAX_WORKERS
N_BOOTSTRAP = legacy.N_BOOTSTRAP
ANALYSIS_FILES = legacy.ANALYSIS_FILES
FINAL_SOURCE_GRAPH_KEYS = (
    "downstream_v2_controller",
    "downstream_v2_controller_test",
    "downstream_v2_contract",
    "downstream_v2_score_job_plan",
    "downstream_v2_deep_preflight",
    "downstream_v2_scoring_completion",
    "downstream_v2_inference_environment",
    "downstream_v2_inference_seal",
    "downstream_v2_analysis_contract",
    "downstream_v2_patient_native_logits",
    "downstream_v2_bootstrap_distributions",
    "downstream_v2_results",
    "downstream_v2_analysis_completion",
)


def continuation_root(campaign_root: Path) -> Path:
    return legacy._safe_campaign_root(campaign_root) / NAMESPACE


def legacy_prepared_root(campaign_root: Path) -> Path:
    return legacy._safe_campaign_root(campaign_root) / LEGACY_NAMESPACE


def legacy_blind_path(campaign_root: Path, target: str) -> Path:
    if target not in TARGET_ORDER:
        raise GovernanceError(f"Unknown target: {target}")
    return legacy_prepared_root(campaign_root) / f"inputs/label_blind/{target}.csv"


def continuation_analysis_root(campaign_root: Path) -> Path:
    return continuation_root(campaign_root) / "analysis"


def _artifact_record(path: Path, sha256: str, size_bytes: int) -> dict[str, Any]:
    return {"path": str(path.resolve(strict=False)), "sha256": sha256, "size_bytes": size_bytes}


def _canonical_record_tree(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(records), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _expected_legacy_records(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": relative, "sha256": sha256, "size_bytes": size_bytes}
        for relative, sha256, size_bytes in LEGACY_PREPARED_FILES
    ]


def _assert_root_analysis_absent(root: Path) -> None:
    forbidden = root / "analysis"
    if forbidden.exists() or forbidden.is_symlink():
        raise GovernanceError("Legacy root/analysis is forbidden; v2 analysis must stay isolated")


def _legacy_tool_identities() -> dict[str, dict[str, Any]]:
    return {
        "analysis_controller": _artifact_record(
            Path(legacy.__file__),
            LEGACY_ANALYSIS_CONTROLLER_SHA256,
            LEGACY_ANALYSIS_CONTROLLER_SIZE,
        ),
        "analysis_test": _artifact_record(
            legacy.ANALYSIS_TEST,
            LEGACY_ANALYSIS_TEST_SHA256,
            LEGACY_ANALYSIS_TEST_SIZE,
        ),
        "recovery_controller": _artifact_record(
            legacy.TRAINING_RECOVERY_CONTROLLER,
            LEGACY_RECOVERY_CONTROLLER_SHA256,
            LEGACY_RECOVERY_CONTROLLER_SIZE,
        ),
        "recovery_test": _artifact_record(
            legacy.TRAINING_RECOVERY_TEST,
            LEGACY_RECOVERY_TEST_SHA256,
            LEGACY_RECOVERY_TEST_SIZE,
        ),
    }


def validate_legacy_prepared_bundle(campaign_root: Path) -> dict[str, Any]:
    """Authenticate the exact seven-file label-blind predecessor bundle."""
    root = legacy._safe_campaign_root(campaign_root)
    _assert_root_analysis_absent(root)
    directory = legacy_prepared_root(root)
    if not directory.is_dir() or directory.is_symlink():
        raise GovernanceError("Legacy prepared downstream namespace is missing or symlinked")
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise GovernanceError(f"Legacy prepared bundle contains a symlink: {path}")
    expected_records = _expected_legacy_records(root)
    expected_paths = {root / record["path"] for record in expected_records}
    observed_paths = {path for path in directory.rglob("*") if path.is_file()}
    if observed_paths != expected_paths:
        raise GovernanceError(
            "Legacy prepared seven-file inventory drifted: "
            f"missing={sorted(str(path) for path in expected_paths - observed_paths)}, "
            f"unexpected={sorted(str(path) for path in observed_paths - expected_paths)}"
        )
    if (
        len(expected_records) != LEGACY_PREPARED_ARTIFACT_COUNT
        or sum(int(record["size_bytes"]) for record in expected_records)
        != LEGACY_PREPARED_TOTAL_SIZE
        or _canonical_record_tree(expected_records) != LEGACY_PREPARED_TREE_SHA256
    ):
        raise GovernanceError("Compiled legacy prepared-bundle census constants drifted")
    artifacts: dict[str, dict[str, Any]] = {}
    for record in expected_records:
        path = root / str(record["path"])
        observed = legacy._artifact(path)
        wanted = _artifact_record(path, str(record["sha256"]), int(record["size_bytes"]))
        if observed != wanted:
            raise GovernanceError(f"Legacy prepared artifact drifted: {path}")
        artifacts[str(record["path"])] = observed

    contract_path = root / "downstream/contract.json"
    contract = legacy._read_json(contract_path)
    if legacy._artifact(contract_path) != _artifact_record(
        contract_path,
        str(LEGACY_PREPARED_CONTRACT_IDENTITY["sha256"]),
        int(LEGACY_PREPARED_CONTRACT_IDENTITY["size_bytes"]),
    ):
        raise GovernanceError("Legacy prepared contract identity drifted")
    if (
        contract.get("status") != "prepared_label_blind_before_outcome_join"
        or (contract.get("score_contract") or {}).get("jobs") != 50
        or (contract.get("score_contract") or {}).get("slide_rows") != 4_790
        or (contract.get("score_contract") or {}).get("target_outcomes_present") is not False
        or (contract.get("governance") or {}).get("prepare_reads_target_outcomes") is not False
        or (contract.get("governance") or {}).get("preflight_reads_target_outcomes") is not False
        or (contract.get("training") or {}).get("terminal")
        != _artifact_record(
            root / LEGACY_RECOVERY_TERMINAL,
            str(LEGACY_RECOVERY_TERMINAL_IDENTITY["sha256"]),
            int(LEGACY_RECOVERY_TERMINAL_IDENTITY["size_bytes"]),
        )
    ):
        raise GovernanceError("Legacy prepared contract semantics drifted")
    tool_identities = _legacy_tool_identities()
    for identity in tool_identities.values():
        legacy._validate_identity(identity)
    if not all(
        identity in contract.get("implementation", []) for identity in tool_identities.values()
    ):
        raise GovernanceError("Legacy prepared contract implementation graph drifted")

    snapshots: dict[str, dict[str, Any]] = {}
    snapshot_frames: dict[str, pd.DataFrame] = {}
    for target in TARGET_ORDER:
        path = legacy_blind_path(root, target)
        identity = legacy._artifact(path)
        if (contract.get("label_blind_snapshots") or {}).get(target) != identity:
            raise GovernanceError(f"Legacy {target} snapshot identity drifted from contract")
        frame = legacy._blind_frame(
            pd.read_csv(path, low_memory=False),
            spec=TARGETS[target],
            context=f"legacy_prepared/{target}",
        )
        snapshots[target] = identity
        snapshot_frames[target] = frame

    jobs_path = root / "downstream/jobs/score_jobs.json"
    jobs_document = legacy._read_json(jobs_path)
    jobs = jobs_document.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 50:
        raise GovernanceError("Legacy prepared score-job roster drifted")
    expected_factors = {
        (encoder, target, seed) for encoder in ENCODERS for target in TARGET_ORDER for seed in SEEDS
    }
    observed_factors = {
        (str(job.get("encoder")), str(job.get("target")), int(job.get("seed", -1)))
        for job in jobs
        if isinstance(job, dict)
    }
    if observed_factors != expected_factors or any(
        not isinstance(job, dict)
        or job.get("fit_count") != 0
        or job.get("contains_target_outcomes") is not False
        or not str(job.get("manifest", "")).startswith(str(directory / "inputs/label_blind"))
        or not str(job.get("output", "")).startswith(str(directory / "scores"))
        for job in jobs
    ):
        raise GovernanceError("Legacy prepared score-job semantics drifted")
    return {
        "namespace": LEGACY_NAMESPACE,
        "status": "authenticated_immutable_label_blind_predecessor",
        "artifact_count": LEGACY_PREPARED_ARTIFACT_COUNT,
        "total_size_bytes": LEGACY_PREPARED_TOTAL_SIZE,
        "tree_sha256": LEGACY_PREPARED_TREE_SHA256,
        "artifacts": artifacts,
        "contract": legacy._artifact(contract_path),
        "score_jobs": legacy._artifact(jobs_path),
        "label_blind_snapshots": snapshots,
        "legacy_implementation": tool_identities,
        "legacy_training": contract["training"],
        "snapshot_frames": snapshot_frames,
    }


def _recovery_v2_module() -> Any:
    try:
        return importlib.import_module("tools.aim1_tcga_surgen_two_encoder_recovery_v2")
    except ImportError as exc:  # pragma: no cover - production dependency gate
        raise GovernanceError("Frozen recovery-v2 controller is unavailable") from exc


def _recovery_v2_source_identities() -> dict[str, dict[str, Any]]:
    if (
        len(RECOVERY_V2_CONTROLLER_SHA256) != 64
        or RECOVERY_V2_CONTROLLER_SIZE <= 0
        or len(RECOVERY_V2_TEST_SHA256) != 64
        or RECOVERY_V2_TEST_SIZE <= 0
    ):
        raise GovernanceError("Recovery-v2 source pins are not frozen")
    expected = {
        "controller": _artifact_record(
            RECOVERY_V2_CONTROLLER,
            RECOVERY_V2_CONTROLLER_SHA256,
            RECOVERY_V2_CONTROLLER_SIZE,
        ),
        "controller_test": _artifact_record(
            RECOVERY_V2_TEST,
            RECOVERY_V2_TEST_SHA256,
            RECOVERY_V2_TEST_SIZE,
        ),
    }
    for name, identity in expected.items():
        if legacy._artifact(Path(str(identity["path"]))) != identity:
            raise GovernanceError(f"Frozen recovery-v2 {name} bytes drifted")
    return expected


def _validate_training_bundle_v2(campaign_root: Path, *, deep: bool = True) -> dict[str, Any]:
    """Validate and normalize the exact scoped recovery-v2 terminal."""
    root = legacy._safe_campaign_root(campaign_root)
    sources = _recovery_v2_source_identities()
    module = _recovery_v2_module()
    if (
        Path(str(getattr(module, "__file__", ""))).resolve(strict=False)
        != RECOVERY_V2_CONTROLLER.resolve(strict=False)
        or getattr(module, "RECOVERY_V2_TERMINAL", None) != RECOVERY_V2_TERMINAL
        or getattr(module, "RECOVERY_STATUS", None) != RECOVERY_V2_STATUS
        or getattr(module, "TERMINAL_FIELDS", None) != RECOVERY_V2_TERMINAL_FIELDS
        or not callable(getattr(module, "validate_scoped_terminal", None))
    ):
        raise GovernanceError("Recovery-v2 module API/constants drifted")
    try:
        terminal = module.validate_scoped_terminal(root, deep_scope=deep)
    except Exception as exc:
        contract_error = getattr(module, "ContractError", Exception)
        if isinstance(exc, contract_error):
            raise GovernanceError("Scoped training-recovery-v2 replay failed") from exc
        raise
    terminal_path = root / RECOVERY_V2_TERMINAL
    expected_terminal_identity = _artifact_record(
        terminal_path,
        RECOVERY_V2_TERMINAL_SHA256,
        RECOVERY_V2_TERMINAL_SIZE,
    )
    if legacy._artifact(terminal_path) != expected_terminal_identity:
        raise GovernanceError("Frozen recovery-v2 terminal bytes drifted")
    if legacy._read_json(terminal_path) != terminal:
        raise GovernanceError("Recovery-v2 terminal differs from public replay")
    if set(terminal) != RECOVERY_V2_TERMINAL_FIELDS:
        raise GovernanceError("Recovery-v2 terminal field roster drifted")
    if (
        terminal.get("schema_version") != module.SCHEMA_VERSION
        or terminal.get("recovery") != module.RECOVERY
        or terminal.get("status") != RECOVERY_V2_STATUS
        or terminal.get("base_campaign") != module.campaign.CAMPAIGN
    ):
        raise GovernanceError("Recovery-v2 terminal identity/status semantics drifted")
    predecessor = terminal.get("predecessor_v1_terminal")
    expected_predecessor = _artifact_record(
        root / LEGACY_RECOVERY_TERMINAL,
        str(LEGACY_RECOVERY_TERMINAL_IDENTITY["sha256"]),
        int(LEGACY_RECOVERY_TERMINAL_IDENTITY["size_bytes"]),
    )
    if predecessor != expected_predecessor:
        raise GovernanceError("Recovery-v2 predecessor-v1 terminal binding drifted")
    if terminal.get("recovery_implementation") != sources:
        raise GovernanceError("Recovery-v2 implementation identity drifted")
    for key, relative in (
        ("scope_contract", "recovery_v2/contract_scope_erratum.json"),
        ("scope_adjudication", "recovery_v2/receipts/scope_adjudication.json"),
    ):
        if terminal.get(key) != legacy._artifact(root / relative):
            raise GovernanceError(f"Recovery-v2 {key} identity drifted")
    scoped = terminal.get("training_scoped_census") or {}
    if (
        set(scoped)
        != {
            "root",
            "artifact_count",
            "total_size_bytes",
            "tree_sha256",
            "original_census_source",
            "original_census_json_pointer",
            "all_original_records_rehashed_at_certification",
            "closed_roster_outside_exclusions",
        }
        or scoped.get("root") != str(root.resolve())
        or scoped.get("artifact_count") != 441
        or scoped.get("total_size_bytes") != 989_858_771
        or scoped.get("tree_sha256")
        != "a2ffd5b61eaf65dc8d0c5df6a1b29867ffabc57f0af869120603ea37591ac261"
        or scoped.get("original_census_source")
        != legacy._artifact(root / "recovery_v1/receipts/validator_adjudication.json")
        or scoped.get("original_census_json_pointer") != "/raw_artifact_hash_census/before"
        or scoped.get("all_original_records_rehashed_at_certification") is not True
        or scoped.get("closed_roster_outside_exclusions") is not True
    ):
        raise GovernanceError("Recovery-v2 scoped training census drifted")
    baseline = terminal.get("prepared_downstream_baseline") or {}
    expected_baseline_records = _expected_legacy_records(root)
    expected_baseline_artifacts = [
        _artifact_record(
            root / str(record["path"]),
            str(record["sha256"]),
            int(record["size_bytes"]),
        )
        for record in expected_baseline_records
    ]
    if (
        set(baseline)
        != {
            "root",
            "canonical_record_schema",
            "artifact_count",
            "total_size_bytes",
            "tree_sha256",
            "artifacts",
        }
        or baseline.get("root") != str(root.resolve())
        or baseline.get("canonical_record_schema")
        != "sorted campaign-relative {path,sha256,size_bytes}"
        or baseline.get("artifact_count") != LEGACY_PREPARED_ARTIFACT_COUNT
        or baseline.get("total_size_bytes") != LEGACY_PREPARED_TOTAL_SIZE
        or baseline.get("tree_sha256") != LEGACY_PREPARED_TREE_SHA256
        or baseline.get("artifacts") != expected_baseline_artifacts
    ):
        raise GovernanceError("Recovery-v2 prepared downstream baseline drifted")
    expected_policy = {
        "immutable_training_scope": "exact original 441-file campaign census",
        "excluded_root_prefixes": [
            "recovery_v1/",
            "recovery_v2/",
            "downstream/",
            "downstream_v2/",
        ],
        "pinned_predecessor_namespace": "recovery_v1/ (exact nine-file graph)",
        "current_certificate_namespace": "recovery_v2/ (exact three-file graph)",
        "prepared_baseline_namespace": "downstream/ (seven named bytes remain immutable)",
        "delegated_growth_namespaces": ["downstream_v2/"],
        "delegated_growth_policy": (
            "regular non-symlink contents wholly delegated to the downstream_v2 controller; "
            "not training-authenticated"
        ),
        "root_analysis_namespace_authorized": False,
    }
    if terminal.get("namespace_policy") != expected_policy:
        raise GovernanceError("Recovery-v2 namespace policy drifted")
    expected_certification_boundary = (
        "training-only immutability over the exact original 441 records; recovery_v1 "
        "and recovery_v2 are pinned receipt namespaces; seven prepared downstream "
        "records and their namespace roster remain immutable; future downstream_v2 "
        "contents are delegated and are not authenticated as training evidence"
    )
    if (
        terminal.get("fit_accounting") != legacy.EXPECTED_FIT_ACCOUNTING
        or terminal.get("execution_accounting") != legacy.EXPECTED_EXECUTION_ACCOUNTING
        or terminal.get("concurrency")
        != {"maximum": 6, "observed_peak": 6, "witness_utc": "2026-08-25T01:10:34.730410+00:00"}
        or terminal.get("certification_boundary") != expected_certification_boundary
    ):
        raise GovernanceError("Recovery-v2 accounting/certification semantics drifted")
    source_manifest_path = Path(
        str((terminal.get("source_population") or {}).get("manifest", {}).get("path", ""))
    )
    if not source_manifest_path.is_file():
        # The v1 prepared contract remains an authenticated semantic predecessor.
        old_contract = legacy._read_json(root / "downstream/contract.json")
        source_manifest_path = Path(str(old_contract["training"]["source_manifest_path"]))
    return {
        "terminal": expected_terminal_identity,
        "status": terminal["status"],
        "scope_contract": terminal["scope_contract"],
        "scope_adjudication": terminal["scope_adjudication"],
        "recovery_implementation": sources,
        "predecessor_v1_terminal": expected_predecessor,
        "training_scoped_census": scoped,
        "prepared_downstream_baseline": baseline,
        "namespace_policy": expected_policy,
        "certification_boundary": expected_certification_boundary,
        "fit_accounting": terminal["fit_accounting"],
        "execution_accounting": terminal["execution_accounting"],
        "concurrency": terminal["concurrency"],
        "source_manifest": legacy._artifact(source_manifest_path),
        "source_manifest_path": str(source_manifest_path.resolve()),
    }


def _implementation_sources_v2() -> list[dict[str, Any]]:
    legacy_sources = _ORIGINAL_IMPLEMENTATION_SOURCES()
    extra = [
        legacy._artifact(Path(__file__).resolve()),
        legacy._artifact(ANALYSIS_V2_TEST),
        *_recovery_v2_source_identities().values(),
    ]
    combined = [*legacy_sources, *extra]
    paths = [str(record["path"]) for record in combined]
    if len(paths) != len(set(paths)):
        raise GovernanceError("Downstream-v2 implementation source graph contains duplicates")
    return combined


def _score_jobs_v2(campaign_root: Path) -> list[dict[str, Any]]:
    root = legacy._safe_campaign_root(campaign_root)
    jobs: list[dict[str, Any]] = []
    for encoder in ENCODERS:
        for target in TARGET_ORDER:
            for seed in SEEDS:
                jobs.append(
                    {
                        "job_id": f"final_v11.downstream_v2.score.{encoder}.{target}.seed{seed}",
                        "encoder": encoder,
                        "target": target,
                        "seed": seed,
                        "fit_count": 0,
                        "contains_target_outcomes": False,
                        "expected_rows": TARGETS[target].slides,
                        "checkpoint": str(legacy.checkpoint_path(root, encoder, seed)),
                        "manifest": str(legacy_blind_path(root, target)),
                        "output": str(
                            continuation_root(root)
                            / f"scores/{encoder}/{target}/seed{seed}.parquet"
                        ),
                    }
                )
    return jobs


def _internal_score_command_v2(
    campaign_root: Path,
    job: Mapping[str, Any],
    *,
    device: str,
    num_workers: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_score-one",
        "--campaign-root",
        str(legacy._safe_campaign_root(campaign_root)),
        "--encoder",
        str(job["encoder"]),
        "--target",
        str(job["target"]),
        "--seed",
        str(job["seed"]),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
    ]


def _delegation_contract() -> dict[str, Any]:
    return {
        "legacy_controller": _legacy_tool_identities()["analysis_controller"],
        "continuation_controller": legacy._artifact(Path(__file__).resolve()),
        "continuation_test": legacy._artifact(ANALYSIS_V2_TEST),
        "runtime_bindings": {
            "legacy.downstream_root": "continuation_root -> <campaign>/downstream_v2",
            "legacy.blind_path": (
                "legacy_blind_path -> <campaign>/downstream/inputs/label_blind/<target>.csv "
                "(read-only)"
            ),
            "legacy.analysis_root": (
                "continuation_analysis_root -> <campaign>/downstream_v2/analysis"
            ),
            "legacy._score_jobs": "_score_jobs_v2 (50 jobs; all outputs downstream_v2)",
            "legacy._load_contract": "_load_contract_v2",
            "legacy._validate_training_bundle": (
                "_validate_training_bundle_v2 -> recovery_v2.validate_scoped_terminal"
            ),
            "legacy._implementation_sources": "_implementation_sources_v2",
            "legacy._internal_score_command": (
                f"_internal_score_command_v2 -> {Path(__file__).resolve()} _score-one"
            ),
            "legacy.seal_inference": "_seal_inference_v2 (deep recovery-v2 gate)",
            "legacy.EXPERIMENT": EXPERIMENT,
            "legacy.TRAINING_RECOVERY_TERMINAL": str(RECOVERY_V2_TERMINAL),
        },
        "runtime_binding_symbol_roster": sorted(_RUNTIME_BINDINGS),
        "subprocess_runtime_initialization": (
            "every v2 _score-one subprocess installs the same scoped legacy bindings and "
            "uses the shallow recovery-v2 worker gate after deep preflight"
        ),
        "legacy_mutations_allowed": False,
        "root_analysis_allowed": False,
    }


def _contract_payload_v2(
    root: Path,
    training: dict[str, Any],
    bundle: dict[str, Any],
    target_records: dict[str, Any],
    snapshots: dict[str, Any],
    feature_stores: dict[str, Any],
    score_job_plan: dict[str, Any],
    *,
    created_utc: str,
) -> dict[str, Any]:
    payload = legacy._contract_payload(
        root,
        training,
        target_records,
        snapshots,
        feature_stores,
        created_utc=created_utc,
    )
    payload["status"] = "prepared_label_blind_continuation_v2_before_outcome_join"
    payload["continuation"] = {
        "namespace": NAMESPACE,
        "predecessor_namespace": LEGACY_NAMESPACE,
        "legacy_prepared_bundle": {
            key: value for key, value in bundle.items() if key != "snapshot_frames"
        },
        "recovery_v2_terminal": training["terminal"],
        "delegation": _delegation_contract(),
        "score_job_plan": score_job_plan,
        "all_new_artifacts_below": str(continuation_root(root)),
        "analysis_output_root": str(continuation_analysis_root(root)),
        "legacy_root_analysis_status": "ABSENT_AND_FORBIDDEN",
    }
    payload["score_contract"]["output_namespace"] = NAMESPACE
    payload["analysis_output_root"] = str(continuation_analysis_root(root))
    return payload


def _target_state(
    root: Path, bundle: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    target_records, source_frames = legacy._target_source_records()
    snapshots = dict(bundle["label_blind_snapshots"])
    frames = dict(bundle["snapshot_frames"])
    for target in TARGET_ORDER:
        if not frames[target].equals(source_frames[target]):
            raise GovernanceError(f"{target}: legacy snapshot differs from frozen blind source")
    target_slides = {
        slide for frame in frames.values() for slide in frame["slide_id"].astype(str).tolist()
    }
    stores = {
        encoder: legacy._validate_pack(legacy.ENCODER_SPECS[encoder], target_slides)
        for encoder in ENCODERS
    }
    return target_records, frames, snapshots, stores


def _validate_contract_identity_graph(payload: Mapping[str, Any]) -> None:
    for identity in legacy._iter_identities(payload):
        legacy._validate_identity(identity)


def _strict_continuation_json_tree(root: Path, contract: Mapping[str, Any]) -> None:
    """Recursively traverse the v2-owned JSON graph and authenticate external leaves.

    Recovery-v2 and the immutable predecessor use their own contextual census
    semantics and are replayed by dedicated validators.  They are therefore
    authenticated leaves here.  Any JSON identity owned by ``downstream_v2``
    must be traversed, and the only contracted child is the score-job plan.
    """
    namespace = continuation_root(root)
    contract_file = legacy.contract_path(root).resolve(strict=False)
    plan_file = (namespace / "jobs/score_jobs.json").resolve(strict=False)
    expected_plan = contract.get("continuation", {}).get("score_job_plan")
    if expected_plan != legacy._artifact(plan_file):
        raise GovernanceError("Downstream-v2 contract does not bind its score-job plan")
    external_json: set[Path] = set()
    owned_json: set[Path] = set()
    for identity in legacy._iter_identities(contract):
        raw_path = Path(str(identity["path"]))
        if raw_path.suffix.casefold() != ".json":
            continue
        resolved = raw_path.resolve(strict=False)
        try:
            resolved.relative_to(namespace)
        except ValueError:
            external_json.add(resolved)
        else:
            owned_json.add(resolved)
    if owned_json != {plan_file}:
        raise GovernanceError(
            f"Downstream-v2 JSON child roster drifted: {sorted(str(path) for path in owned_json)}"
        )
    # The root contract itself is deliberately absent from ``visited``.  The
    # generic safe walker validates every identity in it and recursively opens
    # the one v2-owned JSON child; authenticated external JSON leaves are not
    # reinterpreted using incompatible ownership bases.
    replayed = legacy._strict_json_tree(contract_file, visited=external_json)
    if replayed != contract:
        raise GovernanceError("Downstream-v2 strict JSON-tree replay drifted")


def _load_contract_v2(campaign_root: Path, *, deep: bool = True) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    _assert_root_analysis_absent(root)
    training = _validate_training_bundle_v2(root, deep=deep)
    bundle = validate_legacy_prepared_bundle(root)
    target_records, _frames, snapshots, stores = _target_state(root, bundle)
    path = legacy.contract_path(root)
    stored = legacy._read_json(path)
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise GovernanceError("Downstream-v2 contract lacks creation timestamp")
    expected = _contract_payload_v2(
        root,
        training,
        bundle,
        target_records,
        snapshots,
        stores,
        legacy._artifact(continuation_root(root) / "jobs/score_jobs.json"),
        created_utc=created,
    )
    if stored != expected:
        raise GovernanceError("Stored downstream-v2 contract does not replay exactly")
    _validate_contract_identity_graph(stored)
    if deep:
        # This authenticates the complete historical graph without invoking the
        # obsolete dynamic v1 root census.  Recovery-v2 separately authenticates
        # its own terminal and the scoped 441-record training census.
        legacy._strict_json_tree(root / "downstream/contract.json")
        _strict_continuation_json_tree(root, stored)
    return stored


def _write_json_v2_once(path: Path, payload: Any, *, root: Path) -> None:
    resolved_root = continuation_root(root)
    resolved = Path(path).resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise GovernanceError(f"Refusing continuation write outside downstream_v2: {path}") from exc
    legacy._write_json_once(path, payload)


def _prepare_impl(campaign_root: Path, *, apply: bool) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    training = _validate_training_bundle_v2(root, deep=True)
    bundle = validate_legacy_prepared_bundle(root)
    target_records, _frames, snapshots, stores = _target_state(root, bundle)
    plan = {
        "status": "dry_run_ready_to_adopt_legacy_label_blind_bundle",
        "campaign_root": str(root),
        "continuation_root": str(continuation_root(root)),
        "legacy_prepared_artifacts": 7,
        "label_blind_targets": list(TARGET_ORDER),
        "score_jobs": 50,
        "score_slide_rows": 4_790,
        "outcome_files_opened": False,
        "legacy_files_modified": False,
    }
    if not apply:
        return plan
    destination = continuation_root(root)
    if destination.exists() or destination.is_symlink():
        # Atomic publication means an existing namespace must already be the
        # complete, immutable two-file prepare bundle.  Exact replay makes a
        # repeated apply idempotent and rejects every partial/colliding state.
        return _load_contract_v2(root, deep=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.downstream-v2-stage-", dir=root.parent))
    published = False
    try:
        jobs_document = {"jobs": _score_jobs_v2(root)}
        staged_jobs = stage / "jobs/score_jobs.json"
        legacy._write_json_once(staged_jobs, jobs_document)
        observed_jobs = legacy._artifact(staged_jobs)
        logical_jobs = {
            **observed_jobs,
            "path": str((destination / "jobs/score_jobs.json").resolve(strict=False)),
        }
        payload = _contract_payload_v2(
            root,
            training,
            bundle,
            target_records,
            snapshots,
            stores,
            logical_jobs,
            created_utc=legacy._utcnow(),
        )
        legacy._write_json_once(stage / "contract.json", payload)
        observed = {
            path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()
        }
        if observed != {"contract.json", "jobs/score_jobs.json"} or any(
            path.is_symlink() for path in stage.rglob("*")
        ):
            raise GovernanceError("Staged downstream-v2 prepare roster drifted")
        # Revalidate all immutable inputs immediately before the only publish
        # operation.  The stage lives beside the campaign root and is outside
        # both the recovery scope and the final continuation namespace.
        if _validate_training_bundle_v2(root, deep=True) != training:
            raise GovernanceError("Recovery-v2 evidence changed during prepare staging")
        refreshed = validate_legacy_prepared_bundle(root)
        if {key: value for key, value in refreshed.items() if key != "snapshot_frames"} != {
            key: value for key, value in bundle.items() if key != "snapshot_frames"
        }:
            raise GovernanceError("Legacy prepared evidence changed during prepare staging")
        os.rename(stage, destination)
        published = True
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)
    return _load_contract_v2(root, deep=True)


def _seal_inference_v2(campaign_root: Path) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    _validate_training_bundle_v2(root, deep=True)
    validate_legacy_prepared_bundle(root)
    return _ORIGINAL_SEAL_INFERENCE(root)


_RUNTIME_LOCK = threading.RLock()
_ORIGINAL_IMPLEMENTATION_SOURCES = legacy._implementation_sources
_ORIGINAL_SEAL_INFERENCE = legacy.seal_inference
_RUNTIME_BINDINGS: dict[str, Any] = {
    "downstream_root": continuation_root,
    "blind_path": legacy_blind_path,
    "analysis_root": continuation_analysis_root,
    "_score_jobs": _score_jobs_v2,
    "_load_contract": _load_contract_v2,
    "_validate_training_bundle": _validate_training_bundle_v2,
    "_implementation_sources": _implementation_sources_v2,
    "_internal_score_command": _internal_score_command_v2,
    "seal_inference": _seal_inference_v2,
    "EXPERIMENT": EXPERIMENT,
    "TRAINING_RECOVERY_TERMINAL": RECOVERY_V2_TERMINAL,
}


@contextlib.contextmanager
def _scoped_legacy_runtime() -> Iterator[None]:
    """Install the exact receipted delegation map for one continuation stage."""
    with _RUNTIME_LOCK:
        original = {name: getattr(legacy, name) for name in _RUNTIME_BINDINGS}
        try:
            for name, value in _RUNTIME_BINDINGS.items():
                setattr(legacy, name, value)
            yield
        finally:
            for name, value in original.items():
                setattr(legacy, name, value)


def prepare(campaign_root: Path, *, apply: bool) -> dict[str, Any]:
    with _scoped_legacy_runtime():
        return _prepare_impl(campaign_root, apply=apply)


def preflight(campaign_root: Path, *, apply: bool) -> dict[str, Any]:
    with _scoped_legacy_runtime():
        return legacy.preflight(campaign_root, apply=apply)


def score(
    campaign_root: Path,
    *,
    apply: bool,
    max_workers: int,
    device: str,
    num_workers: int,
) -> dict[str, Any]:
    with _scoped_legacy_runtime():
        root = legacy._safe_campaign_root(campaign_root)
        if legacy.seal_path(root).exists():
            _validate_training_bundle_v2(root, deep=True)
        return legacy.score(
            root,
            apply=apply,
            max_workers=max_workers,
            device=device,
            num_workers=num_workers,
        )


def _score_one(
    campaign_root: Path,
    encoder: str,
    target: str,
    seed: int,
    *,
    device: str,
    num_workers: int,
) -> None:
    with _scoped_legacy_runtime():
        legacy._score_one(
            campaign_root,
            encoder,
            target,
            seed,
            device=device,
            num_workers=num_workers,
        )


def analyze(
    campaign_root: Path,
    *,
    apply: bool,
    n_bootstrap: int = N_BOOTSTRAP,
    test_only_noncanonical_parameters: bool = False,
) -> dict[str, Any]:
    with _scoped_legacy_runtime():
        _load_contract_v2(campaign_root, deep=True)
        return legacy.analyze(
            campaign_root,
            apply=apply,
            n_bootstrap=n_bootstrap,
            test_only_noncanonical_parameters=test_only_noncanonical_parameters,
        )


def _expected_final_inventory(root: Path) -> set[Path]:
    expected = {
        continuation_root(root) / "contract.json",
        continuation_root(root) / "jobs/score_jobs.json",
        continuation_root(root) / "receipts/deep_preflight.json",
        continuation_root(root) / "receipts/scoring_complete.json",
        continuation_root(root) / "inference/environment.json",
        continuation_root(root) / "inference/inference_seal.json",
        *(continuation_analysis_root(root) / name for name in ANALYSIS_FILES),
    }
    for encoder in ENCODERS:
        for target in TARGET_ORDER:
            for seed in SEEDS:
                score_path = (
                    continuation_root(root) / f"scores/{encoder}/{target}/seed{seed}.parquet"
                )
                expected.add(score_path)
                expected.add(score_path.with_suffix(".receipt.json"))
    return expected


def _final_source_graph(root: Path) -> dict[str, dict[str, Any]]:
    namespace = continuation_root(root)
    analysis_dir = continuation_analysis_root(root)
    graph = {
        "downstream_v2_controller": legacy._artifact(Path(__file__).resolve()),
        "downstream_v2_controller_test": legacy._artifact(ANALYSIS_V2_TEST),
        "downstream_v2_contract": legacy._artifact(namespace / "contract.json"),
        "downstream_v2_score_job_plan": legacy._artifact(namespace / "jobs/score_jobs.json"),
        "downstream_v2_deep_preflight": legacy._artifact(
            namespace / "receipts/deep_preflight.json"
        ),
        "downstream_v2_scoring_completion": legacy._artifact(
            namespace / "receipts/scoring_complete.json"
        ),
        "downstream_v2_inference_environment": legacy._artifact(
            namespace / "inference/environment.json"
        ),
        "downstream_v2_inference_seal": legacy._artifact(
            namespace / "inference/inference_seal.json"
        ),
        "downstream_v2_analysis_contract": legacy._artifact(analysis_dir / "contract.json"),
        "downstream_v2_patient_native_logits": legacy._artifact(
            analysis_dir / "patient_native_logits.parquet"
        ),
        "downstream_v2_bootstrap_distributions": legacy._artifact(
            analysis_dir / "bootstrap_distributions.npz"
        ),
        "downstream_v2_results": legacy._artifact(analysis_dir / "results.json"),
        "downstream_v2_analysis_completion": legacy._artifact(
            analysis_dir / "analysis_completion_receipt.json"
        ),
    }
    if tuple(graph) != FINAL_SOURCE_GRAPH_KEYS:
        raise GovernanceError("Downstream-v2 direct source-graph roster drifted")
    return graph


def verify(campaign_root: Path) -> dict[str, Any]:
    with _scoped_legacy_runtime():
        root = legacy._safe_campaign_root(campaign_root)
        _validate_training_bundle_v2(root, deep=True)
        result = legacy.verify(root)
        validate_legacy_prepared_bundle(root)
        expected = _expected_final_inventory(root)
        observed = {path for path in continuation_root(root).rglob("*") if path.is_file()}
        if observed != expected:
            raise GovernanceError(
                "Final downstream-v2 inventory drifted: "
                f"missing={sorted(str(path) for path in expected - observed)}, "
                f"unexpected={sorted(str(path) for path in observed - expected)}"
            )
        if any(path.is_symlink() for path in continuation_root(root).rglob("*")):
            raise GovernanceError("Final downstream-v2 inventory contains a symlink")
        source_graph = _final_source_graph(root)
        analysis_artifacts = {
            key: source_graph[key]
            for key in FINAL_SOURCE_GRAPH_KEYS
            if key.startswith("downstream_v2_analysis_")
            or key
            in {
                "downstream_v2_patient_native_logits",
                "downstream_v2_bootstrap_distributions",
                "downstream_v2_results",
            }
        }
        if len(analysis_artifacts) != 5:
            raise GovernanceError("Downstream-v2 analysis source-graph roster drifted")
        return {
            **result,
            "continuation_namespace": NAMESPACE,
            "continuation_artifacts": len(expected),
            "analysis_root": str(continuation_analysis_root(root)),
            "legacy_prepared_artifacts": LEGACY_PREPARED_ARTIFACT_COUNT,
            "direct_source_graph_count": len(source_graph),
            "source_graph": source_graph,
            "analysis_artifacts": analysis_artifacts,
        }


def status(campaign_root: Path) -> dict[str, Any]:
    with _scoped_legacy_runtime():
        root = legacy._safe_campaign_root(campaign_root)
        observed = legacy.status(root)
        observed["continuation_namespace"] = NAMESPACE
        observed["legacy_prepared_bundle_present"] = all(
            (root / relative).is_file() for relative, _sha256, _size in LEGACY_PREPARED_FILES
        )
        observed["root_analysis_forbidden_present"] = (root / "analysis").exists()
        return observed


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
        return child

    prepare_parser = common("prepare")
    prepare_parser.add_argument("--apply", action="store_true")
    preflight_parser = common("preflight")
    preflight_parser.add_argument("--apply", action="store_true")
    score_parser = common("score")
    score_parser.add_argument("--apply", action="store_true")
    score_parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    score_parser.add_argument("--device", choices=("cuda",), default="cuda")
    score_parser.add_argument("--num-workers", type=int, default=4)
    analyze_parser = common("analyze")
    analyze_parser.add_argument("--apply", action="store_true")
    analyze_parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    analyze_parser.add_argument("--test-only-noncanonical-parameters", action="store_true")
    common("verify")
    common("status")
    internal = common("_score-one")
    internal.add_argument("--encoder", choices=ENCODERS, required=True)
    internal.add_argument("--target", choices=TARGET_ORDER, required=True)
    internal.add_argument("--seed", choices=SEEDS, type=int, required=True)
    internal.add_argument("--device", choices=("cuda",), default="cuda")
    internal.add_argument("--num-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        _print(prepare(args.campaign_root, apply=args.apply))
    elif args.command == "preflight":
        _print(preflight(args.campaign_root, apply=args.apply))
    elif args.command == "score":
        _print(
            score(
                args.campaign_root,
                apply=args.apply,
                max_workers=args.max_workers,
                device=args.device,
                num_workers=args.num_workers,
            )
        )
    elif args.command == "analyze":
        _print(
            analyze(
                args.campaign_root,
                apply=args.apply,
                n_bootstrap=args.n_bootstrap,
                test_only_noncanonical_parameters=args.test_only_noncanonical_parameters,
            )
        )
    elif args.command == "verify":
        _print(verify(args.campaign_root))
    elif args.command == "status":
        _print(status(args.campaign_root))
    elif args.command == "_score-one":
        _score_one(
            args.campaign_root,
            args.encoder,
            args.target,
            args.seed,
            device=args.device,
            num_workers=args.num_workers,
        )
    else:  # pragma: no cover
        raise GovernanceError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
