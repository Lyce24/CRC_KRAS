#!/usr/bin/env python3
"""Additive FINAL-v11 continuation after the immutable v2 prepare incident.

``downstream_v2`` atomically published a correct two-file, label-blind prepare
bundle.  Its post-publication check then rejected a typed recovery-baseline
object as an ambiguous generic census.  Those bytes and the frozen v2 sources
remain immutable.  This controller authenticates and reconstructs that bundle
without calling the defective v2 strict walker, then owns every new artifact
below ``downstream_v2/continuation_v3``.  Target outcomes remain unopened until
the new exactly-once inference seal exists.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import aim1_tcga_surgen_full_pipeline_analysis_v2 as v2  # noqa: E402

legacy = v2.legacy
GovernanceError = v2.GovernanceError

SCHEMA_VERSION = 1
EXPERIMENT = "final_v11_tcga_surgen_two_encoder_full_pipeline_continuation_v3"
DEFAULT_CAMPAIGN_ROOT = v2.DEFAULT_CAMPAIGN_ROOT
NAMESPACE = Path("downstream_v2/continuation_v3")
PREDECESSOR_NAMESPACE = Path("downstream_v2")
ANALYSIS_V3_TEST = REPO / "tests/test_aim1_tcga_surgen_full_pipeline_analysis_v3.py"

V2_CONTROLLER = REPO / "tools/aim1_tcga_surgen_full_pipeline_analysis_v2.py"
V2_CONTROLLER_SHA256 = "5ff574f3143c1a2c695334e6b9afb6fa126e7a7e814f787d156f63511fe92345"
V2_CONTROLLER_SIZE = 49_018
V2_TEST = REPO / "tests/test_aim1_tcga_surgen_full_pipeline_analysis_v2.py"
V2_TEST_SHA256 = "db2c668c8d1be483e0f8aeecd06fd21b099a5eae03efa9011e632618a51fb08e"
V2_TEST_SIZE = 21_028
V2_CONTRACT_SHA256 = "f50eba65b70b8b4b138dfa5a718f237087883de31d3ecc7c4bc7b1a339b84c95"
V2_CONTRACT_SIZE = 46_304
V2_JOB_PLAN_SHA256 = "d0bade099ad22dbb1b8cbe3c734669dd2515191e82a79e77f539f40cbfc200a1"
V2_JOB_PLAN_SIZE = 36_689

SEEDS = v2.SEEDS
ENCODERS = v2.ENCODERS
TARGET_ORDER = v2.TARGET_ORDER
TARGETS = v2.TARGETS
DEFAULT_MAX_WORKERS = v2.DEFAULT_MAX_WORKERS
N_BOOTSTRAP = v2.N_BOOTSTRAP
ANALYSIS_FILES = v2.ANALYSIS_FILES

FINAL_SOURCE_GRAPH_KEYS = tuple(
    key.replace("downstream_v2", "continuation_v3", 1) for key in v2.FINAL_SOURCE_GRAPH_KEYS
)


def continuation_root(campaign_root: Path) -> Path:
    return legacy._safe_campaign_root(campaign_root) / NAMESPACE


def predecessor_root(campaign_root: Path) -> Path:
    return legacy._safe_campaign_root(campaign_root) / PREDECESSOR_NAMESPACE


def legacy_blind_path(campaign_root: Path, target: str) -> Path:
    return v2.legacy_blind_path(campaign_root, target)


def continuation_analysis_root(campaign_root: Path) -> Path:
    return continuation_root(campaign_root) / "analysis"


def _artifact_record(path: Path, sha256: str, size_bytes: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=False)),
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _v2_source_identities() -> dict[str, dict[str, Any]]:
    expected = {
        "controller": _artifact_record(V2_CONTROLLER, V2_CONTROLLER_SHA256, V2_CONTROLLER_SIZE),
        "controller_test": _artifact_record(V2_TEST, V2_TEST_SHA256, V2_TEST_SIZE),
    }
    for name, identity in expected.items():
        if legacy._artifact(Path(str(identity["path"]))) != identity:
            raise GovernanceError(f"Frozen downstream-v2 {name} bytes drifted")
    return expected


def _v2_artifact_identities(root: Path) -> dict[str, dict[str, Any]]:
    return {
        "contract": _artifact_record(
            predecessor_root(root) / "contract.json",
            V2_CONTRACT_SHA256,
            V2_CONTRACT_SIZE,
        ),
        "score_job_plan": _artifact_record(
            predecessor_root(root) / "jobs/score_jobs.json",
            V2_JOB_PLAN_SHA256,
            V2_JOB_PLAN_SIZE,
        ),
    }


def _assert_predecessor_namespace(root: Path) -> None:
    v2._assert_root_analysis_absent(root)
    base = predecessor_root(root)
    if not base.is_dir() or base.is_symlink():
        raise GovernanceError("Published downstream-v2 namespace is missing or symlinked")
    destination = continuation_root(root)
    for path in base.rglob("*"):
        if path.is_symlink():
            raise GovernanceError(f"Downstream-v2 tree contains a symlink: {path}")
        relative = path.relative_to(base)
        try:
            relative.relative_to(Path("continuation_v3"))
        except ValueError:
            if path.is_dir() and relative.as_posix() == "jobs":
                continue
            if path.is_file() and relative.as_posix() in {
                "contract.json",
                "jobs/score_jobs.json",
            }:
                continue
            raise GovernanceError(
                f"Unexpected immutable downstream-v2 path: {relative.as_posix()}"
            ) from None
        if path.is_dir() or path.is_file():
            continue
        raise GovernanceError(f"Continuation-v3 contains a special path: {path}")
    if destination.exists() and (not destination.is_dir() or destination.is_symlink()):
        raise GovernanceError("Continuation-v3 destination is not a regular directory")


def _typed_identity_graph(payload: Mapping[str, Any]) -> None:
    """Validate all identities without interpreting typed census containers generically."""
    for index, identity in enumerate(legacy._iter_identities(payload)):
        raw = identity.get("path")
        if not isinstance(raw, str) or not raw:
            raise GovernanceError(f"Typed identity {index} has no path")
        path = Path(raw)
        if not path.is_absolute() or path.as_posix() != raw or path != path.resolve(strict=False):
            raise GovernanceError(f"Typed identity {index} is non-absolute/normalized/symlinked")
        legacy._validate_identity(identity, expected_path=path)


def validate_predecessor_v2_bundle(campaign_root: Path, *, deep: bool = True) -> dict[str, Any]:
    """Reconstruct the exact v2 prepare while deliberately avoiding its strict walker."""
    root = legacy._safe_campaign_root(campaign_root)
    _assert_predecessor_namespace(root)
    sources = _v2_source_identities()
    artifacts = _v2_artifact_identities(root)
    for name, identity in artifacts.items():
        if legacy._artifact(Path(str(identity["path"]))) != identity:
            raise GovernanceError(f"Published downstream-v2 {name} bytes drifted")
    contract_path = Path(str(artifacts["contract"]["path"]))
    plan_path = Path(str(artifacts["score_job_plan"]["path"]))
    stored = legacy._read_json(contract_path)
    stored_plan = legacy._read_json(plan_path)

    # The nested v2 scope restores its own receipted globals while recreating
    # the historical payload.  `_load_contract_v2` is intentionally never used.
    with v2._scoped_legacy_runtime():
        training = v2._validate_training_bundle_v2(root, deep=deep)
        legacy_bundle = v2.validate_legacy_prepared_bundle(root)
        target_records, frames, snapshots, stores = v2._target_state(root, legacy_bundle)
        created = stored.get("created_utc")
        if not isinstance(created, str) or not created:
            raise GovernanceError("Published downstream-v2 contract lacks created_utc")
        expected = v2._contract_payload_v2(
            root,
            training,
            legacy_bundle,
            target_records,
            snapshots,
            stores,
            artifacts["score_job_plan"],
            created_utc=created,
        )
    if stored != expected:
        raise GovernanceError("Published downstream-v2 contract does not typed-replay exactly")
    _typed_identity_graph(stored)
    expected_plan = {"jobs": v2._score_jobs_v2(root)}
    if stored_plan != expected_plan:
        raise GovernanceError("Published downstream-v2 score-job plan drifted")
    jobs = stored_plan.get("jobs")
    if (
        not isinstance(jobs, list)
        or len(jobs) != 50
        or sum(int(job.get("expected_rows", -1)) for job in jobs if isinstance(job, dict)) != 4_790
        or any(
            not isinstance(job, dict)
            or job.get("fit_count") != 0
            or job.get("contains_target_outcomes") is not False
            for job in jobs
        )
    ):
        raise GovernanceError("Published downstream-v2 score-job semantics drifted")
    return {
        "status": "authenticated_immutable_two_file_predecessor",
        "namespace": str(PREDECESSOR_NAMESPACE),
        "source_implementation": sources,
        "artifacts": artifacts,
        "created_utc": stored["created_utc"],
        "contract_status": stored["status"],
        "score_jobs": 50,
        "score_slide_rows": 4_790,
        "outcomes_opened": False,
        "training": training,
        "legacy_prepared_artifacts": legacy_bundle["artifacts"],
        "legacy_label_blind_snapshots": legacy_bundle["label_blind_snapshots"],
        "target_records": target_records,
        "snapshot_frames": frames,
        "snapshots": snapshots,
        "feature_stores": stores,
    }


def _validate_training_bundle_v3(campaign_root: Path, *, deep: bool = True) -> dict[str, Any]:
    return v2._validate_training_bundle_v2(campaign_root, deep=deep)


def _implementation_sources_v3() -> list[dict[str, Any]]:
    sources = [
        *v2._implementation_sources_v2(),
        legacy._artifact(Path(__file__).resolve()),
        legacy._artifact(ANALYSIS_V3_TEST),
    ]
    paths = [str(item["path"]) for item in sources]
    if len(paths) != len(set(paths)):
        raise GovernanceError("Continuation-v3 implementation graph contains duplicates")
    return sources


def _score_jobs_v3(campaign_root: Path) -> list[dict[str, Any]]:
    root = legacy._safe_campaign_root(campaign_root)
    jobs: list[dict[str, Any]] = []
    for encoder in ENCODERS:
        for target in TARGET_ORDER:
            for seed in SEEDS:
                jobs.append(
                    {
                        "job_id": f"final_v11.continuation_v3.score.{encoder}.{target}.seed{seed}",
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


def _internal_score_command_v3(
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


def _delegation_contract(root: Path) -> dict[str, Any]:
    return {
        "predecessor_v2": {
            **_v2_source_identities(),
            **_v2_artifact_identities(root),
        },
        "continuation_controller": legacy._artifact(Path(__file__).resolve()),
        "continuation_test": legacy._artifact(ANALYSIS_V3_TEST),
        "runtime_bindings": {
            "legacy.downstream_root": (
                "continuation_root -> <campaign>/downstream_v2/continuation_v3"
            ),
            "legacy.blind_path": (
                "legacy_blind_path -> <campaign>/downstream/inputs/label_blind/<target>.csv "
                "(read-only)"
            ),
            "legacy.analysis_root": (
                "continuation_analysis_root -> <campaign>/downstream_v2/continuation_v3/analysis"
            ),
            "legacy._score_jobs": "_score_jobs_v3 (50 jobs; all outputs continuation_v3)",
            "legacy._load_contract": "_load_contract_v3 (typed graph; no v2 strict walker)",
            "legacy._validate_training_bundle": "_validate_training_bundle_v3",
            "legacy._implementation_sources": "_implementation_sources_v3",
            "legacy._internal_score_command": (
                f"_internal_score_command_v3 -> {Path(__file__).resolve()} _score-one"
            ),
            "legacy.seal_inference": "_seal_inference_v3 (deep typed predecessor gate)",
            "legacy.EXPERIMENT": EXPERIMENT,
            "legacy.TRAINING_RECOVERY_TERMINAL": str(v2.RECOVERY_V2_TERMINAL),
        },
        "runtime_binding_symbol_roster": sorted(_RUNTIME_BINDINGS),
        "predecessor_strict_walker_called": False,
        "root_analysis_allowed": False,
    }


def _contract_payload_v3(
    root: Path,
    predecessor: Mapping[str, Any],
    score_job_plan: Mapping[str, Any],
    *,
    created_utc: str,
) -> dict[str, Any]:
    payload = legacy._contract_payload(
        root,
        dict(predecessor["training"]),
        dict(predecessor["target_records"]),
        dict(predecessor["snapshots"]),
        dict(predecessor["feature_stores"]),
        created_utc=created_utc,
    )
    payload["status"] = "prepared_label_blind_continuation_v3_before_outcome_join"
    payload["continuation"] = {
        "namespace": str(NAMESPACE),
        "predecessor_namespace": str(PREDECESSOR_NAMESPACE),
        "predecessor_v2_typed_leaf": {
            key: predecessor[key]
            for key in (
                "status",
                "namespace",
                "source_implementation",
                "artifacts",
                "created_utc",
                "contract_status",
                "score_jobs",
                "score_slide_rows",
                "outcomes_opened",
            )
        },
        "legacy_prepared_artifacts": dict(predecessor["legacy_prepared_artifacts"]),
        "legacy_label_blind_snapshots": dict(predecessor["legacy_label_blind_snapshots"]),
        "recovery_v2_terminal": predecessor["training"]["terminal"],
        "delegation": _delegation_contract(root),
        "score_job_plan": dict(score_job_plan),
        "typed_identity_policy": (
            "absolute normalized identities validated directly; v2 contract is a typed leaf; "
            "only the continuation-v3 job-plan JSON is recursively opened"
        ),
        "all_new_artifacts_below": str(continuation_root(root)),
        "analysis_output_root": str(continuation_analysis_root(root)),
        "legacy_root_analysis_status": "ABSENT_AND_FORBIDDEN",
    }
    payload["score_contract"]["output_namespace"] = str(NAMESPACE)
    payload["analysis_output_root"] = str(continuation_analysis_root(root))
    return payload


def _strict_continuation_graph(root: Path, contract: Mapping[str, Any]) -> None:
    _typed_identity_graph(contract)
    namespace = continuation_root(root)
    plan_path = namespace / "jobs/score_jobs.json"
    wanted = contract.get("continuation", {}).get("score_job_plan")
    if wanted != legacy._artifact(plan_path):
        raise GovernanceError("Continuation-v3 contract does not bind its score-job plan")
    owned_json: set[Path] = set()
    for identity in legacy._iter_identities(contract):
        path = Path(str(identity["path"]))
        if path.suffix.casefold() != ".json":
            continue
        try:
            path.relative_to(namespace)
        except ValueError:
            continue
        owned_json.add(path)
    if owned_json != {plan_path.resolve()}:
        raise GovernanceError(
            f"Continuation-v3 owned JSON roster drifted: {sorted(map(str, owned_json))}"
        )
    if legacy._read_json(plan_path) != {"jobs": _score_jobs_v3(root)}:
        raise GovernanceError("Continuation-v3 score-job plan does not replay")


def _load_contract_v3(campaign_root: Path, *, deep: bool = True) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    predecessor = validate_predecessor_v2_bundle(root, deep=deep)
    path = legacy.contract_path(root)
    stored = legacy._read_json(path)
    created = stored.get("created_utc")
    if not isinstance(created, str) or not created:
        raise GovernanceError("Continuation-v3 contract lacks created_utc")
    expected = _contract_payload_v3(
        root,
        predecessor,
        legacy._artifact(continuation_root(root) / "jobs/score_jobs.json"),
        created_utc=created,
    )
    if stored != expected:
        raise GovernanceError("Stored continuation-v3 contract does not replay exactly")
    _strict_continuation_graph(root, stored)
    return stored


def _prepare_impl(campaign_root: Path, *, apply: bool) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    predecessor = validate_predecessor_v2_bundle(root, deep=True)
    plan = {
        "status": "dry_run_ready_to_continue_from_immutable_v2_prepare",
        "campaign_root": str(root),
        "continuation_root": str(continuation_root(root)),
        "predecessor_v2_artifacts": 2,
        "legacy_prepared_artifacts": 7,
        "label_blind_targets": list(TARGET_ORDER),
        "score_jobs": 50,
        "score_slide_rows": 4_790,
        "outcome_files_opened": False,
        "predecessor_files_modified": False,
    }
    if not apply:
        return plan
    destination = continuation_root(root)
    if destination.exists() or destination.is_symlink():
        return _load_contract_v3(root, deep=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.continuation-v3-stage-", dir=root.parent))
    published = False
    try:
        jobs_document = {"jobs": _score_jobs_v3(root)}
        staged_jobs = stage / "jobs/score_jobs.json"
        legacy._write_json_once(staged_jobs, jobs_document)
        observed_jobs = legacy._artifact(staged_jobs)
        logical_jobs = {
            **observed_jobs,
            "path": str((destination / "jobs/score_jobs.json").resolve(strict=False)),
        }
        payload = _contract_payload_v3(
            root,
            predecessor,
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
            raise GovernanceError("Staged continuation-v3 prepare roster drifted")
        refreshed = validate_predecessor_v2_bundle(root, deep=True)
        stable_keys = {
            key: predecessor[key] for key in predecessor if key not in {"snapshot_frames"}
        }
        refreshed_keys = {
            key: refreshed[key] for key in refreshed if key not in {"snapshot_frames"}
        }
        if refreshed_keys != stable_keys:
            raise GovernanceError("Governed predecessor changed during v3 staging")
        os.rename(stage, destination)
        published = True
        descriptor = os.open(predecessor_root(root), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)
    return _load_contract_v3(root, deep=True)


def _seal_inference_v3(campaign_root: Path) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    _load_contract_v3(root, deep=True)
    return _ORIGINAL_SEAL_INFERENCE(root)


_RUNTIME_LOCK = threading.RLock()
_ORIGINAL_SEAL_INFERENCE = legacy.seal_inference
_RUNTIME_BINDINGS: dict[str, Any] = {
    "downstream_root": continuation_root,
    "blind_path": legacy_blind_path,
    "analysis_root": continuation_analysis_root,
    "_score_jobs": _score_jobs_v3,
    "_load_contract": _load_contract_v3,
    "_validate_training_bundle": _validate_training_bundle_v3,
    "_implementation_sources": _implementation_sources_v3,
    "_internal_score_command": _internal_score_command_v3,
    "seal_inference": _seal_inference_v3,
    "EXPERIMENT": EXPERIMENT,
    "TRAINING_RECOVERY_TERMINAL": v2.RECOVERY_V2_TERMINAL,
}


@contextlib.contextmanager
def _scoped_legacy_runtime() -> Iterator[None]:
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
            _load_contract_v3(root, deep=True)
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
        _load_contract_v3(campaign_root, deep=True)
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
        "continuation_v3_controller": legacy._artifact(Path(__file__).resolve()),
        "continuation_v3_controller_test": legacy._artifact(ANALYSIS_V3_TEST),
        "continuation_v3_contract": legacy._artifact(namespace / "contract.json"),
        "continuation_v3_score_job_plan": legacy._artifact(namespace / "jobs/score_jobs.json"),
        "continuation_v3_deep_preflight": legacy._artifact(
            namespace / "receipts/deep_preflight.json"
        ),
        "continuation_v3_scoring_completion": legacy._artifact(
            namespace / "receipts/scoring_complete.json"
        ),
        "continuation_v3_inference_environment": legacy._artifact(
            namespace / "inference/environment.json"
        ),
        "continuation_v3_inference_seal": legacy._artifact(
            namespace / "inference/inference_seal.json"
        ),
        "continuation_v3_analysis_contract": legacy._artifact(analysis_dir / "contract.json"),
        "continuation_v3_patient_native_logits": legacy._artifact(
            analysis_dir / "patient_native_logits.parquet"
        ),
        "continuation_v3_bootstrap_distributions": legacy._artifact(
            analysis_dir / "bootstrap_distributions.npz"
        ),
        "continuation_v3_results": legacy._artifact(analysis_dir / "results.json"),
        "continuation_v3_analysis_completion": legacy._artifact(
            analysis_dir / "analysis_completion_receipt.json"
        ),
    }
    if tuple(graph) != FINAL_SOURCE_GRAPH_KEYS:
        raise GovernanceError("Continuation-v3 direct source-graph roster drifted")
    return graph


def verify(campaign_root: Path) -> dict[str, Any]:
    with _scoped_legacy_runtime():
        root = legacy._safe_campaign_root(campaign_root)
        _load_contract_v3(root, deep=True)
        result = legacy.verify(root)
        expected = _expected_final_inventory(root)
        observed = {path for path in continuation_root(root).rglob("*") if path.is_file()}
        if observed != expected:
            raise GovernanceError(
                "Final continuation-v3 inventory drifted: "
                f"missing={sorted(map(str, expected - observed))}, "
                f"unexpected={sorted(map(str, observed - expected))}"
            )
        if any(path.is_symlink() for path in continuation_root(root).rglob("*")):
            raise GovernanceError("Final continuation-v3 inventory contains a symlink")
        source_graph = _final_source_graph(root)
        analysis_artifacts = {
            key: source_graph[key]
            for key in FINAL_SOURCE_GRAPH_KEYS
            if key.startswith("continuation_v3_analysis_")
            or key
            in {
                "continuation_v3_patient_native_logits",
                "continuation_v3_bootstrap_distributions",
                "continuation_v3_results",
            }
        }
        if len(analysis_artifacts) != 5:
            raise GovernanceError("Continuation-v3 analysis source graph drifted")
        return {
            **result,
            "continuation_namespace": str(NAMESPACE),
            "continuation_artifacts": len(expected),
            "analysis_root": str(continuation_analysis_root(root)),
            "predecessor_v2_artifacts": 2,
            "legacy_prepared_artifacts": 7,
            "direct_source_graph_count": len(source_graph),
            "source_graph": source_graph,
            "analysis_artifacts": analysis_artifacts,
        }


def status(campaign_root: Path) -> dict[str, Any]:
    with _scoped_legacy_runtime():
        root = legacy._safe_campaign_root(campaign_root)
        observed = legacy.status(root)
        observed["continuation_namespace"] = str(NAMESPACE)
        observed["predecessor_v2_bundle_present"] = all(
            Path(str(identity["path"])).is_file()
            for identity in _v2_artifact_identities(root).values()
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
