#!/usr/bin/env python3
"""Bounded analysis-only mean-validator erratum for sealed FINAL-v11 v3.

The frozen v3 inference campaign completed and sealed 50 score files before any
target outcome join.  Its first analysis attempt opened outcomes only after the
seal, then stopped before creating ``analysis/`` because 18 two-slide target
patients exhibited floating-point association differences between
``mean(mean(seed logits per slide))`` and ``mean(mean(slide logits per seed))``.
The maximum difference is 8.881784197001252e-16.

This additive controller never prepares, scores, refits, calibrates on targets,
or changes the returned patient table.  It authenticates the exact frozen v3
control plane and seal, requires the exact 18-row discrepancy profile, and runs
the frozen validator on a copy whose redundant mean column is recomputed from
the five patient seed columns.  The original patient table and all reported
metrics retain the frozen aggregation values.  The same bounded bridge is used
for analysis verification.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import threading
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import aim1_tcga_surgen_full_pipeline_analysis_v3 as v3  # noqa: E402

legacy = v3.legacy
GovernanceError = v3.GovernanceError

SCHEMA_VERSION = 1
EXPERIMENT = "final_v11_tcga_surgen_analysis_mean_validator_erratum_v4"
DEFAULT_CAMPAIGN_ROOT = v3.DEFAULT_CAMPAIGN_ROOT
ERRATUM_TEST = REPO / "tests/test_aim1_tcga_surgen_full_pipeline_analysis_mean_erratum.py"
ATOL = 1.0e-15
EXPECTED_PATIENT_ROWS = 6_342
EXPECTED_MISMATCH_ROWS = 18
EXPECTED_EXACT_ROWS = 6_324
EXPECTED_MAX_ABS = 8.881784197001252e-16
EXPECTED_PROFILE_SHA256 = "ea5ef308db921565024fb1fa37598ab1d771d886d2b5a98bc6ba23fcb6b9874f"
EXPECTED_BLOCK_COUNTS = {
    "univ1|cptac_primary": 3,
    "univ1|sr1482_metastatic": 10,
    "virchow2_cls|cptac_primary": 1,
    "virchow2_cls|sr1482_metastatic": 4,
}

V3_CONTROLLER_SHA256 = "fadffae63348b2d664ea9f49b2d8126cc0e7869a2eb9bed1245153800b974767"
V3_CONTROLLER_SIZE = 30_953
V3_TEST_SHA256 = "f0329f2f7bd6cf9f549ae6f8763480a8766819c9ad3cf6920f4d364bb61aa32e"
V3_TEST_SIZE = 16_686

V3_CONTROL_PINS: dict[str, tuple[str, str, int]] = {
    "contract": (
        "contract.json",
        "fa5946d13aa848de56e8446dda26afe439d5bd9e05b968aa605e9fcc55782a27",
        32_676,
    ),
    "score_job_plan": (
        "jobs/score_jobs.json",
        "efae666a75cb38044064acec14908f660e014ac63f7b9c331e18f9711299e166",
        37_589,
    ),
    "deep_preflight": (
        "receipts/deep_preflight.json",
        "65b61b0cb28811b6306aa8429d12f921ae65a387905ec9fda8340e2a09409640",
        1_239,
    ),
    "scoring_completion": (
        "receipts/scoring_complete.json",
        "f86ab3a29455f4b8664b877484a5050ee775243f6011567dd4f05626b7cdffb3",
        539,
    ),
    "inference_environment": (
        "inference/environment.json",
        "fd475052f51e5a89bf9052509ddafb17c42b565bd1777339e963c286a7793dc0",
        461,
    ),
    "inference_seal": (
        "inference/inference_seal.json",
        "d7a435ddae080fd8fed814a1d9e659844a83cf140565ed1e57972a3be4b24f45",
        39_086,
    ),
}

EXPECTED_DISCREPANCY_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "encoder": "univ1",
        "dataset": "cptac_primary",
        "patient_id": "CPTAC:05CO047",
        "recomputed_mean": -1.48642578125,
        "stored_associative_mean": -1.4864257812500001,
        "absolute_difference": 2.220446049250313e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "cptac_primary",
        "patient_id": "CPTAC:11CO008",
        "recomputed_mean": 5.653125,
        "stored_associative_mean": 5.653124999999999,
        "absolute_difference": 8.881784197001252e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "cptac_primary",
        "patient_id": "CPTAC:11CO039",
        "recomputed_mean": -5.6453125,
        "stored_associative_mean": -5.645312499999999,
        "absolute_difference": 8.881784197001252e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T016",
        "recomputed_mean": 3.146630859375,
        "stored_associative_mean": 3.1466308593749996,
        "absolute_difference": 4.440892098500626e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T043",
        "recomputed_mean": -1.88203125,
        "stored_associative_mean": -1.8820312499999998,
        "absolute_difference": 2.220446049250313e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T053",
        "recomputed_mean": -0.194140625,
        "stored_associative_mean": -0.19414062499999973,
        "absolute_difference": 2.7755575615628914e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T083",
        "recomputed_mean": 6.98125,
        "stored_associative_mean": 6.981249999999999,
        "absolute_difference": 8.881784197001252e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T225",
        "recomputed_mean": -2.465625,
        "stored_associative_mean": -2.4656249999999997,
        "absolute_difference": 4.440892098500626e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T285",
        "recomputed_mean": -0.090625,
        "stored_associative_mean": -0.09062499999999973,
        "absolute_difference": 2.636779683484747e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T295",
        "recomputed_mean": 0.323046875,
        "stored_associative_mean": 0.32304687500000007,
        "absolute_difference": 5.551115123125783e-17,
    },
    {
        "encoder": "univ1",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T332",
        "recomputed_mean": -0.637109375,
        "stored_associative_mean": -0.6371093750000001,
        "absolute_difference": 1.1102230246251565e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T351",
        "recomputed_mean": -0.940625,
        "stored_associative_mean": -0.9406249999999999,
        "absolute_difference": 1.1102230246251565e-16,
    },
    {
        "encoder": "univ1",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T417",
        "recomputed_mean": 1.0359375,
        "stored_associative_mean": 1.0359375000000002,
        "absolute_difference": 2.220446049250313e-16,
    },
    {
        "encoder": "virchow2_cls",
        "dataset": "cptac_primary",
        "patient_id": "CPTAC:05CO015",
        "recomputed_mean": 0.375390625,
        "stored_associative_mean": 0.37539062499999987,
        "absolute_difference": 1.1102230246251565e-16,
    },
    {
        "encoder": "virchow2_cls",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T025",
        "recomputed_mean": -3.65703125,
        "stored_associative_mean": -3.6570312499999997,
        "absolute_difference": 4.440892098500626e-16,
    },
    {
        "encoder": "virchow2_cls",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T043",
        "recomputed_mean": -1.7439453125,
        "stored_associative_mean": -1.7439453125000002,
        "absolute_difference": 2.220446049250313e-16,
    },
    {
        "encoder": "virchow2_cls",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T044",
        "recomputed_mean": -0.3640625,
        "stored_associative_mean": -0.3640625000000002,
        "absolute_difference": 1.6653345369377348e-16,
    },
    {
        "encoder": "virchow2_cls",
        "dataset": "sr1482_metastatic",
        "patient_id": "SurGen:SR1482_T351",
        "recomputed_mean": -5.840625,
        "stored_associative_mean": -5.840624999999999,
        "absolute_difference": 8.881784197001252e-16,
    },
)

SEED_COLUMNS = tuple(f"logit_seed{seed}" for seed in v3.SEEDS)
_ORIGINAL_PATIENT_VALIDATOR = legacy._validate_patient_output_table
_ORIGINAL_ANALYSIS_CONTRACT_PAYLOAD = legacy._analysis_contract_payload
_RUNTIME_LOCK = threading.RLock()


def _artifact_record(path: Path, sha256: str, size_bytes: int) -> dict[str, Any]:
    return {"path": str(path.resolve(strict=False)), "sha256": sha256, "size_bytes": size_bytes}


def _v3_source_identities() -> dict[str, dict[str, Any]]:
    expected = {
        "controller": _artifact_record(Path(v3.__file__), V3_CONTROLLER_SHA256, V3_CONTROLLER_SIZE),
        "controller_test": _artifact_record(v3.ANALYSIS_V3_TEST, V3_TEST_SHA256, V3_TEST_SIZE),
    }
    for name, identity in expected.items():
        if legacy._artifact(Path(str(identity["path"]))) != identity:
            raise GovernanceError(f"Frozen continuation-v3 {name} bytes drifted")
    return expected


def _v3_control_identities(root: Path) -> dict[str, dict[str, Any]]:
    namespace = v3.continuation_root(root)
    expected = {
        key: _artifact_record(namespace / relative, sha256, size_bytes)
        for key, (relative, sha256, size_bytes) in V3_CONTROL_PINS.items()
    }
    for key, identity in expected.items():
        if legacy._artifact(Path(str(identity["path"]))) != identity:
            raise GovernanceError(f"Frozen continuation-v3 {key} bytes drifted")
    return expected


def _implementation_sources_erratum() -> list[dict[str, Any]]:
    sources = [
        *v3._implementation_sources_v3(),
        legacy._artifact(Path(__file__).resolve()),
        legacy._artifact(ERRATUM_TEST),
    ]
    paths = [str(item["path"]) for item in sources]
    if len(paths) != len(set(paths)):
        raise GovernanceError("Analysis-erratum implementation graph contains duplicates")
    return sources


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _mean_discrepancy_profile(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "analysis_family",
        "encoder",
        "dataset",
        "patient_id",
        "n_slides",
        "mean_logit_5seed",
        *SEED_COLUMNS,
    }
    if required - set(frame) or len(frame) != EXPECTED_PATIENT_ROWS:
        raise GovernanceError("Mean-validator patient schema/row census drifted")
    seed_values = frame[list(SEED_COLUMNS)].to_numpy(float)
    stored = frame["mean_logit_5seed"].to_numpy(float)
    if not np.isfinite(seed_values).all() or not np.isfinite(stored).all():
        raise GovernanceError("Mean-validator inputs contain non-finite values")
    recomputed = frame[list(SEED_COLUMNS)].mean(axis=1).to_numpy(float)
    difference = np.abs(recomputed - stored)
    mismatch = difference != 0.0
    records: list[dict[str, Any]] = []
    for index in np.flatnonzero(mismatch):
        row = frame.iloc[int(index)]
        records.append(
            {
                "encoder": str(row["encoder"]),
                "dataset": str(row["dataset"]),
                "patient_id": str(row["patient_id"]),
                "recomputed_mean": float(recomputed[index]),
                "stored_associative_mean": float(stored[index]),
                "absolute_difference": float(difference[index]),
            }
        )
    records.sort(key=lambda item: (item["encoder"], item["dataset"], item["patient_id"]))
    mismatch_frame = frame.iloc[np.flatnonzero(mismatch)]
    block_counts = dict(
        sorted(
            Counter(
                f"{encoder}|{dataset}"
                for encoder, dataset in zip(
                    mismatch_frame["encoder"].astype(str),
                    mismatch_frame["dataset"].astype(str),
                    strict=True,
                )
            ).items()
        )
    )
    profile = {
        "patient_rows": len(frame),
        "exact_rows": int((~mismatch).sum()),
        "mismatch_rows": int(mismatch.sum()),
        "block_counts": block_counts,
        "max_abs": float(difference.max()),
        "rtol": 0.0,
        "atol": ATOL,
        "allclose": bool(np.allclose(recomputed, stored, rtol=0.0, atol=ATOL)),
        "records_sha256": _canonical_sha256(records),
        "records": records,
    }
    if (
        profile["exact_rows"] != EXPECTED_EXACT_ROWS
        or profile["mismatch_rows"] != EXPECTED_MISMATCH_ROWS
        or profile["block_counts"] != EXPECTED_BLOCK_COUNTS
        or profile["max_abs"] != EXPECTED_MAX_ABS
        or profile["allclose"] is not True
        or profile["records_sha256"] != EXPECTED_PROFILE_SHA256
        or tuple(records) != EXPECTED_DISCREPANCY_RECORDS
        or not mismatch_frame["analysis_family"]
        .astype(str)
        .eq("target_refit_zero_shot_or_sensitivity")
        .all()
        or not pd.to_numeric(mismatch_frame["n_slides"], errors="raise").eq(2).all()
    ):
        raise GovernanceError(f"Five-seed mean discrepancy profile drifted: {profile}")
    return profile


def _bounded_patient_validator(frame: pd.DataFrame) -> None:
    _mean_discrepancy_profile(frame)
    repaired = frame.copy(deep=True)
    repaired["mean_logit_5seed"] = repaired[list(SEED_COLUMNS)].mean(axis=1)
    _ORIGINAL_PATIENT_VALIDATOR(repaired)


def _erratum_contract_record() -> dict[str, Any]:
    return {
        "status": "bounded_floating_point_mean_validator_erratum",
        "returned_patient_table_mutated": False,
        "scores_or_seal_mutated": False,
        "mismatch_rows": EXPECTED_MISMATCH_ROWS,
        "exact_rows": EXPECTED_EXACT_ROWS,
        "block_counts": EXPECTED_BLOCK_COUNTS,
        "max_abs": EXPECTED_MAX_ABS,
        "rtol": 0.0,
        "atol": ATOL,
        "records_sha256": EXPECTED_PROFILE_SHA256,
        "validation_bridge": (
            "run frozen patient validator on a copy whose redundant mean is recomputed "
            "from the five patient seed columns"
        ),
    }


def _analysis_contract_payload_erratum(*args: Any, **kwargs: Any) -> dict[str, Any]:
    value = _ORIGINAL_ANALYSIS_CONTRACT_PAYLOAD(*args, **kwargs)
    value["analysis"]["mean_validator_erratum"] = _erratum_contract_record()
    return value


def _load_contract_erratum(campaign_root: Path, *, deep: bool = True) -> dict[str, Any]:
    original_implementation = legacy._implementation_sources
    original_experiment = legacy.EXPERIMENT
    try:
        legacy._implementation_sources = v3._implementation_sources_v3
        legacy.EXPERIMENT = v3.EXPERIMENT
        return v3._load_contract_v3(campaign_root, deep=deep)
    finally:
        legacy._implementation_sources = original_implementation
        legacy.EXPERIMENT = original_experiment


@contextlib.contextmanager
def _scoped_erratum_runtime() -> Iterator[None]:
    with _RUNTIME_LOCK, v3._scoped_legacy_runtime():
        names = (
            "_validate_patient_output_table",
            "_implementation_sources",
            "_analysis_contract_payload",
            "_load_contract",
            "EXPERIMENT",
        )
        original = {name: getattr(legacy, name) for name in names}
        try:
            legacy._validate_patient_output_table = _bounded_patient_validator
            legacy._implementation_sources = _implementation_sources_erratum
            legacy._analysis_contract_payload = _analysis_contract_payload_erratum
            legacy._load_contract = _load_contract_erratum
            legacy.EXPERIMENT = EXPERIMENT
            yield
        finally:
            for name, value in original.items():
                setattr(legacy, name, value)


def _validate_sealed_v3(root: Path, *, deep: bool) -> dict[str, Any]:
    sources = _v3_source_identities()
    controls = _v3_control_identities(root)
    contract = _load_contract_erratum(root, deep=deep)
    seal = legacy.verify_inference_seal(root)
    if (
        legacy._artifact(v3.continuation_root(root) / "contract.json") != controls["contract"]
        or legacy._artifact(v3.continuation_root(root) / "inference/inference_seal.json")
        != controls["inference_seal"]
        or seal.get("status") != "sealed_before_outcome_join"
        or seal.get("score_artifact_count") != 50
        or seal.get("score_slide_rows") != 4_790
        or seal.get("target_outcome_files_opened") is not False
        or contract.get("governance", {}).get("analysis_requires_inference_seal") is not True
    ):
        raise GovernanceError("Frozen continuation-v3 seal/contract semantics drifted")
    return {"sources": sources, "controls": controls, "seal": seal}


def _analysis_precondition(root: Path) -> None:
    destination = v3.continuation_analysis_root(root)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            observed = list(destination.iterdir())
            if not observed:
                raise GovernanceError("Empty analysis directory is an invalid partial publication")
            if (destination / "analysis_completion_receipt.json").is_file():
                return
        raise GovernanceError("Partial or symlinked analysis output already exists")


def analyze(
    campaign_root: Path,
    *,
    apply: bool,
    n_bootstrap: int = v3.N_BOOTSTRAP,
    test_only_noncanonical_parameters: bool = False,
) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    _analysis_precondition(root)
    completion = v3.continuation_analysis_root(root) / "analysis_completion_receipt.json"
    if completion.is_file():
        return {"status": "already_complete", "verification": verify(root)}
    with _scoped_erratum_runtime():
        _validate_sealed_v3(root, deep=True)
        return legacy.analyze(
            root,
            apply=apply,
            n_bootstrap=n_bootstrap,
            test_only_noncanonical_parameters=test_only_noncanonical_parameters,
        )


INFERENCE_SOURCE_KEYS = (
    "continuation_v3_controller",
    "continuation_v3_controller_test",
    "continuation_v3_contract",
    "continuation_v3_score_job_plan",
    "continuation_v3_deep_preflight",
    "continuation_v3_scoring_completion",
    "continuation_v3_inference_environment",
    "continuation_v3_inference_seal",
)
ANALYSIS_SOURCE_KEYS = (
    "mean_erratum_controller",
    "mean_erratum_controller_test",
    "analysis_contract",
    "patient_native_logits",
    "bootstrap_distributions",
    "results",
    "analysis_completion",
)


def _source_graph(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = _v3_source_identities()
    controls = _v3_control_identities(root)
    analysis_root = v3.continuation_analysis_root(root)
    inference = {
        "continuation_v3_controller": sources["controller"],
        "continuation_v3_controller_test": sources["controller_test"],
        "continuation_v3_contract": controls["contract"],
        "continuation_v3_score_job_plan": controls["score_job_plan"],
        "continuation_v3_deep_preflight": controls["deep_preflight"],
        "continuation_v3_scoring_completion": controls["scoring_completion"],
        "continuation_v3_inference_environment": controls["inference_environment"],
        "continuation_v3_inference_seal": controls["inference_seal"],
    }
    analysis = {
        "mean_erratum_controller": legacy._artifact(Path(__file__).resolve()),
        "mean_erratum_controller_test": legacy._artifact(ERRATUM_TEST),
        "analysis_contract": legacy._artifact(analysis_root / "contract.json"),
        "patient_native_logits": legacy._artifact(analysis_root / "patient_native_logits.parquet"),
        "bootstrap_distributions": legacy._artifact(analysis_root / "bootstrap_distributions.npz"),
        "results": legacy._artifact(analysis_root / "results.json"),
        "analysis_completion": legacy._artifact(analysis_root / "analysis_completion_receipt.json"),
    }
    if tuple(inference) != INFERENCE_SOURCE_KEYS or tuple(analysis) != ANALYSIS_SOURCE_KEYS:
        raise GovernanceError("Analysis-erratum source-graph roster drifted")
    return inference, analysis


def _validate_analysis_contract_erratum(contract: dict[str, Any]) -> None:
    analysis_spec = contract.get("analysis") or {}
    if (
        analysis_spec.get("mean_validator_erratum") != _erratum_contract_record()
        or contract.get("experiment") != EXPERIMENT
        or contract.get("implementation") != _implementation_sources_erratum()
    ):
        raise GovernanceError(
            "Analysis contract does not bind the exact ordered mean-erratum implementation"
        )


def verify(campaign_root: Path) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    with _scoped_erratum_runtime():
        _validate_sealed_v3(root, deep=True)
        result = legacy.verify(root)
        expected = v3._expected_final_inventory(root)
        observed = {path for path in v3.continuation_root(root).rglob("*") if path.is_file()}
        if observed != expected or any(
            path.is_symlink() for path in v3.continuation_root(root).rglob("*")
        ):
            raise GovernanceError("Final erratum-backed v3 inventory/symlink roster drifted")
        contract = legacy._read_json(v3.continuation_analysis_root(root) / "contract.json")
        _validate_analysis_contract_erratum(contract)
        inference_graph, analysis_graph = _source_graph(root)
        return {
            **result,
            "status": "PASS_WITH_BOUNDED_MEAN_VALIDATOR_ERRATUM",
            "mean_validator_erratum": _erratum_contract_record(),
            "direct_source_graph_count": 15,
            "inference_source_graph": inference_graph,
            "analysis_source_graph": analysis_graph,
        }


def status(campaign_root: Path) -> dict[str, Any]:
    root = legacy._safe_campaign_root(campaign_root)
    analysis_root = v3.continuation_analysis_root(root)
    return {
        "campaign_root": str(root),
        "inference_sealed": (
            v3.continuation_root(root) / "inference/inference_seal.json"
        ).is_file(),
        "analysis_directory_present": analysis_root.exists() or analysis_root.is_symlink(),
        "analysis_complete": (analysis_root / "analysis_completion_receipt.json").is_file(),
        "erratum": _erratum_contract_record(),
    }


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
        return child

    analyze_parser = common("analyze")
    analyze_parser.add_argument("--apply", action="store_true")
    analyze_parser.add_argument("--n-bootstrap", type=int, default=v3.N_BOOTSTRAP)
    analyze_parser.add_argument("--test-only-noncanonical-parameters", action="store_true")
    common("verify")
    common("status")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "analyze":
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
    else:  # pragma: no cover
        raise GovernanceError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
