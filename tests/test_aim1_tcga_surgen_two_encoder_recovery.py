"""Focused tests for the additive Aim-1 validator-recovery certifier."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_tcga_surgen_two_encoder_recovery as recovery  # noqa: E402


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    current = target
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _full_material(root: Path, seed: int = 42) -> dict[str, Any]:
    material: dict[str, Any] = {}
    for dotted, expected in recovery.campaign._material_expectations(
        root,
        seed,
        encoder="virchow2_cls",
        skip_finalize=False,
    ).items():
        _set_dotted(material, dotted, expected)
    return material


def _reduced_material(root: Path, seed: int = 42) -> dict[str, Any]:
    material = _full_material(root, seed)
    material.pop("extraction")
    return material


def _identity(path: str, token: str = "0") -> dict[str, Any]:
    return {"path": path, "sha256": token * 64, "size_bytes": 1}


def _fake_evidence(tmp_path: Path) -> dict[str, Any]:
    pinned = {
        key: _identity(str(tmp_path / f"{key}.json"), str(index % 10))
        for index, key in enumerate(
            (
                "controller",
                "controller_test",
                "training_implementation",
                "base_contract",
                "base_preflight",
            ),
            start=1,
        )
    }
    univ1 = [
        {
            "seed": seed,
            "job_receipt": _identity(
                str(tmp_path / f"stock/univ1/seed{seed}.json"), str(seed % 10)
            ),
            "request": _identity(str(tmp_path / f"requests/u{seed}.json")),
            "request_created_utc": "2026-08-25T01:10:27+00:00",
            "log": _identity(str(tmp_path / f"logs/u{seed}.log")),
            "finished_utc": f"2026-08-25T01:{30 + seed - 42:02d}:00+00:00",
            "artifacts": {},
            "oof_fits": 0,
            "refits": 1,
            "total_fits": 1,
            "attempt": 1,
            "retries": 0,
        }
        for seed in recovery.campaign.SEEDS
    ]
    v2 = []
    defect = []
    for seed in recovery.campaign.SEEDS:
        v2.append(
            {
                "seed": seed,
                "request": _identity(str(tmp_path / f"requests/v{seed}.json")),
                "request_created_utc": "2026-08-25T01:10:28+00:00",
                "log": _identity(str(tmp_path / f"logs/v{seed}.log")),
                "contracted_command": {
                    "contract_job_sha256": "1" * 64,
                    "training_command_sha256": "2" * 64,
                    "training_command_argc": 3,
                    "extraction_overrides": recovery.BOUNDED_EXTRACTION_VALUES,
                },
                "native": {
                    "training_identity": {
                        "bridge": {
                            "accepted_absent_keys_only": list(recovery.BOUNDED_ABSENT_IDENTITY_KEYS)
                        }
                    },
                    "native_status": {
                        "started_utc": "2026-08-25T01:10:34+00:00",
                        "derived_finished_utc": "2026-08-25T02:40:34+00:00",
                    },
                    "oof_fits": 5,
                    "refits": 1,
                    "total_fits": 6,
                },
                "attempt": 1,
                "retries": 0,
            }
        )
        defect.append(
            {
                "seed": seed,
                "wrapper_returncode_evidence": "deterministic replay",
                "original_wrapper_returncode": 1,
            }
        )
    return {
        "pinned": pinned,
        "roster": {"stock_virchow2_job_receipts": 0},
        "pack": {"path": "pack", "artifacts": {}},
        "adopted_oof": [{"seed": seed} for seed in recovery.campaign.SEEDS],
        "univ1": univ1,
        "v2": v2,
        "defect": defect,
        "concurrency": {
            "configured_max_workers": 6,
            "observed_peak_parallel_workers": 6,
            "witness_utc": "2026-08-25T01:10:34+00:00",
        },
        "raw_census": {
            "root": str(tmp_path),
            "excluded_prefix": "recovery_v1/",
            "artifact_count": 1,
            "total_size_bytes": 1,
            "tree_sha256": "f" * 64,
            "artifacts": [],
        },
    }


def test_frozen_incident_pins_are_exact() -> None:
    assert recovery.PINNED_BASE_ARTIFACTS["controller"]["sha256"] == (
        "4635042e76c3ad3f8fc9aea46e478881681d4a1dfeaef046e0968424d7e2bcc7"
    )
    assert recovery.PINNED_BASE_ARTIFACTS["controller_test"]["sha256"] == (
        "9609411e1bff2feb7517a23931a1e3907eb4984bee90da5d7cdd95c750ae3a16"
    )
    assert recovery.PINNED_BASE_ARTIFACTS["training_implementation"]["sha256"] == (
        "a22cd0579533cbd43941d338a15d0eb6b972f8c440f4a622361ca22de8e939b6"
    )
    assert recovery.PINNED_BASE_ARTIFACTS["base_contract"]["sha256"] == (
        "0aa475fa30123b5f2e44c6d3b54de5ade5ebcadeb5a2aa08224558bdc41509d9"
    )
    assert recovery.PINNED_BASE_ARTIFACTS["base_preflight"]["sha256"] == (
        "b6643d123470030d7ea237d83bdcc7eabbf6446ddb3fa24f7430a357c2ed973d"
    )


def test_bounded_bridge_accepts_exact_three_schema_v2_omissions(
    tmp_path: Path,
) -> None:
    proof = recovery._validate_bounded_identity_material(_reduced_material(tmp_path), tmp_path, 42)

    assert proof["accepted_absent_keys_only"] == list(recovery.BOUNDED_ABSENT_IDENTITY_KEYS)
    assert proof["values_proven_outside_reduced_identity"] == {
        "extraction.coords_dir": "20x_224px_0px_overlap_mpp0.5",
        "extraction.coords_subdir": "20x_224px_0px_overlap_mpp0.5",
        "extraction.patch_size": 224,
    }
    assert proof["other_material_mismatches"] == 0


def test_bounded_bridge_rejects_a_fourth_missing_material_key(
    tmp_path: Path,
) -> None:
    material = _reduced_material(tmp_path)
    del material["model"]["name"]

    with pytest.raises(recovery.ContractError, match="not exact"):
        recovery._validate_bounded_identity_material(material, tmp_path, 42)


def test_bounded_bridge_rejects_wrong_present_material_value(
    tmp_path: Path,
) -> None:
    material = _reduced_material(tmp_path)
    material["training"]["lr"] = 9e-4

    with pytest.raises(recovery.ContractError, match="not exact"):
        recovery._validate_bounded_identity_material(material, tmp_path, 42)


def test_bounded_bridge_rejects_extraction_inside_reduced_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(recovery.ContractError, match="material sections drifted"):
        recovery._validate_bounded_identity_material(_full_material(tmp_path), tmp_path, 42)


def test_full_config_requires_exact_extraction_values(tmp_path: Path) -> None:
    material = _full_material(tmp_path)
    assert (
        recovery._validate_full_material_config(material, tmp_path, 42, context="test")
        == recovery.BOUNDED_EXTRACTION_VALUES
    )
    material["extraction"]["patch_size"] = 256

    with pytest.raises(recovery.ContractError, match="material recipe mismatch"):
        recovery._validate_full_material_config(material, tmp_path, 42, context="test")


def test_immutable_contract_command_proves_each_extraction_override(
    tmp_path: Path,
) -> None:
    contract = recovery.campaign.build_contract(tmp_path, created_utc="fixed")
    proof = recovery._validate_command_proof(contract, tmp_path, 42)

    assert proof["extraction_overrides"] == recovery.BOUNDED_EXTRACTION_VALUES
    assert proof["training_command_argc"] > 30


def test_immutable_contract_command_rejects_any_command_drift(
    tmp_path: Path,
) -> None:
    contract = recovery.campaign.build_contract(tmp_path, created_utc="fixed")
    contract = copy.deepcopy(contract)
    job = next(
        item for item in contract["jobs"] if item["kind"] == "virchow2_full" and item["seed"] == 42
    )
    job["training_command"].remove("extraction.patch_size=224")

    with pytest.raises(recovery.ContractError, match="contract job drifted"):
        recovery._validate_command_proof(contract, tmp_path, 42)


def test_v2_request_rejects_undeclared_extra_key(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    recovery.campaign._write_json_once(contract, {"status": "sealed"})
    manifest = recovery.campaign.manifest_path(tmp_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("slide_id\nslide-1\n", encoding="utf-8")
    split = recovery.campaign.split_dir(tmp_path) / "splits.parquet"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_bytes(b"split")
    contract_identity = recovery._artifact(contract)
    request = {
        "schema_version": recovery.campaign.SCHEMA_VERSION,
        "campaign": recovery.campaign.CAMPAIGN,
        "status": "requested",
        "created_utc": "2026-08-25T01:10:28+00:00",
        "kind": "virchow2_full",
        "encoder": "Virchow2-CLS",
        "seed": 42,
        "oof_fits": 5,
        "refits": 1,
        "total_fits": 6,
        "attempt": 1,
        "contract": contract_identity,
        "manifest": recovery._artifact(manifest),
        "split": recovery._artifact(split),
        "output": str(recovery.campaign.run_dir(tmp_path, "virchow2_full", 42)),
        "undeclared": "reject",
    }
    recovery.campaign._write_json_once(
        recovery.campaign.request_path(tmp_path, "virchow2_full", 42),
        request,
    )

    with pytest.raises(recovery.ContractError, match="request is not exact"):
        recovery._validate_request(tmp_path, 42, contract_identity)


def test_recovery_paths_are_additive_and_never_stock(tmp_path: Path) -> None:
    paths = recovery.recovery_write_inventory(tmp_path)
    relative = {path.relative_to(tmp_path).as_posix() for path in paths}

    assert len(paths) == 9
    assert relative == {
        "recovery_v1/contract_erratum.json",
        "recovery_v1/receipts/scheduler_recovery.json",
        "recovery_v1/receipts/validator_adjudication.json",
        "recovery_v1/receipts/training_complete_recovered.json",
        *(
            f"recovery_v1/receipts/jobs/virchow2_full/seed{seed}.json"
            for seed in recovery.campaign.SEEDS
        ),
    }
    assert "receipts/scheduler.json" not in relative
    assert "receipts/training_complete.json" not in relative
    assert not any(path.startswith("receipts/jobs/virchow2_full") for path in relative)


def test_production_root_is_exactly_pinned(tmp_path: Path) -> None:
    with pytest.raises(recovery.ContractError, match="single affected production root"):
        recovery._assert_production_root(tmp_path)


def test_concurrency_witness_proves_exact_six_from_intervals(tmp_path: Path) -> None:
    for kind in ("virchow2_full", "univ1_refit"):
        for seed in recovery.campaign.SEEDS:
            recovery.campaign._write_json_once(
                recovery.campaign.request_path(tmp_path, kind, seed),
                {
                    "created_utc": (
                        "2026-08-25T01:10:28+00:00"
                        if kind == "virchow2_full"
                        else "2026-08-25T01:10:27+00:00"
                    )
                },
            )
    v2 = [
        {
            "seed": seed,
            "native": {
                "native_status": {
                    "started_utc": f"2026-08-25T01:10:{30 + seed - 42:02d}+00:00",
                    "derived_finished_utc": "2026-08-25T02:30:00+00:00",
                }
            },
        }
        for seed in recovery.campaign.SEEDS
    ]
    univ1 = [
        {
            "seed": seed,
            "finished_utc": (
                "2026-08-25T01:22:54+00:00" if seed == 42 else "2026-08-25T02:40:00+00:00"
            ),
        }
        for seed in recovery.campaign.SEEDS
    ]

    witness = recovery._concurrency_witness(v2, univ1, tmp_path)

    assert witness["observed_peak_parallel_workers"] == 6
    assert witness["active_jobs"] == [
        "univ1_refit.seed42",
        "virchow2_full.seed42",
        "virchow2_full.seed43",
        "virchow2_full.seed44",
        "virchow2_full.seed45",
        "virchow2_full.seed46",
    ]


def _make_stock_univ1_fixture(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    request_attempt: int = 1,
    receipt_request_sha: str | None = None,
    request_extra: bool = False,
) -> None:
    seed = 42
    recovery.campaign._write_json_once(recovery.campaign.contract_path(root), {"ok": True})
    manifest = recovery.campaign.manifest_path(root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("slide_id\nslide-1\n", encoding="utf-8")
    split = recovery.campaign.split_dir(root) / "splits.parquet"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_bytes(b"split")
    request_path = recovery.campaign.request_path(root, "univ1_refit", seed)
    request = {
        "schema_version": recovery.campaign.SCHEMA_VERSION,
        "campaign": recovery.campaign.CAMPAIGN,
        "status": "requested",
        "created_utc": "2026-08-25T01:10:27+00:00",
        "kind": "univ1_refit",
        "encoder": "UNI-v1",
        "seed": seed,
        "oof_fits": 0,
        "refits": 1,
        "total_fits": 1,
        "attempt": request_attempt,
        "contract": recovery._artifact(recovery.campaign.contract_path(root)),
        "manifest": recovery._artifact(manifest),
        "split": recovery._artifact(split),
        "output": str(recovery.campaign.run_dir(root, "univ1_refit", seed)),
    }
    if request_extra:
        request["undeclared"] = "reject"
    recovery.campaign._write_json_once(request_path, request)
    log = recovery.campaign.log_path(root, "univ1_refit", seed)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("complete\n", encoding="utf-8")
    directory = recovery.campaign.run_dir(root, "univ1_refit", seed)
    checkpoint = directory / "final/refit/model.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    request_identity = recovery._artifact(request_path)
    if receipt_request_sha is not None:
        request_identity = {**request_identity, "sha256": receipt_request_sha}
    receipt = {
        "status": "completed",
        "finished_utc": "2026-08-25T01:22:54+00:00",
        "attempt": 1,
        "oof_fits": 0,
        "refits": 1,
        "total_fits": 1,
        "request": request_identity,
        "log": recovery._artifact(log),
        "artifacts": {},
    }
    recovery.campaign._write_json_once(
        recovery.campaign.job_receipt_path(root, "univ1_refit", seed), receipt
    )
    monkeypatch.setattr(
        recovery.campaign,
        "_validate_job",
        lambda observed_root, kind, observed_seed: receipt,
    )


def test_stock_univ1_request_and_receipt_control_plane_are_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _make_stock_univ1_fixture(monkeypatch, tmp_path)

    record = recovery._validate_stock_univ1(tmp_path, 42)

    assert record["attempt"] == 1
    assert record["retries"] == 0
    assert record["request_created_utc"] == "2026-08-25T01:10:27+00:00"
    assert record["request"] == recovery._artifact(
        recovery.campaign.request_path(tmp_path, "univ1_refit", 42)
    )


def test_stock_univ1_rejects_semantically_drifted_request_even_if_receipted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _make_stock_univ1_fixture(monkeypatch, tmp_path, request_attempt=2)

    with pytest.raises(recovery.ContractError, match="request is not exact"):
        recovery._validate_stock_univ1(tmp_path, 42)


def test_stock_univ1_rejects_undeclared_request_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _make_stock_univ1_fixture(monkeypatch, tmp_path, request_extra=True)

    with pytest.raises(recovery.ContractError, match="request is not exact"):
        recovery._validate_stock_univ1(tmp_path, 42)


def test_stock_univ1_rejects_receipt_request_or_log_identity_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _make_stock_univ1_fixture(
        monkeypatch,
        tmp_path,
        receipt_request_sha="f" * 64,
    )

    with pytest.raises(recovery.ContractError, match="request/log identity drifted"):
        recovery._validate_stock_univ1(tmp_path, 42)


def test_raw_census_excludes_only_additive_recovery_namespace(
    tmp_path: Path,
) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/data.bin").write_bytes(b"raw")
    (tmp_path / "recovery_v1").mkdir()
    (tmp_path / "recovery_v1/new.json").write_text("{}", encoding="utf-8")

    census = recovery._raw_artifact_census(tmp_path)

    assert census["artifact_count"] == 1
    assert [record["path"] for record in census["artifacts"]] == ["raw/data.bin"]


def test_raw_census_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"raw")
    (tmp_path / "alias.bin").symlink_to(target)

    with pytest.raises(recovery.ContractError, match="symlinked"):
        recovery._raw_artifact_census(tmp_path)


def test_original_defect_reproduction_is_exactly_three_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    directory = recovery.campaign.run_dir(tmp_path, "virchow2_full", 42)
    directory.mkdir(parents=True)
    recovery.campaign._write_json_once(
        directory / "training_identity.json",
        {"payload": {"material_config": _reduced_material(tmp_path)}},
    )
    expectations = recovery.campaign._material_expectations(
        tmp_path,
        42,
        encoder="virchow2_cls",
        skip_finalize=False,
    )
    expected_mismatch = {
        key: {
            "expected": recovery.BOUNDED_EXTRACTION_VALUES[key],
            "observed": "<missing>",
        }
        for key in expectations
        if key in recovery.BOUNDED_ABSENT_IDENTITY_KEYS
    }
    message = (
        f"virchow2_cls/seed42: training identity material recipe mismatch: {expected_mismatch}"
    )

    def fail(*args: Any, **kwargs: Any) -> None:
        raise recovery.ContractError(message)

    monkeypatch.setattr(recovery.campaign, "_native_validation", fail)
    proof = recovery._reproduce_original_defect(tmp_path, 42)

    assert proof["native_study_train_returncode"] == 0
    assert proof["original_wrapper_returncode"] == 1
    assert proof["missing_keys_only"] == list(recovery.BOUNDED_ABSENT_IDENTITY_KEYS)
    assert proof["other_mismatches"] == 0


@pytest.mark.skipif(
    not (
        recovery.DEFAULT_OUTPUT_ROOT / "train/e0/virchow2_cls/seed46/training_identity.json"
    ).is_file(),
    reason="live immutable incident evidence is unavailable",
)
def test_live_seed46_regression_binds_exact_defect_and_checkpoint_topology() -> None:
    root = recovery.DEFAULT_OUTPUT_ROOT
    directory = recovery.campaign.run_dir(root, "virchow2_full", 46)
    identity = recovery._read_json(directory / "training_identity.json")
    material = identity["payload"]["material_config"]

    proof = recovery._validate_bounded_identity_material(material, root, 46)
    assert proof["accepted_absent_keys_only"] == list(recovery.BOUNDED_ABSENT_IDENTITY_KEYS)
    with pytest.raises(recovery.ContractError) as caught:
        recovery.campaign._validate_material_config(
            material,
            root,
            46,
            encoder="virchow2_cls",
            skip_finalize=False,
            context="training identity",
        )
    message = str(caught.value)
    positions = [
        message.index(key)
        for key in (
            "extraction.patch_size",
            "extraction.coords_dir",
            "extraction.coords_subdir",
        )
    ]
    assert positions == sorted(positions)
    assert message.count("'<missing>'") == 3
    expected_checkpoints = {
        directory / "final/best_fold/model.ckpt",
        directory / "final/refit/model.ckpt",
    }
    for fold in recovery.campaign.FOLDS:
        checkpoint_dir = directory / f"fold_{fold}/checkpoints"
        best = [path for path in checkpoint_dir.rglob("*.ckpt") if path.name != "last.ckpt"]
        assert len(best) == 1
        expected_checkpoints.update({best[0], checkpoint_dir / "last.ckpt"})
    assert set(directory.rglob("*.ckpt")) == expected_checkpoints


def test_erratum_contract_has_zero_fit_and_zero_mutation_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = _fake_evidence(tmp_path)
    monkeypatch.setattr(
        recovery,
        "_recovery_implementation",
        lambda: {
            "controller": _identity("recovery.py"),
            "controller_test": _identity("test_recovery.py"),
        },
    )

    payload = recovery._erratum_payload(evidence, created_utc="2026-08-25T03:00:00+00:00")

    assert payload["defect"]["accepted_absent_keys_only"] == list(
        recovery.BOUNDED_ABSENT_IDENTITY_KEYS
    )
    assert payload["defect"]["other_missing_keys_allowed"] == 0
    assert payload["mutation_policy"]["raw_campaign_writes"] == 0
    assert payload["mutation_policy"]["training_or_refit_runs"] == 0
    assert payload["fit_accounting"]["recovery_new_fits"] == 0


def test_recovered_v2_receipt_keeps_native_rc0_and_wrapper_rc1_distinct(
    tmp_path: Path,
) -> None:
    evidence = _fake_evidence(tmp_path)
    payload = recovery._recovered_v2_job_payload(
        evidence,
        42,
        created_utc="2026-08-25T03:00:00+00:00",
        erratum_identity=_identity("erratum.json"),
        adjudication_identity=_identity("adjudication.json"),
    )

    assert payload["status"] == "recovered_complete_via_bounded_erratum"
    assert payload["native_study_train_returncode"] == 0
    assert payload["original_wrapper_returncode"] == 1
    assert payload["failure_phase"] == "post_fit_native_validator"
    assert payload["attempt"] == 1
    assert payload["retries"] == 0
    assert payload["total_fits"] == 6


def test_terminal_schema_has_exact_chain_fit_and_execution_accounting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence = _fake_evidence(tmp_path)
    monkeypatch.setattr(recovery, "DEFAULT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        recovery,
        "_artifact",
        lambda path: _identity(str(path)),
    )
    monkeypatch.setattr(
        recovery,
        "_recovery_implementation",
        lambda: {
            "controller": _identity("recovery.py"),
            "controller_test": _identity("test_recovery.py"),
        },
    )
    recovered = {
        seed: _identity(f"recovery_v1/receipts/jobs/virchow2_full/seed{seed}.json")
        for seed in recovery.campaign.SEEDS
    }

    terminal = recovery._terminal_payload(
        evidence,
        recovered,
        created_utc="2026-08-25T03:00:00+00:00",
        erratum_identity=_identity("erratum.json"),
        adjudication_identity=_identity("adjudication.json"),
        scheduler_identity=_identity("scheduler.json"),
    )

    assert terminal["status"] == recovery.RECOVERY_STATUS
    assert terminal["fit_accounting"] == {
        "adopted_oof_fits": 25,
        "new_oof_fits": 25,
        "new_refits": 10,
        "new_fits": 35,
        "physical_new_fits": 35,
        "operational_lineage_fits": 60,
        "recovery_new_fits": 0,
        "hidden_fits": 0,
    }
    assert terminal["execution_accounting"] == {
        "job_count": 10,
        "attempts_per_job": 1,
        "total_attempts": 10,
        "retries": 0,
    }
    assert terminal["concurrency"]["maximum"] == 6
    assert terminal["concurrency"]["observed_peak"] == 6
    assert len(terminal["adopted_univ1_oof_chains"]) == 5
    assert len(terminal["new_chains"]) == 10
    assert len(terminal["stock_univ1_job_receipts"]) == 5
    assert len(terminal["recovered_virchow2_job_receipts"]) == 5
    assert len(terminal["new_job_receipts"]) == 10
    assert terminal["stock_receipts_fabricated"] is False


def test_atomic_publication_changes_only_recovery_v1_and_refuses_reseal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "immutable.bin").write_bytes(b"raw")
    before = recovery._raw_artifact_census(root)
    evidence = {"raw_census": before}

    def materialize(
        observed_root: Path,
        stage: Path,
        observed_evidence: dict[str, Any],
        *,
        created_utc: str,
    ) -> None:
        assert observed_root == root
        assert observed_evidence == evidence
        assert dt.datetime.fromisoformat(created_utc).tzinfo is not None
        recovery.campaign._write_json_once(stage / "marker.json", {"status": "ok"})

    monkeypatch.setattr(recovery, "_materialize_recovery", materialize)
    recovery._publish_atomic(root, evidence)

    assert (root / "recovery_v1/marker.json").is_file()
    assert recovery._raw_artifact_census(root) == before
    with pytest.raises(recovery.ContractError, match="exactly once"):
        recovery._publish_atomic(root, evidence)


def test_public_terminal_validator_replays_deep_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "recovery_v1").mkdir()
    evidence = {"status": "audited"}
    terminal = {"status": recovery.RECOVERY_STATUS}
    monkeypatch.setattr(recovery, "_assert_production_root", lambda root: Path(root))

    def audit(root: Path, *, deep_pack: bool, allow_published_recovery: bool) -> dict[str, str]:
        assert root == tmp_path
        assert deep_pack is True
        assert allow_published_recovery is True
        return evidence

    monkeypatch.setattr(recovery, "_audit_base", audit)
    monkeypatch.setattr(
        recovery,
        "_verify_recovery_files",
        lambda root, observed: (
            terminal
            if root == tmp_path and observed is evidence
            else pytest.fail("wrong replay evidence")
        ),
    )

    assert recovery.validate_recovered_terminal(tmp_path) is terminal


def test_certify_requires_apply_without_auditing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        recovery,
        "_audit_base",
        lambda *args, **kwargs: pytest.fail("audit must not run without --apply"),
    )
    args = argparse.Namespace(output_root=recovery.DEFAULT_OUTPUT_ROOT, apply=False)

    with pytest.raises(recovery.ContractError, match="requires --apply"):
        recovery.cmd_certify(args)


def test_parser_exposes_only_read_only_and_additive_recovery_stages() -> None:
    parser = recovery.build_parser()
    choices = next(
        action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(choices) == {"plan", "audit", "certify", "verify", "status"}
