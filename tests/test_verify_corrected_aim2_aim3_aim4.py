"""Focused fail-closed contracts for the full corrected-results verifier."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import verify_corrected_aim2_aim3_aim4 as verify  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _e4_manifest(root: Path, relatives: tuple[str, ...]) -> dict[str, dict]:
    artifacts = {}
    for relative in relatives:
        observed = verify._identity(root / relative)
        artifacts[relative] = {**observed, "path": relative}
    return artifacts


def _payload_inventory(root: Path) -> list[dict]:
    excluded = {"completion_receipt.json", "verification_receipt.json"}
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": verify._sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.relative_to(root).as_posix() not in excluded
    ]


def test_seven_roots_are_explicit_distinct_and_non_nested(tmp_path: Path) -> None:
    roots = {f"r{index}": (tmp_path / f"r{index}").resolve() for index in range(7)}
    for root in roots.values():
        root.mkdir()
    verify._validate_distinct_roots(roots)
    with pytest.raises(verify.FullVerificationError, match="distinct"):
        verify._validate_distinct_roots({**roots, "r6": roots["r0"]})
    nested = roots["r0"] / "child"
    nested.mkdir()
    with pytest.raises(verify.FullVerificationError, match="nested"):
        verify._validate_distinct_roots({**roots, "r6": nested})
    with pytest.raises(verify.FullVerificationError, match="absolute"):
        verify._absolute_root(Path("relative"), "test root")


def test_partial_roots_fail_before_any_delegate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = [tmp_path / f"r{index}" for index in range(7)]
    for root in roots:
        root.mkdir()

    def forbidden(*args: object, **kwargs: object) -> dict:
        raise AssertionError(f"delegate called for partial root: {args}, {kwargs}")

    monkeypatch.setattr(verify.aim23, "verify", forbidden)
    monkeypatch.setattr(verify, "_delegate_e4", forbidden)
    monkeypatch.setattr(verify, "_delegate_stability", forbidden)
    with pytest.raises(verify.FullVerificationError, match="partial/incomplete"):
        verify.verify(
            aim2_core_root=roots[0],
            e2c_root=roots[1],
            aim2_refresh_root=roots[2],
            aim3_root=roots[3],
            aim3_repeated_root=roots[4],
            aim4_root=roots[5],
            aim4_stability_root=roots[6],
            aim4_pathology_state="pending",
        )


def _touch_required_precheck_markers(roots: dict[str, Path]) -> None:
    required = {
        "aim2_core": ("lineage_start.json",),
        "e2c_offset": (
            "lineage_complete.json",
            "analysis/e2c_native_logit_offset_cap8192.json",
        ),
        "aim2_e2d_refresh": ("refresh_receipt.json", "verification_receipt.json"),
        "aim3_corrected": (
            "lineage_start.json",
            "lineage_complete.json",
            "analysis/aim3_corrected.json",
        ),
        "aim3_repeated_controls": (
            "lineage_start.json",
            "lineage_complete.json",
            "analysis/aim3_repeated_control_report.json",
            "analysis/bootstrap_distributions.npz",
            "analysis/analysis_audit.json",
        ),
        "aim4_corrected": (
            "lineage_start.json",
            "numeric_complete.json",
            "numeric_verification.json",
            "profiles/patient_profiles_k32.parquet",
            "analysis/specificity_k32.json",
            "analysis/aim4_corrected_k32.json",
            "receipts/inputs.json",
            "receipts/review_packets.json",
            "receipts/source_snapshot.json",
            "source_snapshot/aim4_morphologic_atlas.py",
        ),
        "aim4_vocab_stability": (
            "completion_receipt.json",
            "verification_receipt.json",
            "results.json",
            "input_receipt.json",
            "development_patient_abundance.parquet",
            "association_effects.csv",
            "all_anchor_correspondence.csv",
            "source_snapshot/manifest.json",
            "source_snapshot/tools/aim4_vocab_stability.py",
        ),
    }
    for name, relatives in required.items():
        for relative in relatives:
            path = roots[name] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


def test_pathology_state_rejects_ambiguous_partial_finalization(tmp_path: Path) -> None:
    names = (
        "aim2_core",
        "e2c_offset",
        "aim2_e2d_refresh",
        "aim3_corrected",
        "aim3_repeated_controls",
        "aim4_corrected",
        "aim4_vocab_stability",
    )
    roots = {name: (tmp_path / name).resolve() for name in names}
    for root in roots.values():
        root.mkdir()
    _touch_required_precheck_markers(roots)
    (roots["aim4_corrected"] / "receipts" / "review_import.json").touch()
    with pytest.raises(verify.FullVerificationError, match="pathology-pending"):
        verify._precheck_roots(roots, pathology_state="pending")
    with pytest.raises(verify.FullVerificationError, match="final artifacts are missing"):
        verify._precheck_roots(roots, pathology_state="final")


def test_receipt_is_atomic_exclusive_and_outside_all_roots(tmp_path: Path) -> None:
    roots = tuple((tmp_path / f"root{index}").resolve() for index in range(7))
    for root in roots:
        root.mkdir()
    destination = (tmp_path / "full_verification.json").resolve()
    assert verify._receipt_destination(destination, roots) == destination
    verify._write_json_once_atomic(destination, {"status": "PASS"})
    with pytest.raises(FileExistsError, match="overwrite"):
        verify._write_json_once_atomic(destination, {"status": "changed"})
    assert json.loads(destination.read_text()) == {"status": "PASS"}
    with pytest.raises(verify.FullVerificationError, match="outside"):
        verify._receipt_destination(roots[0] / "bad.json", roots)


def _packet_fixture(root: Path) -> None:
    blocks = {}
    selections = {
        "base": [17],
        "attention_addendum": [28],
    }
    for slug, prototypes in selections.items():
        bundle = root / "review_bundles" / "k32" / slug
        packet = bundle / "packet"
        packet.mkdir(parents=True)
        key = bundle / "unblinding_key_DO_NOT_SHARE.csv"
        key.write_text(f"montage_id,prototype\nM01,{prototypes[0]}\n")
        blocks[slug] = {
            "bundle": str(bundle.resolve()),
            "packet": str(packet.resolve()),
            "key": verify._identity(key),
        }
    receipt = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "PASS",
        "selection": {
            "separate_keys": True,
            "tiles_per_montage": 12,
            "base_packet_prototypes": selections["base"],
            "attention_addendum_prototypes": selections["attention_addendum"],
        },
        "base": blocks["base"],
        "attention_addendum": blocks["attention_addendum"],
        "blinding": "prototype IDs occur only in external keys, never packet files",
    }
    _write_json(root / "receipts" / "review_packets.json", receipt)
    _write_json(
        root / "review_bundles" / "k32" / "_bundle_receipt_DO_NOT_SHARE.json",
        receipt,
    )


def _e4_fixture(tmp_path: Path) -> tuple[Path, dict]:
    root = (tmp_path / "e4").resolve()
    (root / "profiles").mkdir(parents=True)
    (root / "analysis").mkdir()
    (root / "source_snapshot").mkdir()
    (root / "profiles" / "patient_profiles_k32.parquet").write_bytes(b"profiles")
    (root / "source_snapshot" / "aim4_morphologic_atlas.py").write_text("# archived\n")
    source_identity = verify._identity(root / "source_snapshot" / "aim4_morphologic_atlas.py")
    _write_json(
        root / "receipts" / "source_snapshot.json",
        {
            "schema_version": 1,
            "component": "aim4_corrected_cap8192",
            "files": [
                {
                    "relative_path": "aim4_morphologic_atlas.py",
                    "source": {
                        "path": "/retired/aim4_morphologic_atlas.py",
                        "size_bytes": source_identity["size_bytes"],
                        "sha256": source_identity["sha256"],
                    },
                    "imported": source_identity,
                }
            ],
        },
    )
    _write_json(
        root / "receipts" / "inputs.json",
        {
            "schema_version": 1,
            "component": "aim4_corrected_cap8192",
            "input_e4_root": "/retired/e4",
            "aim2_root": "/not-bound-in-contract-fixture",
        },
    )
    _write_json(
        root / "lineage_start.json",
        {
            "schema_version": 1,
            "component": "aim4_corrected_cap8192",
            "status": "prepared",
            "output_root": str(root),
        },
    )
    structural = {
        "auc": None,
        "ci_low": None,
        "ci_high": None,
        "p": 1.0,
        "q": 1.0,
        "estimable": False,
        "significant": False,
    }
    protocol = {
        "cap": 8192,
        "k": 32,
        "seeds": [42, 43, 44],
        "n_bootstrap": 2000,
        "bh_family": "fixed prototypes 0..31, including structural p=1 rows",
    }
    _write_json(
        root / "analysis" / "specificity_k32.json",
        {
            "protocol": protocol,
            "prototypes": {
                str(prototype): {"abundance": {"A": dict(structural), "D": dict(structural)}}
                for prototype in range(32)
            },
        },
    )
    _write_json(
        root / "analysis" / "aim4_corrected_k32.json",
        {
            "status": "NUMERIC_ANALYSIS_COMPLETE_PATHOLOGY_REVIEW_PENDING",
            "protocol": protocol,
            "review_selection": {
                "base_packet": [17],
                "attention_addendum_packet": [28],
            },
            "claim_limits": [
                "Pathology descriptions remain pending explicit completed forms.",
                "Numeric effects remain conditional on the frozen k32 vocabulary.",
            ],
        },
    )
    _packet_fixture(root)
    completion_path = root / "numeric_complete.json"
    artifacts = _e4_manifest(
        root,
        tuple(sorted(verify._e4_manifest_paths(root, phase="numeric"))),
    )
    completion = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "numeric_completed_pathology_review_pending",
        "output_root": str(root),
        "artifacts": artifacts,
        "aggregate_sha256": hashlib.sha256(verify._canonical(artifacts).encode()).hexdigest(),
        "validation": {
            "status": "PASS",
            "pathology_review": "PENDING_EXPLICIT_COMPLETED_FORMS",
        },
        "pathology_status": ("fresh corrected blinded packets sealed; completed review pending"),
    }
    _write_json(completion_path, completion)
    verification = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "PASS",
        "receipt_role": "immutable_numeric_verification_addendum",
        "output_root": str(root),
        "numeric_completion": verify._identity(completion_path),
        "numeric_aggregate_sha256": completion["aggregate_sha256"],
        "checks": {
            "frozen_source_snapshot": "PASS",
            "numeric_completion_inventory_and_hashes": "PASS",
            "all_15_attention_files_and_tile_order": "PASS",
            "profile_and_candidate_exact_coverage": "PASS",
            "fixed_32_prototype_families": "PASS",
            "deterministic_statistical_replay": "PASS",
            "fresh_separate_pending_review_packets": "PASS",
        },
        "replay": {"status": "PASS"},
        "analysis": {"fixed_bh_family_size": 32},
    }
    _write_json(root / "numeric_verification.json", verification)
    return root, verification


def _reseal_e4_numeric(root: Path) -> dict:
    completion_path = root / "numeric_complete.json"
    completion = json.loads(completion_path.read_text())
    relatives = tuple(completion["artifacts"])
    artifacts = _e4_manifest(root, relatives)
    completion["artifacts"] = artifacts
    completion["aggregate_sha256"] = hashlib.sha256(
        verify._canonical(artifacts).encode()
    ).hexdigest()
    _write_json(completion_path, completion)
    receipt_path = root / "numeric_verification.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["numeric_completion"] = verify._identity(completion_path)
    receipt["numeric_aggregate_sha256"] = completion["aggregate_sha256"]
    _write_json(receipt_path, receipt)
    return receipt


def _seal_e4_final(root: Path) -> dict:
    bundle = root / "review_completion" / "k32"
    imported_forms = {}
    annotations = {}
    for slug, prototype, montage_id in (
        ("base", 17, "M01"),
        ("attention_addendum", 28, "A01"),
    ):
        imported_path = bundle / "submissions" / f"{slug}_completed.csv"
        imported_path.parent.mkdir(parents=True, exist_ok=True)
        imported_path.write_text(
            "montage_id,review_status,blinding_attestation\n"
            f"{montage_id},complete,confirmed_no_key_access\n"
        )
        imported = verify._identity(imported_path)
        imported_forms[slug] = {
            "source": {
                **imported,
                "path": str((root.parent / f"submitted-{slug}.csv").resolve()),
            },
            "imported": imported,
        }
        annotations[str(prototype)] = {
            "prototype": prototype,
            "packet": slug,
            "montage_id": montage_id,
            "assessment": {
                "montage_id": montage_id,
                "review_status": "complete",
                "blinding_attestation": "confirmed_no_key_access",
            },
        }
    pathology = {
        "schema_version": 1,
        "component": "aim4_corrected_pathology_review",
        "status": "complete",
        "structured_not_concatenated": True,
        "annotations": annotations,
        "submission_imports": imported_forms,
    }
    pathology_path = bundle / "pathology_review_k32.json"
    _write_json(pathology_path, pathology)
    numeric_report = json.loads((root / "analysis" / "aim4_corrected_k32.json").read_text())
    claim_limits = [
        value
        for value in numeric_report["claim_limits"]
        if not value.startswith("Pathology descriptions remain pending")
    ]
    claim_limits.append(
        "Pathology descriptions are the imported structured assessments from the fresh "
        "corrected blinded packets."
    )
    reviewed_path = bundle / "aim4_corrected_reviewed_k32.json"
    _write_json(
        reviewed_path,
        {
            **numeric_report,
            "status": "ANALYSIS_AND_PATHOLOGY_REVIEW_COMPLETE",
            "claim_limits": claim_limits,
            "pathology_review": pathology,
        },
    )
    review_receipt = {
        "schema_version": 1,
        "component": "aim4_corrected_pathology_review",
        "status": "PASS",
        "forms": imported_forms,
        "pathology": verify._identity(pathology_path),
        "reviewed_report": verify._identity(reviewed_path),
        "n_completed_prototypes": 2,
    }
    _write_json(bundle / "_import_receipt_DO_NOT_SHARE.json", review_receipt)
    _write_json(root / "receipts" / "review_import.json", review_receipt)
    excluded = {"lineage_complete.json", "verification.json"}
    relatives = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.relative_to(root).as_posix() not in excluded
    )
    artifacts = _e4_manifest(root, relatives)
    completion_path = root / "lineage_complete.json"
    completion = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "completed",
        "output_root": str(root),
        "artifacts": artifacts,
        "aggregate_sha256": hashlib.sha256(verify._canonical(artifacts).encode()).hexdigest(),
        "pathology_status": ("completed structured blinded pathology review imported and sealed"),
    }
    _write_json(completion_path, completion)
    numeric_completion = json.loads((root / "numeric_complete.json").read_text())
    receipt = {
        "schema_version": 1,
        "component": "aim4_corrected_cap8192",
        "status": "PASS",
        "receipt_role": "immutable_completion_verification_addendum",
        "output_root": str(root),
        "completion": verify._identity(completion_path),
        "completion_aggregate_sha256": completion["aggregate_sha256"],
        "checks": {
            "upstream_identities": "PASS",
            "completion_inventory_and_hashes": "PASS",
            "all_15_attention_files_and_tile_order": "PASS",
            "profile_and_candidate_exact_coverage": "PASS",
            "fixed_32_prototype_families": "PASS",
            "deterministic_statistical_replay": "PASS",
            "fresh_separate_review_packets": "PASS",
        },
        "numeric_seal": {
            "status": "PASS",
            "numeric_completion": verify._identity(root / "numeric_complete.json"),
            "numeric_verification": verify._identity(root / "numeric_verification.json"),
            "numeric_aggregate_sha256": numeric_completion["aggregate_sha256"],
        },
        "completed_pathology_review": {"status": "PASS"},
        "replay": {"status": "PASS"},
    }
    _write_json(root / "verification.json", receipt)
    return receipt


def _reseal_e4_final(root: Path) -> dict:
    completion_path = root / "lineage_complete.json"
    completion = json.loads(completion_path.read_text())
    artifacts = _e4_manifest(
        root,
        tuple(sorted(verify._e4_manifest_paths(root, phase="final"))),
    )
    completion["artifacts"] = artifacts
    completion["aggregate_sha256"] = hashlib.sha256(
        verify._canonical(artifacts).encode()
    ).hexdigest()
    _write_json(completion_path, completion)
    verification_path = root / "verification.json"
    receipt = json.loads(verification_path.read_text())
    receipt["completion"] = verify._identity(completion_path)
    receipt["completion_aggregate_sha256"] = completion["aggregate_sha256"]
    _write_json(verification_path, receipt)
    return receipt


def test_e4_pending_numeric_m32_and_packet_separation_pass(tmp_path: Path) -> None:
    root, replay = _e4_fixture(tmp_path)
    result = verify._verify_e4_contract(
        root,
        pathology_state="pending",
        numeric_replay=replay,
        final_replay=None,
    )
    assert result["status"] == "PASS"
    assert result["pathology"] == {
        "state": "PENDING",
        "numeric_results_valid": True,
        "pathology_claims_final": False,
    }
    assert result["packets"]["packets"]["base"]["prototypes"] == [17]
    assert result["packets"]["packets"]["attention_addendum"]["prototypes"] == [28]


def test_e4_fast_final_mode_requires_bound_full_replay_receipt(tmp_path: Path) -> None:
    root, numeric = _e4_fixture(tmp_path)
    final = _seal_e4_final(root)
    result = verify._verify_e4_contract(
        root,
        pathology_state="final",
        numeric_replay=numeric,
        final_replay=final,
    )
    assert result["pathology"] == {
        "state": "FINAL_COMPLETE",
        "numeric_results_valid": True,
        "pathology_claims_final": True,
    }
    completed = result["artifacts"]["completed_review_import"]
    assert completed["status"] == "PASS"
    assert completed["n_completed_prototypes"] == 2
    assert not (root / "analysis" / "pathology_review_k32.json").exists()


def test_fast_e4_pending_rejects_unsealed_extra_file(tmp_path: Path) -> None:
    root, replay = _e4_fixture(tmp_path)
    (root / "unsealed-extra.txt").write_text("not part of the numeric seal\n")
    with pytest.raises(verify.FullVerificationError, match="inventory is not exclusive"):
        verify._verify_e4_contract(
            root,
            pathology_state="pending",
            numeric_replay=replay,
            final_replay=None,
        )


def test_fast_e4_final_rejects_divergent_atomic_import_receipt(tmp_path: Path) -> None:
    root, numeric = _e4_fixture(tmp_path)
    _seal_e4_final(root)
    transaction_path = root / "review_completion" / "k32" / "_import_receipt_DO_NOT_SHARE.json"
    transaction = json.loads(transaction_path.read_text())
    transaction["n_completed_prototypes"] = 1
    _write_json(transaction_path, transaction)
    final = _reseal_e4_final(root)
    with pytest.raises(verify.FullVerificationError, match="atomic bundle"):
        verify._verify_e4_contract(
            root,
            pathology_state="final",
            numeric_replay=numeric,
            final_replay=final,
        )


def test_packet_key_inside_reviewer_packet_is_rejected(tmp_path: Path) -> None:
    root, _replay = _e4_fixture(tmp_path)
    receipt_path = root / "receipts" / "review_packets.json"
    receipt = json.loads(receipt_path.read_text())
    inside = root / "review_bundles" / "k32" / "base" / "packet" / "leaked_key.csv"
    inside.write_text("montage_id,prototype\nM01,17\n")
    receipt["base"]["key"] = verify._identity(inside)
    _write_json(receipt_path, receipt)
    _write_json(
        root / "review_bundles" / "k32" / "_bundle_receipt_DO_NOT_SHARE.json",
        receipt,
    )
    with pytest.raises(verify.FullVerificationError, match="key path mismatch"):
        verify._verify_packet_key_separation(root)


def test_packet_transaction_receipt_drift_is_rejected(tmp_path: Path) -> None:
    root, _replay = _e4_fixture(tmp_path)
    transaction_path = root / "review_bundles" / "k32" / "_bundle_receipt_DO_NOT_SHARE.json"
    transaction = json.loads(transaction_path.read_text())
    transaction["selection"]["tiles_per_montage"] = 11
    _write_json(transaction_path, transaction)
    with pytest.raises(verify.FullVerificationError, match="transaction receipt"):
        verify._verify_packet_key_separation(root)


def test_e4_pending_rejects_fake_structural_auc(tmp_path: Path) -> None:
    root, _replay = _e4_fixture(tmp_path)
    specificity_path = root / "analysis" / "specificity_k32.json"
    specificity = json.loads(specificity_path.read_text())
    specificity["prototypes"]["17"]["abundance"]["A"]["auc"] = 0.5
    _write_json(specificity_path, specificity)
    replay = _reseal_e4_numeric(root)
    with pytest.raises(verify.FullVerificationError, match="structural policy"):
        verify._verify_e4_contract(
            root,
            pathology_state="pending",
            numeric_replay=replay,
            final_replay=None,
        )


def test_fast_e4_rejects_result_tamper_without_bound_receipt(tmp_path: Path) -> None:
    root, replay = _e4_fixture(tmp_path)
    report_path = root / "analysis" / "aim4_corrected_k32.json"
    report = json.loads(report_path.read_text())
    report["status"] = "TAMPERED"
    _write_json(report_path, report)
    with pytest.raises(verify.FullVerificationError, match="artifact hash differs"):
        verify._verify_e4_contract(
            root,
            pathology_state="pending",
            numeric_replay=replay,
            final_replay=None,
        )


def test_fast_e4_rejects_partial_full_replay_receipt(tmp_path: Path) -> None:
    root, _replay = _e4_fixture(tmp_path)
    verification_path = root / "numeric_verification.json"
    receipt = json.loads(verification_path.read_text())
    receipt["checks"].pop("deterministic_statistical_replay")
    _write_json(verification_path, receipt)
    with pytest.raises(verify.FullVerificationError, match="full-replay PASS"):
        verify._verify_e4_contract(
            root,
            pathology_state="pending",
            numeric_replay=receipt,
            final_replay=None,
        )


def _effect_row(variant: str, anchor: int, *, passing: bool) -> dict:
    row = {
        "variant": variant,
        "anchor_prototype": anchor,
        "family_size_A": 32,
        "family_size_D": 32,
    }
    for population in ("A", "D"):
        row.update(
            {
                f"auc_{population}": 0.70,
                f"ci_low_{population}": 0.60,
                f"ci_high_{population}": 0.80,
                f"p_{population}": 0.001,
                f"q_{population}": 0.01,
                f"estimable_{population}": True,
                f"significant_{population}": True,
            }
        )
    if not passing:
        row.update(
            {
                "ci_low_D": 0.45,
                "q_D": 0.20,
                "significant_D": False,
            }
        )
    row["positive_significant_A_and_D"] = passing
    return row


def _stability_fixture(tmp_path: Path) -> tuple[Path, Path]:
    e4_root = (tmp_path / "e4-input").resolve()
    (e4_root / "profiles").mkdir(parents=True)
    (e4_root / "analysis").mkdir()
    for relative, payload in (
        ("numeric_complete.json", b"completion"),
        ("numeric_verification.json", b"verification"),
        ("profiles/patient_profiles_k32.parquet", b"profiles"),
        ("analysis/specificity_k32.json", b"specificity"),
    ):
        path = e4_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    root = (tmp_path / "stability").resolve()
    root.mkdir()
    source_tool = root / "source_snapshot" / "tools" / "aim4_vocab_stability.py"
    source_tool.parent.mkdir(parents=True)
    source_tool.write_text("# archived stability verifier\n")
    source_observed = verify._identity(source_tool)
    source_manifest = [
        {
            "path": "tools/aim4_vocab_stability.py",
            "size_bytes": source_observed["size_bytes"],
            "sha256": source_observed["sha256"],
        }
    ]
    _write_json(root / "source_snapshot" / "manifest.json", source_manifest)
    _write_json(root / "feature_inventory.json", {"files": []})
    for name in (
        "development_patient_abundance.parquet",
        "association_effects.csv",
        "all_anchor_correspondence.csv",
    ):
        (root / name).write_bytes(name.encode())
    identities = {
        "corrected_aim4_numeric_completion": verify._identity(e4_root / "numeric_complete.json"),
        "corrected_aim4_numeric_verification": verify._identity(
            e4_root / "numeric_verification.json"
        ),
        "canonical_patient_profiles": verify._identity(
            e4_root / "profiles" / "patient_profiles_k32.parquet"
        ),
        "canonical_specificity": verify._identity(e4_root / "analysis" / "specificity_k32.json"),
    }
    _write_json(
        root / "input_receipt.json",
        {
            "schema_version": verify.STABILITY_SCHEMA_VERSION,
            "corrected_aim4_root": str(e4_root),
            "input_identities": identities,
            "source_snapshot": source_manifest,
            "feature_inventory_sha256": verify._sha256_file(root / "feature_inventory.json"),
        },
    )
    variants = ("canonical_k32", *verify.STABILITY_VARIANTS)
    rows = []
    for variant in variants:
        for anchor in range(32):
            failing = anchor == 28 and variant == verify.STABILITY_VARIANTS[-1]
            rows.append(_effect_row(variant, anchor, passing=not failing))
    summaries = {
        "17": {
            "montage_id": "M04",
            "n_variants": 9,
            "passing_variants": list(verify.STABILITY_VARIANTS),
            "failing_variants": [],
            "all_variants_pass": True,
            "verdict": "ROBUST_ACROSS_PREDECLARED_K_SEED_GRID",
            "claim_limit": (
                "Matched alternative clusters remain pathology-unlabeled pending new blinded review."
            ),
        },
        "28": {
            "montage_id": "M07",
            "n_variants": 9,
            "passing_variants": list(verify.STABILITY_VARIANTS[:-1]),
            "failing_variants": [verify.STABILITY_VARIANTS[-1]],
            "all_variants_pass": False,
            "verdict": "CONDITIONAL_K32_NOT_STABLE_ACROSS_FULL_GRID",
            "claim_limit": (
                "Matched alternative clusters remain pathology-unlabeled pending new blinded review."
            ),
        },
    }
    results = {
        "schema_version": verify.STABILITY_SCHEMA_VERSION,
        "status": "COMPLETE",
        "scope": {
            "claim_limit": (
                "Alternative-cluster correspondences do not transfer M04/M07 pathology identities; "
                "new blinded review is required."
            )
        },
        "protocol": {
            "k_values": [24, 32, 40],
            "cluster_seeds": [20260819, 20260820, 20260821],
            "canonical_k": 32,
            "n_init": 10,
            "association_bootstrap_replicates": 2000,
            "association_fdr_alpha": 0.05,
            "association_family_size": 32,
            "inertia_contract": verify.STABILITY_INERTIA_CONTRACT,
            "canonical_reconstruction_control": {
                "sample_values_sha256": verify.STABILITY_CANONICAL_SAMPLE_SHA256,
                "maximum_cluster_mass_total_variation": (
                    verify.STABILITY_MAX_CLUSTER_MASS_TOTAL_VARIATION
                ),
                "maximum_single_cluster_mass_delta": (
                    verify.STABILITY_MAX_SINGLE_CLUSTER_MASS_DELTA
                ),
                "same_seed_refit_role": (
                    "sensitivity_variant_not_an_identity_control"
                ),
            },
        },
        "counts": {"sampled_tiles": 3200},
        "canonical_reconstruction_control": {
            "name": "exact_sample_and_frozen_centroid_reprojection",
            "sample_values_sha256_expected": verify.STABILITY_CANONICAL_SAMPLE_SHA256,
            "sample_values_sha256_observed": verify.STABILITY_CANONICAL_SAMPLE_SHA256,
            "sample_values_sha256_exact": True,
            "training_cluster_sizes": [100] * 32,
            "reprojected_cluster_sizes": [100] * 32,
            "absolute_cluster_size_differences": [0] * 32,
            "cluster_mass_total_variation": 0.0,
            "maximum_cluster_mass_delta": 0.0,
            "maximum_cluster_mass_total_variation_allowed": (
                verify.STABILITY_MAX_CLUSTER_MASS_TOTAL_VARIATION
            ),
            "maximum_single_cluster_mass_delta_allowed": (
                verify.STABILITY_MAX_SINGLE_CLUSTER_MASS_DELTA
            ),
            "same_seed_refit": {
                "variant": "k32_seed20260819",
                "ari_to_frozen_canonical": 0.75,
                "ami_to_frozen_canonical": 0.85,
                "role": "sensitivity_variant_not_an_identity_control",
                "included_in_all_nine_biological_gate": True,
            },
            "pass": True,
        },
        "canonical_patient_profile_reconstruction": {
            "pass": True,
            "maximum_absolute_abundance_difference": 0.0,
        },
        "downstream_abundance_claim_stability": {
            "canonical_effect_reconstruction": {
                "pass": True,
                "maximum_absolute_auc_difference": 0.0,
                "maximum_absolute_p_difference": 0.0,
                "maximum_absolute_q_difference": 0.0,
            },
            "association_effects": rows,
            "claim_stability": summaries,
        },
        "variant_metrics": [
            {
                "variant": variant,
                "k": int(variant.removeprefix("k").split("_seed", 1)[0]),
                "seed": int(variant.split("_seed", 1)[1]),
                "model_inertia": 100.0 + index,
                "replay_inertia_float64": 100.25 + index,
                "n_iter": index + 1,
            }
            for index, variant in enumerate(verify.STABILITY_VARIANTS)
        ],
    }
    _write_json(root / "results.json", results)
    completion = {
        "schema_version": verify.STABILITY_SCHEMA_VERSION,
        "status": "COMPLETE",
        "output_root": str(root),
        "payload_inventory": _payload_inventory(root),
        "results_sha256": verify._sha256_file(root / "results.json"),
        "input_receipt_sha256": verify._sha256_file(root / "input_receipt.json"),
    }
    _write_json(root / "completion_receipt.json", completion)
    _write_json(
        root / "verification_receipt.json",
        {
            "schema_version": verify.STABILITY_SCHEMA_VERSION,
            "status": "PASS",
            "output_root": str(root),
            "replay_input": True,
            "completion_sha256": verify._sha256_file(root / "completion_receipt.json"),
            "results_sha256": verify._sha256_file(root / "results.json"),
            "checks": {
                "exclusive_payload_inventory": "PASS",
                "input_hashes": "PASS",
                "source_snapshot": "PASS",
                "assignment_and_centroid_shapes": "PASS",
                "statistical_recomputation": "PASS",
                "exact_feature_and_assignment_replay": "PASS",
                "deterministic_inertia_replay": "PASS",
                "full_m32_A_D_family_recomputation": "PASS",
                "canonical_reconstruction_control": "PASS",
                "pathology_label_transfer": ("PROHIBITED_WITHOUT_NEW_BLINDED_REVIEW"),
            },
        },
    )
    return root, e4_root


def _reseal_stability(root: Path) -> dict:
    completion_path = root / "completion_receipt.json"
    completion = json.loads(completion_path.read_text())
    completion["payload_inventory"] = _payload_inventory(root)
    completion["results_sha256"] = verify._sha256_file(root / "results.json")
    completion["input_receipt_sha256"] = verify._sha256_file(root / "input_receipt.json")
    _write_json(completion_path, completion)
    verification_path = root / "verification_receipt.json"
    receipt = json.loads(verification_path.read_text())
    receipt["completion_sha256"] = verify._sha256_file(completion_path)
    receipt["results_sha256"] = verify._sha256_file(root / "results.json")
    _write_json(verification_path, receipt)
    return receipt


def test_stability_recomputes_p17_p28_all_nine_variant_gate(tmp_path: Path) -> None:
    root, e4_root = _stability_fixture(tmp_path)
    receipt = json.loads((root / "verification_receipt.json").read_text())
    checked = verify._verify_stability_contract(root, e4_root, delegated_replay=receipt)
    assert checked["claim_verdicts"] == {
        "17": "ROBUST_ACROSS_PREDECLARED_K_SEED_GRID",
        "28": "CONDITIONAL_K32_NOT_STABLE_ACROSS_FULL_GRID",
    }
    assert checked["inertia_contract"] == verify.STABILITY_INERTIA_CONTRACT


def test_stability_rejects_ambiguous_legacy_inertia_field(tmp_path: Path) -> None:
    root, e4_root = _stability_fixture(tmp_path)
    results_path = root / "results.json"
    results = json.loads(results_path.read_text())
    results["variant_metrics"][0]["inertia"] = results["variant_metrics"][0][
        "model_inertia"
    ]
    _write_json(results_path, results)
    verification = _reseal_stability(root)
    with pytest.raises(verify.FullVerificationError, match="inertia contract"):
        verify._verify_stability_contract(root, e4_root, delegated_replay=verification)


def test_stability_rejects_summary_that_overclaims_grid_robustness(tmp_path: Path) -> None:
    root, e4_root = _stability_fixture(tmp_path)
    results_path = root / "results.json"
    results = json.loads(results_path.read_text())
    results["downstream_abundance_claim_stability"]["claim_stability"]["28"].update(
        {
            "passing_variants": list(verify.STABILITY_VARIANTS),
            "failing_variants": [],
            "all_variants_pass": True,
            "verdict": "ROBUST_ACROSS_PREDECLARED_K_SEED_GRID",
        }
    )
    _write_json(results_path, results)
    verification = _reseal_stability(root)
    with pytest.raises(verify.FullVerificationError, match="claim-verdict algebra"):
        verify._verify_stability_contract(root, e4_root, delegated_replay=verification)


def test_fast_stability_rejects_tampered_input_receipt_hash(tmp_path: Path) -> None:
    root, e4_root = _stability_fixture(tmp_path)
    input_path = root / "input_receipt.json"
    inputs = json.loads(input_path.read_text())
    inputs["feature_inventory_sha256"] = "0" * 64
    _write_json(input_path, inputs)
    receipt = json.loads((root / "verification_receipt.json").read_text())
    with pytest.raises(verify.FullVerificationError, match="completion contract"):
        verify._verify_stability_contract(root, e4_root, delegated_replay=receipt)


def test_fast_stability_rejects_partial_full_replay_receipt(tmp_path: Path) -> None:
    root, e4_root = _stability_fixture(tmp_path)
    verification_path = root / "verification_receipt.json"
    receipt = json.loads(verification_path.read_text())
    receipt["checks"]["exact_feature_and_assignment_replay"] = "NOT_REQUESTED"
    _write_json(verification_path, receipt)
    with pytest.raises(verify.FullVerificationError, match="full-input replay PASS"):
        verify._verify_stability_contract(root, e4_root, delegated_replay=receipt)


def test_fast_stability_rejects_unsealed_extra_file(tmp_path: Path) -> None:
    root, e4_root = _stability_fixture(tmp_path)
    (root / "unsealed-extra.txt").write_text("not part of the completion inventory\n")
    receipt = json.loads((root / "verification_receipt.json").read_text())
    with pytest.raises(verify.FullVerificationError, match="unsealed extras"):
        verify._verify_stability_contract(root, e4_root, delegated_replay=receipt)


def test_fast_delegates_never_execute_archived_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _replay = _e4_fixture(tmp_path)
    aim2 = (tmp_path / "aim2").resolve()
    aim2.mkdir()
    _write_json(aim2 / "lineage_start.json", {"status": "started"})
    inputs_path = root / "receipts" / "inputs.json"
    inputs = json.loads(inputs_path.read_text())
    inputs["aim2_root"] = str(aim2)
    inputs["aim2_lineage_start"] = verify._identity(aim2 / "lineage_start.json")
    _write_json(inputs_path, inputs)
    sealed_numeric = _reseal_e4_numeric(root)

    stability_root, _stability_e4 = _stability_fixture(tmp_path)
    sealed_stability = json.loads((stability_root / "verification_receipt.json").read_text())

    def forbidden(*args: object, **kwargs: object) -> dict:
        raise AssertionError(f"archived replay executed in fast mode: {args}, {kwargs}")

    monkeypatch.setattr(verify, "_run_archived", forbidden)
    numeric, final, _e4_verifier = verify._delegate_e4(
        root,
        aim2,
        pathology_state="pending",
        execute_replay=False,
    )
    stability, _stability_verifier = verify._delegate_stability(
        stability_root,
        execute_replay=False,
    )
    assert numeric == sealed_numeric
    assert final is None
    assert stability == sealed_stability


def test_consolidated_fast_mode_selects_receipts_and_keeps_claim_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = [(tmp_path / f"root{index}").resolve() for index in range(7)]
    for root in roots:
        root.mkdir()
    calls: list[tuple[str, bool]] = []
    aim23_result = {
        "schema_version": 2,
        "status": "PASS",
        "checks": {
            "aim3_native_logit_and_headline_algebra": {
                "status": "PASS",
                "familywise_rungs_passing": ["codon"],
            },
            "aim3_repeated_control_component_and_headline_verifier": {
                "status": "PASS",
                "control_chains": 45,
                "control_folds": 225,
                "n_bootstrap": 20_000,
                "consensus_verdicts": {"codon": "CONSENSUS_CEILING"},
            },
        },
    }
    monkeypatch.setattr(verify, "_precheck_roots", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(verify.aim23, "verify", lambda **_kwargs: aim23_result)
    monkeypatch.setattr(
        verify,
        "_verify_aim23_root_budgets",
        lambda _roots: {"status": "PASS", "artifacts": {}},
    )

    def e4_delegate(*_args: object, **kwargs: object) -> tuple[dict, None, dict]:
        calls.append(("e4", kwargs["execute_replay"]))
        return {"status": "PASS"}, None, {"sha256": "e4"}

    def stability_delegate(*_args: object, **kwargs: object) -> tuple[dict, dict]:
        calls.append(("stability", kwargs["execute_replay"]))
        return {"status": "PASS"}, {"sha256": "stability"}

    monkeypatch.setattr(verify, "_delegate_e4", e4_delegate)
    monkeypatch.setattr(verify, "_delegate_stability", stability_delegate)
    monkeypatch.setattr(
        verify,
        "_verify_e4_contract",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "pathology": {
                "state": "PENDING",
                "numeric_results_valid": True,
                "pathology_claims_final": False,
            },
        },
    )
    monkeypatch.setattr(
        verify,
        "_verify_stability_contract",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "claim_verdicts": {
                "17": "ROBUST_ACROSS_PREDECLARED_K_SEED_GRID",
                "28": "CONDITIONAL_K32_NOT_STABLE_ACROSS_FULL_GRID",
            },
        },
    )
    result = verify.verify(
        aim2_core_root=roots[0],
        e2c_root=roots[1],
        aim2_refresh_root=roots[2],
        aim3_root=roots[3],
        aim3_repeated_root=roots[4],
        aim4_root=roots[5],
        aim4_stability_root=roots[6],
        aim4_pathology_state="pending",
        fast_sealed_receipts=True,
    )
    assert calls == [("e4", False), ("stability", False)]
    assert result["verification_mode"] == verify.FAST_RECEIPT_MODE
    assert result["fresh_aim4_replay_executed"] is False
    assert "aim4_numeric_full_replay_receipt" in result["component_results"]
    assert result["central_claims"]["aim4_p17_p28_abundance_stability"]["28"] == (
        "CONDITIONAL_K32_NOT_STABLE_ACROSS_FULL_GRID"
    )


def test_aim3_repeated_delegate_requires_exact_45_225_full_budget() -> None:
    result = {
        "schema_version": 2,
        "status": "PASS",
        "checks": {
            "aim3_native_logit_and_headline_algebra": {
                "status": "PASS",
                "familywise_rungs_passing": ["codon"],
            },
            "aim3_repeated_control_component_and_headline_verifier": {
                "status": "PASS",
                "control_chains": 45,
                "control_folds": 225,
                "n_bootstrap": 20_000,
                "consensus_verdicts": {"codon": "CONSENSUS_CEILING"},
            },
        },
    }
    assert verify._verify_aim23_delegate(result)["status"] == "PASS"
    result["checks"]["aim3_repeated_control_component_and_headline_verifier"]["control_folds"] = 224
    with pytest.raises(verify.FullVerificationError, match="45-chain/225-fold"):
        verify._verify_aim23_delegate(result)


def test_cli_requires_explicit_pathology_state() -> None:
    parser = verify._parser()
    arguments = []
    for flag in (
        "--aim2-core-root",
        "--e2c-root",
        "--aim2-refresh-root",
        "--aim3-root",
        "--aim3-repeated-root",
        "--aim4-root",
        "--aim4-stability-root",
    ):
        arguments.extend((flag, f"/{flag.removeprefix('--')}"))
    with pytest.raises(SystemExit):
        parser.parse_args(arguments)
    pending = parser.parse_args((*arguments, "--aim4-pathology-state", "pending"))
    assert pending.fast_sealed_receipts is False
    fast = parser.parse_args(
        (
            *arguments,
            "--aim4-pathology-state",
            "pending",
            "--fast-sealed-receipts",
        )
    )
    assert fast.fast_sealed_receipts is True
