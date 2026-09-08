"""Focused contracts for the lineage-aware corrected Aim-2 verifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import verify_aim2_corrected as verifier  # noqa: E402


def _metric(auc: float, ci: list[float]) -> dict:
    return {"auroc": auc, "auroc_ci": ci, "n_bootstrap": 10_000}


def _e2a_target(auc: float, ci: list[float]) -> dict:
    return {
        "seeds_complete": [42, 43, 44],
        "primary_overall": _metric(auc, ci),
        "source_calibrated": {
            "auroc_unchanged": True,
            "auroc_exact_logit_scale": auc,
        },
    }


def _minimal_e2a() -> tuple[dict, dict]:
    values = {
        "CPTAC": (0.70, [0.56, 0.82]),
        "RIH": (0.72, [0.61, 0.81]),
        "SurGen": (0.68, [0.60, 0.75]),
        "TCGA": (0.66, [0.57, 0.73]),
    }
    report = {
        "schema_version": 2,
        "inference": {"n_bootstrap": 10_000, "sampling_unit": "patient"},
        "targets": {name: _e2a_target(auc, ci) for name, (auc, ci) in values.items()},
        "input_refits": {name: {} for name in values},
        "diagnosis": {
            name: {
                "auroc": auc,
                "auroc_ci_low": ci[0],
                "ranking_transports": ci[0] > 0.5,
            }
            for name, (auc, ci) in values.items()
        },
    }
    matched_auc = 0.69
    matched = {
        "schema_version": 2,
        "inference": {"n_bootstrap": 10_000, "sampling_unit": "patient"},
        "targets": {"RIH_sm": _e2a_target(matched_auc, [0.58, 0.78])},
        "input_refits": {"RIH_sm": {}},
        "size_matched_sensitivity": {
            "n_bootstrap": 10_000,
            "auroc_left": values["RIH"][0],
            "auroc_right": matched_auc,
            "delta_auroc": matched_auc - values["RIH"][0],
            "ci_low": -0.09,
            "ci_high": 0.03,
            "standard_refits": {},
            "size_matched_refits": {},
        },
    }
    return report, matched


def _e2b_target(primary: float, metastatic: float) -> dict:
    delta = metastatic - primary
    return {
        "seeds_complete": [42, 43, 44],
        "metastatic_overall": {"auroc": metastatic, "n_bootstrap": 10_000},
        "primary_vs_metastatic": {
            "definition": "patients excluded from both arms",
            "primary_auroc": primary,
            "metastatic_auroc": metastatic,
            "delta_auroc": delta,
            "delta_auroc_ci": [-0.20, 0.05],
            "n_bootstrap": 10_000,
        },
    }


def _minimal_e2b() -> dict:
    targets = {"RIH": _e2b_target(0.72, 0.60), "SurGen": _e2b_target(0.64, 0.56)}
    deltas = [value["primary_vs_metastatic"]["delta_auroc"] for value in targets.values()]
    return {
        "inference": {"n_bootstrap": 10_000, "sampling_unit": "patient"},
        "targets": targets,
        "conclusion": {
            "n_bootstrap": 10_000,
            "metastatic_macro_auroc": 0.58,
            "metastatic_macro_ci": [0.48, 0.67],
            "claim_metastatic_transport": False,
            "both_point_estimates_above_0.5": True,
            "caveat": "Do not claim equivalence or a causal specimen-role effect.",
        },
        "combined_decrement": {
            "delta_auroc": sum(deltas) / len(deltas),
            "delta_auroc_ci": [-0.20, 0.01],
            "n_bootstrap": 10_000,
            "ci_excludes_zero": False,
            "evidence_of_overall_decrement": False,
            "inference": "overall decrement not established",
        },
        "classification": {"outcome": "overall decrement not established"},
    }


def _overlap_block(*, clipped: float, outside: float, ess: float, max_weight: float) -> dict:
    return {
        "diagnostics": {
            "propensity_by_subcohort": {
                "SR1482": {
                    "fraction_clipped": clipped,
                    "fraction_outside_empirical_common_support": outside,
                }
            },
            "SR1482_ess_fraction": ess,
            "SR1482_max_weight": max_weight,
        }
    }


def test_recursive_identity_audit_detects_tampering(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    recorded = verifier._artifact_identity(artifact)
    auditor = verifier.IdentityAuditor()
    auditor.scan({"deep": [{"artifact": recorded}]}, "fixture")
    assert auditor.occurrences == 1

    artifact.write_bytes(b"tampered")
    with pytest.raises(verifier.VerificationError, match="changed during verification"):
        auditor.assert_stable()


def test_live_source_identity_is_checked_against_frozen_snapshot(tmp_path):
    live = tmp_path / "repo"
    frozen = tmp_path / "lineage" / "source_snapshot"
    live.mkdir()
    frozen.mkdir(parents=True)
    (live / "analysis.py").write_text("new bytes")
    (frozen / "analysis.py").write_text("frozen bytes")
    recorded = verifier._artifact_identity(frozen / "analysis.py")
    recorded["path"] = str((live / "analysis.py").resolve())

    auditor = verifier.IdentityAuditor(
        live_source_root=live.resolve(), frozen_source_root=frozen.resolve()
    )
    observed = auditor.validate(recorded, "frozen source")

    assert observed["sha256"] == verifier._sha256_file(frozen / "analysis.py")
    assert observed["sha256"] != verifier._sha256_file(live / "analysis.py")


def test_documentation_exception_is_exactly_scoped(tmp_path):
    repo = tmp_path / "repo"
    protocol = repo / "reports" / "Experimental_Setup.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("current protocol")
    recorded = {
        "path": str(protocol.resolve()),
        "size_bytes": len(b"old protocol"),
        "sha256": verifier._sha256_bytes(b"old protocol"),
    }
    label = "core_results.e2d6.input_artifacts.mapping_sources.prespecified_protocol"

    strict = verifier.IdentityAuditor(live_source_root=repo.resolve())
    with pytest.raises(verifier.VerificationError, match="size mismatch"):
        strict.validate(recorded, label)

    allowed = verifier.IdentityAuditor(
        live_source_root=repo.resolve(), allow_known_documentation_exception=True
    )
    allowed.validate(recorded, label)
    assert len(allowed.documentation_exceptions) == 1
    assert allowed.documentation_exceptions[0]["recorded"] == recorded

    with pytest.raises(verifier.VerificationError, match="size mismatch"):
        allowed.validate(recorded, "core_results.e2d5.some_other_document")


def test_e2a_claim_state_is_derived_from_ci():
    report, matched = _minimal_e2a()
    result = verifier._validate_e2a(report, matched)
    assert set(result["ranking_transport"].values()) == {"above_null_established"}
    assert result["size_matched_minus_full"] == "not_established"

    report["diagnosis"]["CPTAC"]["ranking_transports"] = False
    with pytest.raises(verifier.VerificationError, match="transport claim/CI mismatch"):
        verifier._validate_e2a(report, matched)


def test_e2b_keeps_fixed_transport_and_decrement_rules_separate():
    report = _minimal_e2b()
    result = verifier._validate_e2b(report)
    assert result == {
        "fixed_metastatic_transport": "not_established",
        "overall_primary_to_metastatic_change": "not_established",
    }

    report["conclusion"]["claim_metastatic_transport"] = True
    with pytest.raises(verifier.VerificationError, match="fixed metastatic-transport claim"):
        verifier._validate_e2b(report)


def test_e2d1_rejects_a_replication_verdict_from_one_cohort():
    report = {
        "inference": {"n_bootstrap": 10_000},
        "cohorts": {
            "RIH": {
                "liver_vs_non_liver": {
                    "delta": -0.02,
                    "ci": [-0.30, 0.20],
                    "n_bootstrap": 10_000,
                }
            },
            "SurGen": {
                "liver_vs_non_liver": {
                    "delta": 0.40,
                    "ci": [0.10, 0.70],
                    "n_bootstrap": 10_000,
                }
            },
        },
        "concordance": {
            "same_sign": False,
            "both_cis_exclude_zero": False,
            "per_cohort_ci_excludes_zero": {"RIH": False, "SurGen": True},
            "verdict": "single-cohort finding",
            "point_sign_inferential_role": "none without confidence-interval support",
        },
    }
    result = verifier._validate_e2d1(report)
    assert result["organ_effect_replication"] == "single-cohort finding"

    report["concordance"]["verdict"] = "replicated"
    with pytest.raises(verifier.VerificationError, match="verdict mismatch"):
        verifier._validate_e2d1(report)


def test_overlap_gate_uses_all_four_documented_thresholds():
    adequate = verifier._derive_overlap_gate(
        _overlap_block(clipped=0.10, outside=0.20, ess=0.25, max_weight=10.0),
        "fixture",
    )
    assert adequate["estimable_for_inference"] is True
    assert adequate["failed_checks"] == []

    limited = verifier._derive_overlap_gate(
        _overlap_block(clipped=0.11, outside=0.20, ess=0.24, max_weight=10.1),
        "fixture",
    )
    assert limited["estimable_for_inference"] is False
    assert limited["failed_checks"] == ["clipping", "effective_sample_size", "maximum_weight"]
    assert "positivity diagnostic only" in limited["status"]


def test_e2d5_requires_small_panel_guardrails():
    report = {
        "population_audit": {"n_pairs": 3, "cohort_counts": {"RIH": 2, "TCGA": 1}},
        "paired_table": {"rows": [{}, {}, {}]},
        "headline_RIH_same_model": {
            "n_pairs": 2,
            "bootstrap": {
                "n_bootstrap_requested": 10_000,
                "intervals": {
                    "mean_logit_shift": {
                        "ci_95_percentile": [-1.0, 1.0],
                        "n_bootstrap_valid": 10_000,
                    }
                },
            },
        },
        "TCGA_single_pair": {"scope": "one pair; descriptive only", "row": {}},
        "guardrails": {
            "auroc_computed": False,
            "equivalence_claim_allowed": False,
            "noninferiority_claim_allowed": False,
            "causal_specimen_role_claim_allowed": False,
            "interpretation": "A CI crossing zero is not evidence of equivalence.",
        },
    }
    result = verifier._validate_e2d5(report)
    assert result["paired_logit_shift"] == "not_established"
    report["guardrails"]["equivalence_claim_allowed"] = True
    with pytest.raises(verifier.VerificationError, match="equivalence_claim_allowed"):
        verifier._validate_e2d5(report)


def test_e2c_superiority_language_requires_a_positive_ci():
    verifier._assert_no_unsupported_positive_claims(
        {"contrast": {"ci": [-0.1, 0.2], "inference": "improvement not established"}}
    )
    with pytest.raises(verifier.VerificationError, match="unsupported positive claim"):
        verifier._assert_no_unsupported_positive_claims(
            {"contrast": {"ci": [-0.1, 0.2], "inference": "S2 is superior"}}
        )
    verifier._assert_no_unsupported_positive_claims(
        {"contrast": {"ci": [0.01, 0.2], "inference": "S2 is superior"}}
    )


def test_atomic_receipt_writer_is_exclusive(tmp_path):
    destination = (tmp_path / "verification.json").resolve()
    verifier._write_json_once_atomic(destination, {"status": "PASS"})
    assert json.loads(destination.read_text()) == {"status": "PASS"}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        verifier._write_json_once_atomic(destination, {"status": "different"})
    assert json.loads(destination.read_text()) == {"status": "PASS"}


def test_receipt_destination_cannot_mutate_an_input_lineage(tmp_path):
    core = (tmp_path / "core").resolve()
    e2c = (tmp_path / "e2c").resolve()
    core.mkdir()
    e2c.mkdir()
    with pytest.raises(verifier.VerificationError, match="must not modify"):
        verifier._receipt_destination(
            core / "receipt.json", forbidden_roots=(core, e2c)
        )
    outside = (tmp_path / "verification" / "receipt.json").resolve()
    assert (
        verifier._receipt_destination(outside, forbidden_roots=(core, e2c))
        == outside
    )

def test_cap_scope_and_roots_fail_closed(tmp_path):
    core = tmp_path / "core"
    e2c = tmp_path / "e2c"
    core.mkdir()
    e2c.mkdir()
    with pytest.raises(verifier.VerificationError, match="scoped to cap=8192"):
        verifier.verify(core.resolve(), e2c.resolve(), cap=4096)
    with pytest.raises(verifier.VerificationError, match="explicit absolute"):
        verifier._absolute_root("relative", "fixture")
