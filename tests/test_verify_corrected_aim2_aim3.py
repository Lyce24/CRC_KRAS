from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import verify_corrected_aim2_aim3 as verify  # noqa: E402


def test_roots_must_be_distinct_and_non_nested(tmp_path: Path) -> None:
    roots = {"a": tmp_path / "a", "b": tmp_path / "b"}
    verify._validate_distinct_roots(roots)
    with pytest.raises(verify.ConsolidatedVerificationError, match="distinct"):
        verify._validate_distinct_roots({"a": roots["a"], "b": roots["a"]})
    with pytest.raises(verify.ConsolidatedVerificationError, match="nested"):
        verify._validate_distinct_roots({"a": roots["a"], "b": roots["a"] / "child"})


def test_roundtrip_scanner_rejects_structural_marker() -> None:
    findings = verify._walk_forbidden_roundtrip(
        {"method": {"probability_to_logit": True}}
    )
    assert findings == ["root.method.probability_to_logit"]


def test_roundtrip_scanner_does_not_reject_a_historical_warning() -> None:
    assert verify._walk_forbidden_roundtrip(
        {"note": "probability round trips are forbidden; native logits are used"}
    ) == []


def _aim3_report() -> dict:
    tasks = {}
    for task in verify.aim3_corrected.ALL_TASKS:
        if task == "gene":
            auc, ci = 0.70, [0.65, 0.75]
        elif task.startswith("ctrl_"):
            auc, ci = 0.70, [0.62, 0.77]
        else:
            auc, ci = 0.55, [0.52, 0.58]
        tasks[task] = {"auroc": auc, "ci95": ci}
    pairs = {}
    for fine, control in verify.aim3_corrected.PAIRS:
        pairs[fine] = {
            "control": control,
            "primary_delta_control_minus_fine": {"estimate": 0.15, "ci95": [0.08, 0.22]},
            "nominal_95_verdict": "CEILING_WITH_RESIDUAL_SIGNAL",
            "familywise_ceiling_sensitivity": {
                "fine_99_upper": 0.59,
                "control_99_lower": 0.60,
                "delta_99_lower": 0.05,
                "pass": True,
                "status": "CEILING_FWER_PASS",
            },
            "subcohort_standardized_control_sensitivity": {
                "control_auroc": 0.68,
                "control_ci95": [0.60, 0.75],
                "delta_control_minus_fine": 0.13,
                "delta_ci95": [0.05, 0.21],
            },
        }
    return {
        "component": "aim3_corrected_cap8192",
        "protocol": {
            "cap": 8192,
            "seeds": [42, 43, 44],
            "n_bootstrap": 10_000,
            "bootstrap_seed": 20260817,
            "patient_aggregation": "mean native slide logit, then mean across exactly three seeds",
            "pair_contrast": "cohort-stratified partially paired patient bootstrap",
            "task_ci": "patient bootstrap stratified by cohort x label",
            "familywise_sensitivity": {
                "rungs": 5,
                "bonferroni_one_sided_alpha_per_rung": 0.01,
            },
        },
        "tasks": tasks,
        "pairs": pairs,
        "familywise_summary": {
            "rungs_passing_ceiling_gate": list(verify.aim3_corrected.FINE_TASKS),
            "n_passing": 5,
        },
    }


def test_aim3_headline_verifier_recomputes_verdict_and_familywise_gate(tmp_path: Path) -> None:
    result = tmp_path / "analysis" / "aim3_corrected.json"
    result.parent.mkdir()
    result.write_text(json.dumps(_aim3_report()))
    checked = verify._verify_aim3_headlines(tmp_path)
    assert checked["status"] == "PASS"
    assert checked["familywise_rungs_passing"] == list(verify.aim3_corrected.FINE_TASKS)


def test_aim3_headline_verifier_rejects_wrong_verdict(tmp_path: Path) -> None:
    report = _aim3_report()
    report["pairs"]["codon"]["nominal_95_verdict"] = "INCONCLUSIVE"
    result = tmp_path / "analysis" / "aim3_corrected.json"
    result.parent.mkdir()
    result.write_text(json.dumps(report))
    with pytest.raises(verify.ConsolidatedVerificationError, match="verdict logic"):
        verify._verify_aim3_headlines(tmp_path)


def test_aim3_headline_verifier_rejects_wrong_familywise_boolean(tmp_path: Path) -> None:
    report = _aim3_report()
    report["pairs"]["codon"]["familywise_ceiling_sensitivity"]["pass"] = False
    result = tmp_path / "analysis" / "aim3_corrected.json"
    result.parent.mkdir()
    result.write_text(json.dumps(report))
    with pytest.raises(verify.ConsolidatedVerificationError, match="familywise gate"):
        verify._verify_aim3_headlines(tmp_path)


def _e2c_report(core: Path) -> dict:
    arms = {}
    points = {}
    for arm in ("S1", "S2"):
        for k in (2, 4, 8):
            name = f"{arm}_k{k}"
            point = 0.61 + (0.01 if arm == "S2" else 0.0)
            points[name] = point
            arms[name] = {
                "performance": {
                    "n_procedure_draws": 100,
                    "metrics": {"auroc": point},
                    "per_draw_metrics": [{"auroc": point}] * 100,
                    "expected_auroc_ci": [point - 0.1, point + 0.1],
                }
            }
    contrasts = {}
    for k in (2, 4, 8):
        contrasts[f"S2_minus_S1_k{k}"] = {"delta": 0.01, "ci": [-0.01, 0.03]}
        for arm in ("S1", "S2"):
            name = f"{arm}_k{k}"
            contrasts[f"{name}_minus_S0"] = {
                "delta": points[name] - 0.60,
                "ci": [-0.02, 0.04],
            }
    cohort = {
        "reps": 100,
        "folds": 5,
        "S0": {"auroc": 0.60},
        "arms": arms,
        "contrasts": contrasts,
    }
    return {
        "component": "e2c_native_logit_residual_offset",
        "cap": 8192,
        "reps": 100,
        "n_bootstrap": 10_000,
        "input_lineage_root": str(core),
        "design": {
            "model": "eta_native + H @ delta_w + delta_b",
            "frozen": "encoder, head, embeddings, and native logits",
            "n_bootstrap": 10_000,
            "bootstrap_seed": 20260817,
        },
        "cohorts": {"RIH": cohort, "SurGen": cohort},
    }


def test_e2c_contract_recomputes_all_point_contrasts(tmp_path: Path) -> None:
    core = tmp_path / "core"
    core.mkdir()
    e2c = tmp_path / "e2c"
    result = e2c / "analysis" / "e2c_native_logit_offset_cap8192.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(_e2c_report(core.resolve())))
    checked = verify._verify_e2c_native_contract(e2c, core.resolve())
    assert checked["status"] == "PASS"
    assert len(checked["headline_contrast_states"]) == 18


def test_e2c_contract_rejects_probability_roundtrip_marker(tmp_path: Path) -> None:
    core = tmp_path / "core"
    core.mkdir()
    e2c = tmp_path / "e2c"
    result = e2c / "analysis" / "e2c_native_logit_offset_cap8192.json"
    result.parent.mkdir(parents=True)
    report = _e2c_report(core.resolve())
    report["design"]["probability_to_logit"] = True
    result.write_text(json.dumps(report))
    with pytest.raises(verify.ConsolidatedVerificationError, match="roundtrip"):
        verify._verify_e2c_native_contract(e2c, core.resolve())


def test_receipt_is_exclusive_and_cannot_be_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "receipt.json"
    assert verify._receipt_destination(outside.resolve(), (root.resolve(),)) == outside.resolve()
    with pytest.raises(verify.ConsolidatedVerificationError, match="outside"):
        verify._receipt_destination((root / "receipt.json").resolve(), (root.resolve(),))
    verify._write_json_once_atomic(outside, {"status": "PASS"})
    with pytest.raises(FileExistsError):
        verify._write_json_once_atomic(outside, {"status": "changed"})
    assert json.loads(outside.read_text()) == {"status": "PASS"}


def _repeated_stats(point: float, bounds: list[float]) -> dict:
    return {
        "auroc": point,
        "ci95_two_sided": [point - 0.05, point + 0.05],
        "primary_fwer_one_sided": {
            "lower": bounds[0],
            "upper": bounds[1],
            "one_sided_confidence": 0.99,
        },
        "ultra_conservative_15_comparison_two_sided": [point - 0.1, point + 0.1],
    }


def _build_repeated_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = (tmp_path / "repeated").resolve()
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    fine_root = (tmp_path / "e3a").resolve()
    fine_root.mkdir()
    start = {
        "schema_version": 2,
        "status": "prepared",
        "output_root": str(root),
        "fine_input_root": str(fine_root),
        "protocol": {
            "cap": 8192,
            "model_seeds": [42, 43, 44],
            "wt_draw_seeds": [20260823, 20260824, 20260825],
            "n_control_chains": 45,
            "n_control_folds": 225,
            "fine_folds_reused": 75,
            "bootstrap_seed": 20260826,
            "n_bootstrap": 20_000,
            "primary_fwer": (
                "five rung-level IUTs; Bonferroni alpha=.05/5; "
                "one-sided 99% bounds; three-draw consensus"
            ),
        },
        "packed_store": {"fingerprint_sha256": "packed"},
    }
    (root / "lineage_start.json").write_text(json.dumps(start))

    rungs = {}
    for fine in verify.REPEATED_FINE_TASKS:
        fine_result = _repeated_stats(0.55, [0.51, 0.59])
        draws = {}
        decisions = []
        for draw_seed in verify.REPEATED_DRAW_SEEDS:
            control = _repeated_stats(0.70, [0.60, 0.80])
            delta = _repeated_stats(0.15, [0.01, 0.30])
            delta["estimate"] = delta.pop("auroc")
            verdict = "CEILING_WITH_RESIDUAL_SIGNAL"
            decisions.append(verdict)
            draws[str(draw_seed)] = {
                "bootstrap_seed": 12345,
                "control_per_model_seed_auroc": {
                    "42": 0.69,
                    "43": 0.70,
                    "44": 0.71,
                },
                "fine": fine_result,
                "control": control,
                "delta_control_minus_fine": delta,
                "primary_conditions": {
                    "fine_upper_99_lt_0.60": True,
                    "control_lower_99_gt_0.50": True,
                    "delta_lower_99_gt_0": True,
                    "fine_lower_99_gt_0.50": True,
                },
                "primary_verdict": verdict,
            }
        rungs[fine] = {
            "control": verify.REPEATED_CONTROLS[fine],
            "fine_per_model_seed_auroc": {"42": 0.54, "43": 0.55, "44": 0.56},
            "draws": draws,
            "draw_verdicts": decisions,
            "consensus_verdict": "CONSENSUS_CEILING",
        }
    report = {
        "schema_version": 2,
        "status": "complete",
        "output_root": str(root),
        "protocol": {
            "native_score": "mean slide native logit per patient; mean across seeds 42/43/44",
            "bootstrap": "partial-paired patient bootstrap, subcohort x frozen k_fold stratified",
            "shared_component": "positive patient IDs and resampling indices shared across arms",
            "distinct_component": "fine/control negatives independently resampled within matched strata",
            "n_bootstrap": 20_000,
            "bootstrap_seed_base": 20260826,
            "primary_fwer": {
                "family": "five rung-level intersection-union ceiling tests",
                "alpha": 0.05,
                "per_rung_one_sided_alpha": 0.01,
                "bounds": "one-sided 99%",
                "within_rung_penalty": "none: components and three-draw consensus are intersections",
            },
            "sensitivity_only": "central 99.6667% intervals over 15 draw-by-rung comparisons",
            "ceiling_consensus": (
                "all three predeclared WT draws must meet the primary FWER ceiling rule"
            ),
        },
        "accounting": {
            "fine_folds_reused": 75,
            "control_chains": 45,
            "control_folds": 225,
        },
        "rungs": rungs,
    }
    report_path = analysis / "aim3_repeated_control_report.json"
    report_path.write_text(json.dumps(report))
    distribution_path = analysis / "bootstrap_distributions.npz"
    distribution_path.write_bytes(b"fixture-bootstrap-arrays")

    chains = [
        {
            "control": control,
            "draw_seed": draw_seed,
            "model_seed": model_seed,
            "fold_completions": [{"fold": fold} for fold in range(5)],
        }
        for control in verify.REPEATED_CONTROLS.values()
        for draw_seed in verify.REPEATED_DRAW_SEEDS
        for model_seed in verify.REPEATED_MODEL_SEEDS
    ]
    audit = {
        "schema_version": 2,
        "status": "PASS",
        "lineage_start": verify._identity(root / "lineage_start.json"),
        "packed_store_fingerprint_sha256": "packed",
        "control_chains": chains,
        "report": verify._identity(report_path),
        "bootstrap_distributions": verify._identity(distribution_path),
    }
    audit_path = analysis / "analysis_audit.json"
    audit_path.write_text(json.dumps(audit))
    complete = {
        "schema_version": 2,
        "status": "completed",
        "output_root": str(root),
        "artifacts": {
            "aim3_repeated_control_report.json": verify._identity(report_path),
            "bootstrap_distributions.npz": verify._identity(distribution_path),
            "analysis_audit.json": verify._identity(audit_path),
        },
    }
    (root / "lineage_complete.json").write_text(json.dumps(complete))
    return root, fine_root


def test_repeated_control_fixture_is_verified_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, fine_root = _build_repeated_fixture(tmp_path)
    delegated: list[str] = []

    def fake_component_verify(args: object) -> None:
        delegated.append(str(args.output_root))  # type: ignore[attr-defined]
        print(
            "PASS — immutable Aim-3 campaign verified: 75 reused fine folds, "
            "225 control folds, five FWER-corrected consensus decisions"
        )

    monkeypatch.setattr(
        verify.e3_controls_repeated, "cmd_verify_output", fake_component_verify
    )
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    checked = verify._verify_aim3_repeated(root, expected_fine_root=fine_root)
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert checked["status"] == "PASS"
    assert checked["control_chains"] == 45
    assert checked["control_folds"] == 225
    assert checked["n_bootstrap"] == 20_000
    assert set(checked["consensus_verdicts"].values()) == {"CONSENSUS_CEILING"}
    assert delegated == [str(root)]
    assert after == before


def test_repeated_control_partial_root_fails_before_delegation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "partial"
    root.mkdir()
    fine_root = tmp_path / "e3a"
    fine_root.mkdir()

    def forbidden_delegate(args: object) -> None:
        raise AssertionError(f"delegate called for partial root: {args}")

    monkeypatch.setattr(
        verify.e3_controls_repeated, "cmd_verify_output", forbidden_delegate
    )
    with pytest.raises(verify.ConsolidatedVerificationError, match="partial/incomplete"):
        verify._verify_aim3_repeated(root.resolve(), expected_fine_root=fine_root.resolve())
    assert list(root.iterdir()) == []


def test_final_grade_cli_requires_repeated_control_root() -> None:
    parser = verify._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--aim2-core-root",
                "/a",
                "--e2c-root",
                "/b",
                "--aim2-refresh-root",
                "/c",
                "--aim3-root",
                "/d",
            ]
        )


def test_refresh_binding_uses_frozen_source_and_protocol_context(
    tmp_path: Path,
) -> None:
    core = (tmp_path / "core").resolve()
    core.mkdir()
    refresh = (tmp_path / "refresh").resolve()
    (refresh / "source_snapshot").mkdir(parents=True)
    (refresh / "context_snapshot" / "reports").mkdir(parents=True)
    (refresh / "eval").mkdir()

    source = refresh / "source_snapshot" / "analysis.py"
    source.write_text("frozen source\n")
    protocol = refresh / "context_snapshot" / "reports" / "Experimental_Setup.md"
    protocol.write_text("frozen protocol\n")
    archive = (
        refresh
        / "context_snapshot"
        / "reports"
        / "Experimental_Setup_DESIGN_ARCHIVE.md"
    )
    archive.write_text("frozen archive\n")
    external = tmp_path / "manifest.csv"
    external.write_text("slide_id,label\na,1\n")
    live_protocol = tmp_path / "live_protocol.md"
    live_protocol.write_text("later edited protocol\n")

    for name in verify.aim2_exploratory_refresh.EXPECTED_RESULTS[:3]:
        path = refresh / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"native_logit": True}))

    verification_path = refresh / verify.aim2_exploratory_refresh.VERIFICATION_NAME
    verification_path.write_text(json.dumps({"status": "pass"}))
    receipt_path = refresh / verify.aim2_exploratory_refresh.RECEIPT_NAME
    receipt = {
        "input_root": str(core),
        "cap": 8192,
        "n_bootstrap": 10_000,
        "bootstrap_seed": 20260817,
        "analysis_grade": "full_10000_bootstrap",
        "input_inventory": [
            {"label": "current_protocol", "path": str(live_protocol), **{
                key: value for key, value in verify._identity(protocol).items() if key != "path"
            }},
            {"label": "historical_design_archive", "path": str(live_protocol), **{
                key: value for key, value in verify._identity(archive).items() if key != "path"
            }},
            {"label": "manifest", **verify._identity(external)},
        ],
        "live_source_inventory": [
            {"label": "analysis.py", "path": str(tmp_path / "changed.py"), **{
                key: value for key, value in verify._identity(source).items() if key != "path"
            }}
        ],
    }
    receipt_path.write_text(json.dumps(receipt))

    checked = verify._verify_refresh_binding(refresh, core)
    assert checked["status"] == "PASS"
    assert checked["archival_provenance"] == {
        "external_material_inputs_live_checked": 1,
        "frozen_source_files_checked": 1,
        "frozen_documentation_contexts_checked": 2,
        "later_live_markdown_edits_are_not_analysis_inputs": True,
    }
