from __future__ import annotations

import json
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_loco_five_seed_extension as extension


def test_official_arm_and_seed_rosters_are_exact() -> None:
    assert extension.ADOPTED_SEEDS == (42, 43, 44)
    assert extension.NEW_SEEDS == (45, 46)
    assert extension.ALL_SEEDS == (42, 43, 44, 45, 46)
    assert len(extension.ARMS) == 9
    assert len(extension.CONTROLLING_ARMS) == 8
    assert "family_rih_sm" in extension.ARMS
    assert "family_rih_sm" not in extension.CONTROLLING_ARMS


def test_every_arm_pins_one_manifest_and_one_split_for_all_seeds() -> None:
    for spec in extension.ARMS.values():
        assert len(spec.source_sha256) == 64
        assert len(spec.split_sha256) == 64
        assert spec.source_manifest.name.endswith(".csv")
        assert spec.split_file.name == "splits.parquet"
        assert spec.expected_source_patients > 0
        assert spec.expected_source_slides >= spec.expected_source_patients


def test_training_inventory_is_six_way_compatible_and_exact(tmp_path: Path) -> None:
    jobs = extension.build_training_jobs(tmp_path / "five_seed_overlay")
    assert len(jobs) == 36
    assert sum(job["fit_count"] for job in jobs) == 108
    assert {job["stage"] for job in jobs} == {"source_cv", "refit"}
    assert {job["seed"] for job in jobs} == {45, 46}
    assert {job["arm"] for job in jobs} == set(extension.ARMS)
    assert all(job["depends_on"] == ["aim2.manifest_contract"] for job in jobs)
    assert all("aim2_loco" in job["output"] for job in jobs)
    assert all(job["command"][2] in {"_source-cv", "_refit"} for job in jobs)
    assert all(job["external_worker_command"][2] == "train-one" for job in jobs)
    assert all("--external-scheduler" in job["external_worker_command"] for job in jobs)
    assert all("--recover-orphan" in job["external_worker_command"] for job in jobs)
    assert all("--apply" in job["external_worker_command"] for job in jobs)
    assert extension.DEFAULT_MAX_WORKERS == 6


def test_downstream_inventory_is_complete_and_never_scores_rih_sm_beyond_primary(
    tmp_path: Path,
) -> None:
    jobs = extension.build_score_jobs(tmp_path / "five_seed_overlay")
    assert len(jobs) == 66
    assert sum(job["target"] == "primary" for job in jobs) == 18
    assert sum(job["target"] in extension.MET_TARGETS for job in jobs) == 32
    assert sum(job["target"] == "orion" for job in jobs) == 16
    assert all(job["contains_target_outcomes"] is False for job in jobs)
    rih_sm = [job for job in jobs if job["arm"] == "family_rih_sm"]
    assert {job["target"] for job in rih_sm} == {"primary"}


def test_label_blinding_is_allowlist_based_and_drops_outcomes() -> None:
    source = pd.DataFrame(
        {
            "slide_id": ["s2", "s1"],
            "patient_id": ["p2", "p1"],
            "cohort": ["x", "x"],
            "target_label": [1, 0],
            "kras_status": ["mut", "wt"],
            "braf": [0, 1],
            "free_text": ["secret", "secret"],
        }
    )
    blind = extension._blind_frame(source, context="test")  # noqa: SLF001
    assert list(blind.columns) == ["slide_id", "patient_id", "cohort"]
    assert blind["slide_id"].tolist() == ["s1", "s2"]
    assert not (set(map(str.lower, blind.columns)) & extension.FORBIDDEN_OUTCOME_COLUMNS)


def test_label_blinding_rejects_duplicate_slide_ids() -> None:
    source = pd.DataFrame({"slide_id": ["s1", "s1"], "patient_id": ["p1", "p1"]})
    with pytest.raises(extension.ContractError, match="duplicated"):
        extension._blind_frame(source, context="test")  # noqa: SLF001


def test_extension_resolver_never_selects_sealed_roots(tmp_path: Path) -> None:
    root = tmp_path / "new_overlay"
    assert extension.component_root(root) == root.resolve() / "aim2_loco"
    assert extension.assert_safe_output_root(root) == root.resolve() / "aim2_loco"
    with pytest.raises(extension.ContractError, match="overlaps sealed"):
        extension.assert_safe_output_root(extension.FAMILY_ROOT)


def test_output_root_is_exact_production_or_non_symlink_tmp(tmp_path: Path) -> None:
    assert extension.assert_safe_output_root(extension.DEFAULT_OUTPUT_ROOT) == (
        extension.DEFAULT_OUTPUT_ROOT.resolve() / "aim2_loco"
    )
    with pytest.raises(extension.ContractError, match="absolute"):
        extension.assert_safe_output_root(Path("relative/output"))
    with pytest.raises(extension.ContractError, match="exactly"):
        extension.assert_safe_output_root(Path("/home/yc_liu/alternate_aim2_output"))
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(extension.ContractError, match="symlink"):
        extension.assert_safe_output_root(alias)


def test_checkpoint_resolver_adopts_old_and_routes_new_to_overlay(tmp_path: Path) -> None:
    root = tmp_path / "new_overlay"
    old = extension.checkpoint_path(root, "family_cptac", 42)
    new = extension.checkpoint_path(root, "family_cptac", 45)
    assert old == extension.FAMILY_ROOT / "train/pb_cap8192/cptac/seed42/final/refit/model.ckpt"
    assert new == root.resolve() / "aim2_loco/train/refit/pb_cap8192/family_cptac/seed45/final/refit/model.ckpt"
    with pytest.raises(ValueError, match="Uncontracted seed"):
        extension.checkpoint_path(root, "family_cptac", 47)


def test_job_inventory_json_contract_counts(tmp_path: Path) -> None:
    jobs = extension.build_training_jobs(tmp_path / "overlay")
    payload = {
        "default_max_workers": extension.DEFAULT_MAX_WORKERS,
        "fit_count": sum(job["fit_count"] for job in jobs),
        "jobs": jobs,
    }
    round_trip = json.loads(json.dumps(payload))
    assert round_trip["default_max_workers"] == 6
    assert round_trip["fit_count"] == 108
    assert len({job["job_id"] for job in round_trip["jobs"]}) == 36


def test_cli_defaults_to_six_workers_and_requires_apply_to_launch() -> None:
    args = extension.build_parser().parse_args(["train", "--dry-run"])
    assert args.max_workers == 6
    assert args.dry_run is True
    assert args.apply is False


def test_public_parser_exposes_complete_lifecycle() -> None:
    parser = extension.build_parser()
    for command in (
        "plan",
        "manifest",
        "preflight",
        "train",
        "train-one",
        "quarantine-failed",
        "score",
        "seal-inference",
        "report",
        "verify",
    ):
        extra = ["--dry-run"] if command == "train" else []
        if command == "train-one":
            extra = ["--job-key", "aim2.refit.family_cptac.seed45"]
        if command == "quarantine-failed":
            extra = ["--job-key", "aim2.refit.family_cptac.seed45"]
        args = parser.parse_args([command, *extra])
        assert callable(args.func)


def test_native_logit_patient_ensemble_averages_seeds_then_slides() -> None:
    manifest = pd.DataFrame(
        {
            "slide_id": ["s1", "s2", "s3"],
            "patient_id": ["p1", "p1", "p2"],
            "subcohort": ["A", "A", "A"],
        }
    )
    outcomes = pd.DataFrame({"patient_id": ["p1", "p2"], "target_label": [0, 1]})
    frames = []
    for seed, offset in zip(extension.ALL_SEEDS, (0.0, 1.0, 2.0, 3.0, 4.0), strict=True):
        frames.append(
            pd.DataFrame(
                {
                    "slide_id": ["s1", "s2", "s3"],
                    "seed": seed,
                    "fold": 0,
                    "logit": [offset, offset + 2.0, offset + 10.0],
                }
            )
        )
    patients = extension._patient_from_score_frames(  # noqa: SLF001
        frames, manifest, outcomes, expected_seeds=extension.ALL_SEEDS
    )
    # p1: seed-mean slide logits 2 and 4, then slide mean 3. p2: 12.
    assert patients.set_index("patient_id")["mean_logit"].to_dict() == {"p1": 3.0, "p2": 12.0}


def test_five_seed_calibrator_is_native_logit_and_source_only() -> None:
    source = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(8)],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "mean_logit": [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
        }
    )
    calibrator = extension._fit_five_seed_calibrator(source, "test_arm")  # noqa: SLF001
    assert calibrator["seeds"] == [42, 43, 44, 45, 46]
    assert calibrator["target_outcomes_used"] is False
    assert calibrator["b"] > 0
    assert "native OOF logits" in calibrator["method"]


def test_report_never_opens_outcomes_before_inference_seal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened = False

    def fail_seal(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise extension.ContractError("seal missing")

    def mark_open(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal opened
        opened = True
        return pd.DataFrame()

    monkeypatch.setattr(extension, "verify_inference_seal", fail_seal)
    monkeypatch.setattr(extension, "_outcome_frame_for_target", mark_open)
    with pytest.raises(extension.ContractError, match="seal missing"):
        extension.cmd_report(
            Namespace(
                output_root=tmp_path / "overlay",
                apply=True,
                n_bootstrap=10,
                bootstrap_seed=1,
            )
        )
    assert opened is False


def test_train_one_requires_explicit_external_scheduler_authority(tmp_path: Path) -> None:
    with pytest.raises(extension.ContractError, match="external scheduler"):
        extension.cmd_train_one(
            Namespace(
                output_root=tmp_path / "overlay",
                job_key="aim2.refit.family_cptac.seed45",
                external_scheduler=False,
                apply=True,
            )
        )


def _binary_frame(prefix: str, *, subcohort: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "patient_id": [f"{prefix}{index}" for index in range(12)],
            "label": [0] * 6 + [1] * 6,
            "mean_logit": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.3, -0.2, 0.4, 0.8, 1.2, 1.6, 2.0],
        }
    )
    if subcohort is not None:
        frame["subcohort"] = subcohort
    return frame


def test_family_macro_uses_pooled_tcga_and_frozen_target_bootstrap() -> None:
    surgen = pd.concat(
        [
            _binary_frame("a", subcohort="SR386"),
            _binary_frame("b", subcohort="SR1482"),
        ],
        ignore_index=True,
    )
    tcga = pd.concat(
        [
            _binary_frame("c", subcohort="TCGA-COAD"),
            _binary_frame("d", subcohort="TCGA-READ").iloc[:8],
        ],
        ignore_index=True,
    )
    patients = {
        "family_cptac": _binary_frame("e"),
        "family_rih": _binary_frame("f"),
        "family_surgen": surgen,
        "family_tcga": tcga,
    }
    result = extension._family_standardized_macro(  # noqa: SLF001
        {}, patients, n_bootstrap=50, bootstrap_seed=extension.BOOTSTRAP_SEED
    )
    from sklearn.metrics import roc_auc_score

    assert result["components"]["TCGA_pooled_COAD_READ"] == pytest.approx(
        roc_auc_score(tcga["label"], tcga["mean_logit"])
    )
    assert result["n_bootstrap"] == 50
    assert result["bootstrap_seed"] == 20_260_817
    assert len(result["macro_auroc_ci95"]) == 2
    assert result["directional_gate"]["required_directions"] == 5


def test_sibling_directional_gate_is_explicit_four_of_four() -> None:
    primary = {
        arm: {"auroc": 0.6 + index / 100, "auroc_ci95": [0.51, 0.75]}
        for index, arm in enumerate(
            ("sibling_sr386", "sibling_sr1482", "sibling_tcga_coad", "sibling_tcga_read")
        )
    }
    gate = extension._sibling_directional_gate(primary)  # noqa: SLF001
    assert gate["passed_directions"] == 4
    assert gate["required_directions"] == 4
    assert gate["all_4_points_above_0p5"] is True


def test_confirmatory_met_macro_has_ci_and_exact_gate() -> None:
    rih = _binary_frame("r")
    sr = _binary_frame("s")
    met_results = {
        "family_rih": {"rih_m": {"auroc": 0.75}},
        "family_surgen": {"sr1482_m": {"auroc": 0.70}},
    }
    conclusion = extension._confirmatory_met_conclusion(  # noqa: SLF001
        met_results,
        {("family_rih", "rih_m"): rih, ("family_surgen", "sr1482_m"): sr},
        n_bootstrap=50,
        bootstrap_seed=extension.BOOTSTRAP_SEED,
    )
    assert conclusion["equal_cohort_metastatic_macro_auroc"] == pytest.approx(0.725)
    assert len(conclusion["macro_auroc_ci95"]) == 2
    assert "both target AUROC points" in conclusion["gate"]
    assert conclusion["claim_metastatic_transport"] == (
        conclusion["both_target_points_above_0p5"]
        and conclusion["macro_lower_bound_above_0p5"]
    )


def test_confirmatory_replay_uses_official_schema_and_frozen_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_targets = {"rih_m": {"auroc": 0.71}, "sr1482_m": {"auroc": 0.68}}
    expected_conclusion = {"claim_metastatic_transport": True}

    def official(
        met: dict[tuple[str, str], pd.DataFrame],
        calibrators: dict[str, dict[str, object]],
        *,
        n_bootstrap: int,
        bootstrap_seed: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert met == {}
        assert calibrators == {}
        assert n_bootstrap == 10_000
        assert bootstrap_seed == 20_260_817
        return expected_targets, expected_conclusion

    monkeypatch.setattr(extension, "_official_confirmatory_met", official)
    results = {
        "e2met_confirmatory_family_naive": expected_targets,
        "e2met_confirmatory": expected_conclusion,
    }
    extension._replay_confirmatory_met(results, {}, {})  # noqa: SLF001
    with pytest.raises(extension.ContractError, match="target metrics"):
        extension._replay_confirmatory_met(  # noqa: SLF001
            {**results, "e2met_confirmatory_family_naive": {}}, {}, {}
        )
    with pytest.raises(extension.ContractError, match="macro/gate"):
        extension._replay_confirmatory_met(  # noqa: SLF001
            {**results, "e2met_confirmatory": {}}, {}, {}
        )


def test_legacy_adopted_source_config_tolerates_only_missing_null_refit_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        extension,
        "_material_expectations",
        lambda *_args, **_kwargs: {
            "training.seed": 42,
            "training.refit_max_steps": None,
        },
    )
    legacy = {"training": {"seed": 42}}
    extension._validate_material_config(  # noqa: SLF001
        legacy,
        "family_cptac",
        42,
        source_cv=True,
        context="legacy",
        allow_legacy_missing_refit_steps=True,
    )
    with pytest.raises(extension.ContractError, match="refit_max_steps"):
        extension._validate_material_config(  # noqa: SLF001
            legacy,
            "family_cptac",
            42,
            source_cv=True,
            context="new",
        )


def test_failed_source_cv_attempt_is_quarantined_without_deletion(tmp_path: Path) -> None:
    root = tmp_path / "study" / "overlay"
    directory = extension.source_cv_dir(root, "family_cptac", 45)
    directory.mkdir(parents=True)
    (directory / "partial.bin").write_bytes(b"evidence")
    component = extension.component_root(root)
    request = component / "requests/source_cv/family_cptac/seed45.json"
    log = component / "logs/source_cv/family_cptac/seed45.log"
    failure = request.with_suffix(".failure.json")
    request.parent.mkdir(parents=True)
    log.parent.mkdir(parents=True)
    request.write_text("{}")
    log.write_text("failed")
    failure.write_text("{}")
    destination = extension._quarantine_source_cv(  # noqa: SLF001
        root, "family_cptac", 45, require_failure_marker=True
    )
    assert not directory.exists()
    assert (destination / "partial_output/partial.bin").read_bytes() == b"evidence"
    assert (destination / "quarantine_receipt.json").is_file()


def test_sigterm_releases_job_lock_and_external_recovery_quarantines_orphan(
    tmp_path: Path,
) -> None:
    root = tmp_path / "study" / "overlay"
    marker = tmp_path / "worker_locked"
    job_key = "aim2.source_cv.family_cptac.seed45"
    code = "\n".join(
        (
            "import sys, time",
            f"sys.path.insert(0, {str(extension.REPO)!r})",
            "from pathlib import Path",
            "import aim2_loco_five_seed_extension as extension",
            f"root = Path({str(root)!r})",
            f"with extension._job_lock(root, {job_key!r}):",
            "    directory = extension.source_cv_dir(root, 'family_cptac', 45)",
            "    directory.mkdir(parents=True)",
            "    (directory / 'partial.bin').write_bytes(b'orphan')",
            "    request = extension.component_root(root) / "
            "'requests/source_cv/family_cptac/seed45.json'",
            "    request.parent.mkdir(parents=True)",
            "    request.write_text('{}')",
            "    log = extension.component_root(root) / "
            "'logs/source_cv/family_cptac/seed45.log'",
            "    log.parent.mkdir(parents=True)",
            "    log.write_text('killed')",
            f"    Path({str(marker)!r}).write_text('locked')",
            "    time.sleep(60)",
        )
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=extension.REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.is_file(), process.communicate(timeout=1)
        assert extension._job_lock_is_active(root, job_key)  # noqa: SLF001
        process.terminate()
        process.wait(timeout=10)
        assert extension._job_lock_path(root, job_key).is_file()  # noqa: SLF001
        assert not extension._job_lock_is_active(root, job_key)  # noqa: SLF001
        destination = extension._recover_orphaned_job(  # noqa: SLF001
            root,
            {"stage": "source_cv", "arm": "family_cptac", "seed": 45},
        )
        assert destination is not None
        assert (destination / "partial_output/partial.bin").read_bytes() == b"orphan"
        assert (destination / "quarantine_receipt.json").is_file()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_report_bootstrap_is_not_cli_mutable() -> None:
    parser = extension.build_parser()
    args = parser.parse_args(["report"])
    assert args.n_bootstrap == 10_000
    assert args.bootstrap_seed == 20_260_817
    with pytest.raises(SystemExit):
        parser.parse_args(["report", "--bootstrap-seed", "1"])
