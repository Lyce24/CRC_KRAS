from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim3_tcga_surgen_primary_fine_external_campaign as campaign  # noqa: E402


def _molecular_patients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["d", "v", "c", "g13", "q61", "multi"],
            "kras": ["mutant"] * 6,
            "kras_subvariant": [
                "G12D",
                "G12V",
                "G12C",
                "G13D",
                "Q61H",
                "G12D;G13D",
            ],
            "dataset": ["fixture"] * 6,
            "role": ["primary"] * 6,
        }
    )


def test_score_inventory_is_twenty_five_refit_jobs() -> None:
    jobs = campaign.score_jobs()
    assert len(jobs) == 25
    assert len({job.key for job in jobs}) == 25
    assert {job.task for job in jobs} == set(campaign.TASKS)
    assert {job.seed for job in jobs} == set(campaign.MODEL_SEEDS)


def test_terminal_output_inventory_is_exactly_sixty_seven(tmp_path: Path) -> None:
    inventory = campaign.expected_output_inventory(tmp_path / "external")
    assert len(inventory) == 67
    assert len([path for path in inventory if "/scores/" in str(path)]) == 50
    assert len([path for path in inventory if "/analysis/" in str(path)]) == 5


def test_fine_task_labels_replay_exact_token_membership() -> None:
    patients = _molecular_patients()
    expected = {
        "codon": ({"d", "v", "c", "multi"}, {"g13", "q61"}),
        "g12d_broad": ({"d", "multi"}, {"v", "c", "g13", "q61"}),
        "allele1": ({"d", "multi"}, {"v", "c"}),
        "allele2": ({"v"}, {"d", "c", "multi"}),
        "g12c": ({"c"}, {"d", "v", "multi"}),
    }
    for task, (positive, negative) in expected.items():
        frame = campaign.derive_task_labels(patients, task)
        assert set(frame.loc[frame["label"].eq(1), "patient_id"]) == positive
        assert set(frame.loc[frame["label"].eq(0), "patient_id"]) == negative


def test_label_blind_schema_rejects_molecular_outcomes() -> None:
    spec = next(iter(campaign.TARGETS.values()))
    frame = pd.DataFrame(
        {
            "slide_id": [f"s{index}" for index in range(spec.slides)],
            "patient_id": [f"p{index}" for index in range(spec.slides)],
            "kras_subvariant": ["G12D"] * spec.slides,
        }
    )
    with pytest.raises(campaign.ContractError, match="label-blind schema violation"):
        campaign._blind_frame(frame, spec=spec, context="fixture")


def test_label_blind_schema_is_an_exact_column_allowlist() -> None:
    spec = campaign.TARGETS["cptac_primary"]
    patients = [f"p{index}" for index in range(spec.patients)] + [
        f"p{index}" for index in range(spec.slides - spec.patients)
    ]
    frame = pd.DataFrame(
        {
            "slide_id": [f"s{index}" for index in range(spec.slides)],
            "patient_id": patients,
        }
    )
    frame["unknown_future_field"] = "x"
    with pytest.raises(campaign.ContractError, match="unknown_future_field"):
        campaign._blind_frame(frame, spec=spec, context="unexpected-column")


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("slide_id", np.nan, "null slide or patient"),
        ("patient_id", np.nan, "null slide or patient"),
        ("slide_id", "  ", "blank slide or patient"),
        ("patient_id", "", "blank slide or patient"),
    ],
)
def test_label_blind_schema_rejects_null_and_blank_ids_before_string_cast(
    column: str, bad_value: object, message: str
) -> None:
    spec = campaign.TARGETS["cptac_primary"]
    patients = [f"p{index}" for index in range(spec.patients)] + [
        f"p{index}" for index in range(spec.slides - spec.patients)
    ]
    frame = pd.DataFrame(
        {
            "slide_id": [f"s{index}" for index in range(spec.slides)],
            "patient_id": patients,
        }
    )
    frame.loc[spec.patients - 1, column] = bad_value
    with pytest.raises(campaign.ContractError, match=message):
        campaign._blind_frame(frame, spec=spec, context="adversarial-null-blank")


def test_output_root_is_sibling_only_and_tmp_is_allowed(tmp_path: Path) -> None:
    allowed = tmp_path / "fine-external"
    assert campaign.validate_output_root(allowed) == allowed.resolve()
    with pytest.raises(campaign.ContractError, match="must not overlap"):
        campaign.validate_output_root(campaign.TRAINING_ROOT / "external")
    with pytest.raises(campaign.ContractError, match="production output root"):
        campaign.validate_output_root(Path("/var/tmp/not-governed"))


def test_actual_training_root_and_external_output_must_be_disjoint(tmp_path: Path) -> None:
    training = tmp_path / "training"
    training.mkdir()
    sibling = tmp_path / "external"
    campaign._validate_training_output_separation(sibling, training)
    with pytest.raises(campaign.ContractError, match="resolved training root"):
        campaign._validate_training_output_separation(training / "external", training)


def _metric_frame(n: int = 24) -> pd.DataFrame:
    labels = np.tile([0, 1], n // 2)
    base = np.arange(n, dtype=float) + labels * 3.0
    data: dict[str, object] = {
        "patient_id": [f"p{index}" for index in range(n)],
        "dataset": np.where(np.arange(n) % 4 < 2, "a", "b"),
        "role": ["primary"] * n,
        "label": labels,
    }
    for offset, seed in enumerate(campaign.MODEL_SEEDS):
        data[f"logit_seed{seed}"] = base + offset * 0.01
    frame = pd.DataFrame(data)
    frame["mean_logit_5seed"] = frame[[f"logit_seed{seed}" for seed in campaign.MODEL_SEEDS]].mean(
        axis=1
    )
    return frame


def test_metric_block_reports_seed_mean_sample_sd_and_ensemble_ci() -> None:
    arrays: dict[str, np.ndarray] = {}
    result = campaign._metric_block(
        _metric_frame(),
        n_bootstrap=32,
        stream="fixture",
        arrays=arrays,
        array_key="fixture",
    )
    per_seed = np.asarray(list(result["per_seed_auroc"].values()))
    assert result["status"] == "ESTIMABLE"
    assert result["seed_auroc_mean"] == pytest.approx(per_seed.mean())
    assert result["seed_auroc_sample_sd"] == pytest.approx(per_seed.std(ddof=1))
    assert len(result["five_seed_refit_ensemble"]["ci95"]) == 2
    assert arrays["fixture"].shape == (32,)


def test_metric_block_marks_single_class_not_estimable() -> None:
    frame = _metric_frame()
    frame["label"] = 1
    arrays: dict[str, np.ndarray] = {}
    result = campaign._metric_block(
        frame,
        n_bootstrap=8,
        stream="single",
        arrays=arrays,
        array_key="single",
    )
    assert result["status"] == "NOT_ESTIMABLE_SINGLE_CLASS"
    assert result["five_seed_refit_ensemble"]["auroc"] is None
    assert arrays == {}


def test_equal_cohort_macro_inherits_worst_component_sparse_flag() -> None:
    frames = {}
    for index, target in enumerate(campaign.TARGET_ORDER):
        frame = _metric_frame(8 if index == 0 else 24)
        frame["dataset"] = target
        frames[target] = frame
    arrays: dict[str, np.ndarray] = {}
    result = campaign._macro_block(frames, n_bootstrap=16, task="codon", arrays=arrays)
    assert result["status"] == "ESTIMABLE"
    assert result["support"] == "VERY_SPARSE_LT10"
    assert result["sparse"] is True
    assert result["n_records"] == sum(len(frame) for frame in frames.values())
    assert result["component_support"][campaign.TARGET_ORDER[0]]["sparse"] is True
    assert set(result["component_support"]) == set(campaign.TARGET_ORDER)


def test_cluster_bootstrap_preserves_full_target_label_membership_cells() -> None:
    frame = _metric_frame(8)
    # One patient has two role records. Another patient has the same complete
    # signature so the signature-stratified cluster resample can exchange them.
    duplicate_a = frame.iloc[[0]].copy()
    duplicate_a["dataset"] = "met"
    duplicate_a["patient_id"] = "pair-a"
    duplicate_b = frame.iloc[[2]].copy()
    duplicate_b["dataset"] = "met"
    duplicate_b["patient_id"] = "pair-b"
    frame.loc[0, "patient_id"] = "pair-a"
    frame.loc[2, "patient_id"] = "pair-b"
    work = pd.concat([frame, duplicate_a, duplicate_b], ignore_index=True)
    indices = campaign._clustered_patient_role_indices(
        work, n_bootstrap=20, stream="cluster-fixture"
    )
    expected = work.groupby(["dataset", "label"]).size().sort_index()
    assert indices.shape == (20, len(work))
    for index in indices:
        observed = work.iloc[index].groupby(["dataset", "label"]).size().sort_index()
        pd.testing.assert_series_equal(observed, expected)


def _execution_intervals(peak: int) -> list[tuple[int, int]]:
    return [
        (1_000 + (index // peak) * 100, 1_050 + (index // peak) * 100)
        for index in range(25)
    ]


def _inference_event(
    job: campaign.ScoreJob, interval: tuple[int, int]
) -> dict[str, object]:
    return {
        "job_id": f"fine_external.score.{job.task}.seed{job.seed}",
        "task": job.task,
        "seed": job.seed,
        "started_utc": "2026-08-28T00:00:00.000001+00:00",
        "completed_utc": "2026-08-28T00:01:00.000001+00:00",
        "started_unix_ns": interval[0],
        "completed_unix_ns": interval[1],
        "returncode": 0,
        "cache_hit": False,
        "score_rows": 479,
        "execution_role": "label_blind_native_logit_inference",
    }


def test_inference_event_replay_requires_exact_six_way_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = campaign.score_jobs()
    receipts = {
        campaign.score_receipt_path(tmp_path, job.task, job.seed): {
            "execution": _inference_event(job, interval)
        }
        for job, interval in zip(jobs, _execution_intervals(6), strict=True)
    }
    monkeypatch.setattr(campaign, "_validate_cached_score", lambda _root, _job: pd.DataFrame())
    monkeypatch.setattr(campaign, "_read_json", lambda path: receipts[path])
    events = campaign._inference_events(tmp_path)
    assert len(events) == 25
    assert (
        campaign._parallel_peak(
            events,
            start_key="started_unix_ns",
            completed_key="completed_unix_ns",
        )
        == 6
    )

    for job, interval in zip(jobs, _execution_intervals(5), strict=True):
        receipts[campaign.score_receipt_path(tmp_path, job.task, job.seed)] = {
            "execution": _inference_event(job, interval)
        }
    with pytest.raises(campaign.ContractError, match="exactly six parallel workers"):
        campaign._inference_events(tmp_path)


def test_scheduler_events_bind_job_command_interval_and_returncode(tmp_path: Path) -> None:
    environment = {"device": "cuda", "num_workers_per_job": 4}
    events = []
    for job, interval in zip(campaign.score_jobs(), _execution_intervals(6), strict=True):
        events.append(
            {
                "job_id": f"fine_external.score.{job.task}.seed{job.seed}",
                "task": job.task,
                "seed": job.seed,
                "command": campaign._internal_score_command(
                    tmp_path, job, device="cuda", num_workers=4
                ),
                "cached_before_launch": False,
                "started_utc": "2026-08-28T00:00:00.000001+00:00",
                "completed_utc": "2026-08-28T00:01:00.000001+00:00",
                "started_unix_ns": interval[0],
                "completed_unix_ns": interval[1],
                "returncode": 0,
            }
        )
    assert len(campaign._validate_scheduler_events(tmp_path, events, environment=environment)) == 25
    events[0]["returncode"] = 1
    with pytest.raises(campaign.ContractError, match="scheduler execution event drifted"):
        campaign._validate_scheduler_events(tmp_path, events, environment=environment)


def test_deep_pack_validation_rejects_same_size_binary_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    features = pack / "features.bin"
    coords = pack / "coords.bin"
    features.write_bytes(b"aaaa")
    coords.write_bytes(b"cccc")
    (tmp_path / "features").mkdir()
    slides = [f"s{index}" for index in range(479)]
    patches = [4_915_250 - 478, *([1] * 478)]
    index = pack / "index.parquet"
    pd.DataFrame({"slide_id": slides, "n_patches": patches}).to_parquet(index, index=False)
    meta = pack / "meta.json"
    meta.write_text(json.dumps({"source_dir": str(tmp_path / "features")}), encoding="utf-8")
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "packed_store": {
                    "encoder": "UNI-v1",
                    "path": str(pack),
                    "artifacts": {
                        path.name: campaign._artifact(path)
                        for path in (features, coords, index, meta)
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(campaign.source, "contract_path", lambda _root: contract)
    campaign._validate_pack(
        tmp_path, set(slides), rehash_large_payloads=True
    )
    features.write_bytes(b"bbbb")
    campaign._validate_pack(
        tmp_path, set(slides), rehash_large_payloads=False
    )
    with pytest.raises(campaign.ContractError, match="packed-store payload drifted"):
        campaign._validate_pack(
            tmp_path, set(slides), rehash_large_payloads=True
        )


def test_external_stage_control_gate_resolves_training_root_and_delegates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = tmp_path / "training"
    training.mkdir()
    (tmp_path / "contract.json").write_text(
        json.dumps({"training_root": str(training)}), encoding="utf-8"
    )
    observed: list[Path] = []
    monkeypatch.setattr(
        campaign.source,
        "assert_no_control_artifacts",
        lambda path: observed.append(Path(path)),
    )
    assert campaign._assert_no_control_artifacts_for_external_stage(tmp_path) == training
    assert observed == [training]


def test_terminal_preflight_replay_can_tolerate_later_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = tmp_path / "training"
    training.mkdir()
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"training_root": str(training)}), encoding="utf-8")
    preflight = tmp_path / "receipts/deep_preflight.json"
    preflight.parent.mkdir()
    preflight.write_text(
        json.dumps(
            {
                "status": "ready_for_label_blind_inference",
                "contract": campaign._artifact(contract),
                "authenticated_p75_checkpoints": 25,
                "target_outcomes_opened": False,
                "control_artifacts_absent": True,
            }
        ),
        encoding="utf-8",
    )

    def later_controls_exist(_path: Path) -> None:
        raise campaign.ContractError("later controls exist")

    monkeypatch.setattr(campaign.source, "assert_no_control_artifacts", later_controls_exist)
    assert (
        campaign._load_preflight(tmp_path, require_no_controls=False)["status"]
        == "ready_for_label_blind_inference"
    )
    with pytest.raises(campaign.ContractError, match="later controls exist"):
        campaign._load_preflight(tmp_path, require_no_controls=True)


def test_outcome_opening_stops_at_inference_and_control_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class GateRaised(RuntimeError):
        pass

    def stop_at_gate(_root: Path, *, require_no_controls: bool = False) -> None:
        assert require_no_controls is True
        raise GateRaised

    monkeypatch.setattr(campaign, "verify_inference_seal", stop_at_gate)
    monkeypatch.setattr(
        campaign,
        "_artifact",
        lambda _path: pytest.fail("outcome artifact opened before inference/control gate"),
    )
    with pytest.raises(GateRaised):
        campaign._open_outcomes_after_seal(tmp_path)


def test_strict_disjoint_excludes_all_eight_from_both_rih_roles() -> None:
    dual = sorted(campaign.RIH_DUAL_PATIENTS)
    base = _metric_frame(24)
    frames = {key: base.iloc[:4].copy() for key in campaign.TARGET_ORDER}
    frames["rih_primary"]["patient_id"] = dual[:4]
    frames["rih_metastatic"]["patient_id"] = dual[4:]
    frames["rih_primary"] = pd.concat(
        [frames["rih_primary"], base.iloc[[4]].assign(patient_id="keep-p")],
        ignore_index=True,
    )
    frames["rih_metastatic"] = pd.concat(
        [frames["rih_metastatic"], base.iloc[[5]].assign(patient_id="keep-m")],
        ignore_index=True,
    )
    observed = campaign._strict_disjoint_blocks(frames)
    assert set(observed["rih_primary"]["patient_id"]) == {"keep-p"}
    assert set(observed["rih_metastatic"]["patient_id"]) == {"keep-m"}


def test_declared_external_censes_are_internally_consistent() -> None:
    for task in campaign.TASKS:
        primitive = campaign.EXPECTED_PRIMITIVE_CENSUS[task]
        primary = tuple(
            sum(
                primitive[key][position]
                for key in campaign.TARGET_ORDER
                if campaign.TARGETS[key].role == "primary"
            )
            for position in range(2)
        )
        metastatic = tuple(
            sum(
                primitive[key][position]
                for key in campaign.TARGET_ORDER
                if campaign.TARGETS[key].role == "metastatic"
            )
            for position in range(2)
        )
        all_record = tuple(
            sum(primitive[key][position] for key in campaign.TARGET_ORDER) for position in range(2)
        )
        assert primary == campaign.EXPECTED_PRIMARY_CENSUS[task]
        assert metastatic == campaign.EXPECTED_METASTATIC_CENSUS[task]
        assert all_record == campaign.EXPECTED_ALL_RECORD_CENSUS[task]


@pytest.mark.skipif(
    not all(spec.blind_source.is_file() for spec in campaign.TARGETS.values()),
    reason="production label-blind rosters unavailable",
)
def test_live_canonical_blind_rosters_have_exact_governed_relations() -> None:
    frames = campaign._load_blind_sources()
    relations = campaign._validate_blind_relations(frames)
    assert relations["slides"] == 479
    assert relations["dataset_patient_records"] == 446
    assert relations["unique_patient_ids"] == 438
    assert len(relations["rih_dual_role_patients"]) == 8


def test_plan_is_read_only_and_names_launch_gate(tmp_path: Path) -> None:
    output = tmp_path / "planned"
    result = campaign.plan(output)
    assert result["status"] == "PLAN_ONLY_NO_WRITES"
    assert result["score_jobs"] == 25
    assert result["score_rows"] == 11_975
    assert result["new_fits"] == 0
    assert not output.exists()
