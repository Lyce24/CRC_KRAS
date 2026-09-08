from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_loco_transport
from oceanpath.aim1 import lineage
from oceanpath.workflows import finalize


def test_refit_seeds_before_data_and_model_construction() -> None:
    source = inspect.getsource(finalize._run_refit)

    seed = source.index("L.seed_everything")
    data = source.index("datamodule = MILDataModule")
    model = source.index("module = MILTrainModule")

    assert seed < data < model
    assert "sampling_seed=int(t.seed)" in source


def test_refit_has_exact_optimizer_step_guard() -> None:
    source = inspect.getsource(finalize._run_refit)

    assert '"training.refit_max_steps"' in source
    assert "max_steps=refit_max_steps" in source
    assert "actual_optimizer_steps != refit_max_steps" in source
    assert 'lr_scheduler_interval="step" if refit_max_steps is not None' in source
    assert "lr_scheduler_total_steps=refit_max_steps" in source
    assert "enable_progress_bar=refit_max_steps is None" in source
    assert '"final_learning_rates": final_learning_rates' in source


def test_completed_refit_checks_the_full_training_contract() -> None:
    source = inspect.getsource(aim2_loco_transport._completed_refit)

    for field in (
        "optimizer_step_budget",
        "actual_optimizer_steps",
        "batch_size",
        "accumulate_grad_batches",
        "sampling_seed",
        "patient_natural",
        "lr_scheduler_interval",
        "lr_scheduler_total_steps",
        "final_learning_rates",
    ):
        assert field in source


def test_stepwise_cosine_uses_optimizer_step_horizon() -> None:
    # MILTrainModule is imported lazily by finalize; instantiate it directly so
    # this regression test exercises the scheduler contract, not source text.
    from oceanpath.training.lightning import MILTrainModule

    trained = MILTrainModule(
        arch="abmil",
        in_dim=8,
        num_classes=1,
        model_cfg={"embed_dim": 4, "attn_dim": 2},
        lr=1e-3,
        lr_scheduler="cosine",
        max_epochs=7,
        lr_scheduler_interval="step",
        lr_scheduler_total_steps=6060,
        loss_type="bce",
    )
    configured = trained.configure_optimizers()

    assert configured["lr_scheduler"]["interval"] == "step"
    assert configured["lr_scheduler"]["scheduler"].T_max == 6060


def test_e2a_epoch_ceiling_never_rounds_below_budget(monkeypatch) -> None:
    manifest = pd.DataFrame({"patient_id": [f"p{i}" for i in range(749)]})
    monkeypatch.setattr(aim2_loco_transport.pd, "read_csv", lambda _path: manifest)

    epochs, n_patients = aim2_loco_transport.epochs_for("SurGen")

    assert n_patients == 749
    assert epochs * n_patients >= aim2_loco_transport.STEP_BUDGET
    assert (epochs - 1) * n_patients < aim2_loco_transport.STEP_BUDGET


def test_lineage_rejects_missing_or_path_like_names(monkeypatch) -> None:
    monkeypatch.delenv(lineage.AIM2_LINEAGE_ENV, raising=False)
    with pytest.raises(RuntimeError, match=lineage.AIM2_LINEAGE_ENV):
        lineage.lineage_name()

    monkeypatch.setenv(lineage.AIM2_LINEAGE_ENV, "../legacy")
    with pytest.raises(ValueError, match="Invalid"):
        lineage.lineage_name()


def test_lineage_root_is_new_subdirectory(monkeypatch) -> None:
    monkeypatch.setenv(lineage.AIM2_LINEAGE_ENV, "aim2_test_lineage")

    root = lineage.aim2_root()

    assert root == lineage.paths.OUTPUT_ROOT / "reruns" / "aim2_test_lineage"
    assert root != lineage.paths.OUTPUT_ROOT


def test_exclusive_writer_refuses_existing_artifact(tmp_path) -> None:
    destination = tmp_path / "result.json"
    lineage.write_json_once(destination, {"version": 1})

    with pytest.raises(FileExistsError, match="overwrite"):
        lineage.write_json_once(destination, {"version": 2})

    assert destination.read_text().strip().endswith("}")
