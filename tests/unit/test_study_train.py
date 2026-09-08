from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aim2_loco_transport  # noqa: E402
from oceanpath.workflows import training  # noqa: E402
from tools import study_train  # noqa: E402


def test_hydra_config_dir_is_the_repository_config_root() -> None:
    expected = Path(__file__).resolve().parents[2] / "configs"

    assert study_train.hydra_config_dir() == expected.resolve()
    assert (study_train.hydra_config_dir() / "train.yaml").is_file()


def test_hydra_train_composes_config_without_entering_training(monkeypatch, capsys) -> None:
    def forbidden_training(_cfg):
        raise AssertionError("--cfg job must compose only; it must not start training")

    monkeypatch.setattr(training, "run_training", forbidden_training)
    original_argv = sys.argv

    study_train.run_hydra_train(["--cfg", "job"])

    assert sys.argv is original_argv
    rendered = capsys.readouterr().out
    assert "training:" in rendered
    assert "platform:" in rendered


def test_size_matched_source_cv_dry_run_constructs_repo_launcher_command(
    monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.setattr(aim2_loco_transport.lineage, "lineage_name", lambda required=True: "test_lineage")
    monkeypatch.setattr(aim2_loco_transport, "e2a_root", lambda: tmp_path / "e2a")
    monkeypatch.setattr(aim2_loco_transport, "ensure_splits", lambda _target: None)

    def forbidden_subprocess(*_args, **_kwargs):
        raise AssertionError("a source-CV dry run must not launch training")

    monkeypatch.setattr(aim2_loco_transport, "_run_logged", forbidden_subprocess)
    args = SimpleNamespace(
        cap=8192,
        target=None,
        size_matched=True,
        seed=42,
        dry_run=True,
    )

    aim2_loco_transport.cmd_source_cv(args)

    rendered = capsys.readouterr().out
    launcher = Path(__file__).resolve().parents[2] / "tools" / "study_train.py"
    assert f"{sys.executable} {launcher} hydra-train" in rendered
    assert "data.aim1_model=e2a_rih_sm" in rendered
    assert "training.dataset_max_instances=8192" in rendered
    assert f"train_dir={tmp_path / 'e2a/source_cv/cap8192/rih_sm/seed42'}" in rendered
    assert (
        f"hydra.run.dir={tmp_path / 'e2a/hydra_runs/source_cv_rih_sm_cap8192_seed42'}"
        in rendered
    )
    assert "hydra.job.chdir=false" in rendered
