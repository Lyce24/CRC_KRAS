"""Fail-closed identity and completion evidence for reusable pipeline outputs."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from oceanpath.config import FoundationPaths
from oceanpath.splitting import SplitConfig, generate_splits, split_identity_fingerprint
from oceanpath.workflows.training import (
    _assert_training_output_identity,
    _fold_complete,
    _validate_training_config,
    _write_fold_completion,
    _write_training_completion,
    _write_training_identity,
    training_run_fingerprint,
    validate_training_run_dir,
)

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _compose_train(tmp_path: Path, *overrides: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[f"platform.project_root={tmp_path}", *overrides],
        )

    paths = FoundationPaths.from_config(cfg)
    paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text("De ID,Binary WHO 2022\ncase-1,0\ncase-2,1\n")
    paths.splits_dir.mkdir(parents=True, exist_ok=True)
    (paths.splits_dir / ".integrity_hash").write_text(
        json.dumps({"hash": "split-v1", "csv_path": str(paths.manifest_path)})
    )
    return cfg


def _make_fold_evidence(fold_dir: Path, fingerprint: str, fold_idx: int = 0) -> Path:
    checkpoint = fold_dir / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint-v1")
    (fold_dir / "config.yaml").write_text("model: abmil\n")
    (fold_dir / "fold_metrics.json").write_text('{"val/auroc": 0.75}')
    (fold_dir / "preds_val.parquet").write_bytes(b"predictions-v1")
    _write_fold_completion(
        fold_dir,
        fold_idx=fold_idx,
        training_fingerprint=fingerprint,
        best_checkpoint=str(checkpoint),
    )
    return checkpoint


@pytest.mark.parametrize(
    "override",
    [
        "model.attn_dim=64",
        "training.loss_type=focal",
        "training.batch_size=8",
        "training.instance_dropout=0.5",
        "training.seed=999",
    ],
)
def test_material_training_changes_get_new_identity(tmp_path, override):
    base = _compose_train(tmp_path)
    changed = _compose_train(tmp_path, override)

    assert training_run_fingerprint(base) != training_run_fingerprint(changed)


@pytest.mark.parametrize(
    "override",
    ["verbose=true", "dry_run=true", "training.force_rerun=true", "wandb.offline=true"],
)
def test_operational_training_switches_do_not_change_identity(tmp_path, override):
    base = _compose_train(tmp_path)
    changed = _compose_train(tmp_path, override)

    assert training_run_fingerprint(base) == training_run_fingerprint(changed)


def test_feature_inventory_change_gets_new_training_identity(tmp_path):
    cfg = _compose_train(tmp_path)
    feature_dir = FoundationPaths.from_config(cfg).feature_h5_dir
    feature_dir.mkdir(parents=True)
    feature_path = feature_dir / "case-1.h5"
    feature_path.write_bytes(b"feature-v1")
    first = training_run_fingerprint(cfg)

    feature_path.write_bytes(b"feature-v2-with-different-size")

    assert training_run_fingerprint(cfg) != first


def test_training_requires_one_selected_target(tmp_path):
    cfg = _compose_train(tmp_path)
    cfg.data.label_columns = []
    with pytest.raises(ValueError, match="exactly one encoded target column"):
        _validate_training_config(cfg)


def test_training_directory_rejects_material_config_collision(tmp_path):
    shared_output = tmp_path / "shared-training-output"
    base = _compose_train(tmp_path, f"train_dir={shared_output}")
    changed = _compose_train(
        tmp_path,
        f"train_dir={shared_output}",
        "model.attn_dim=64",
    )
    output_dir = FoundationPaths.from_config(base).train_dir
    assert output_dir == FoundationPaths.from_config(changed).train_dir

    output_dir.mkdir(parents=True)
    _write_training_identity(base, output_dir)

    assert _assert_training_output_identity(base, output_dir) == training_run_fingerprint(base)
    with pytest.raises(RuntimeError, match="identity collision"):
        _assert_training_output_identity(changed, output_dir)


def test_fold_completion_requires_untampered_config_predictions_and_checkpoint(tmp_path):
    fingerprint = "training-fingerprint"
    fold_dir = tmp_path / "fold_0"
    checkpoint = _make_fold_evidence(fold_dir, fingerprint)

    assert _fold_complete(
        fold_dir,
        fold_idx=0,
        training_fingerprint=fingerprint,
    )

    checkpoint.write_bytes(b"different checkpoint")
    assert not _fold_complete(
        fold_dir,
        fold_idx=0,
        training_fingerprint=fingerprint,
    )


def test_training_completion_validates_fold_and_root_artifacts(tmp_path):
    fingerprint = "training-fingerprint"
    fold_dir = tmp_path / "fold_0"
    _make_fold_evidence(fold_dir, fingerprint)
    (tmp_path / "cv_summary.json").write_text('{"n_folds": 1}')

    _write_training_completion(
        tmp_path,
        training_fingerprint=fingerprint,
        n_folds=1,
        skip_finalize=True,
    )
    assert (
        validate_training_run_dir(
            tmp_path,
            expected_fingerprint=fingerprint,
        )["status"]
        == "completed"
    )

    (fold_dir / "preds_val.parquet").write_bytes(b"tampered predictions")
    with pytest.raises(RuntimeError, match="Fold 0 completion evidence is invalid"):
        validate_training_run_dir(tmp_path, expected_fingerprint=fingerprint)


def _write_split_manifest(path: Path) -> None:
    rows = ["filename,label,patient_id"]
    rows.extend(f"slide-{index},{index % 2},patient-{index}" for index in range(40))
    path.write_text("\n".join(rows) + "\n")


def _split_config(tmp_path: Path, *, seed: int) -> SplitConfig:
    manifest = tmp_path / "manifest.csv"
    if not manifest.exists():
        _write_split_manifest(manifest)
    return SplitConfig(
        scheme="holdout",
        name="shared-name",
        csv_path=str(manifest),
        output_dir=str(tmp_path / "splits"),
        filename_column="filename",
        label_column="label",
        group_column="patient_id",
        seed=seed,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )


def test_existing_splits_require_matching_config_and_manifest_identity(tmp_path):
    original = _split_config(tmp_path, seed=42)
    changed = _split_config(tmp_path, seed=123)

    first = generate_splits(original)
    reused = generate_splits(original)
    assert first.integrity_hash == reused.integrity_hash
    assert split_identity_fingerprint(original) != split_identity_fingerprint(changed)

    with pytest.raises(RuntimeError, match="Split output identity collision"):
        generate_splits(changed)

    generate_splits(changed, force=True)
    summary = json.loads((tmp_path / "splits" / "summary.json").read_text())
    assert summary["identity"]["fingerprint"] == split_identity_fingerprint(changed)


def test_split_profile_name_tracks_seed_and_fold_overrides(tmp_path):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        base = compose(
            config_name="make_splits",
            overrides=[f"platform.project_root={tmp_path}"],
        )
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        changed = compose(
            config_name="make_splits",
            overrides=[
                f"platform.project_root={tmp_path}",
                "splits.seed=123",
                "splits.n_folds=3",
            ],
        )

    assert base.splits.name == "oofk5_seed42_es15"
    assert changed.splits.name == "oofk3_seed123_es15"
    assert (
        FoundationPaths.from_config(base).splits_dir
        != FoundationPaths.from_config(changed).splits_dir
    )
