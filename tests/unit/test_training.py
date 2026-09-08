"""CPU contracts for the Lightning module and slide-level training entry point."""

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from oceanpath.training.lightning import MILTrainModule


def _tiny_module() -> MILTrainModule:
    return MILTrainModule(
        arch="abmil",
        in_dim=8,
        num_classes=2,
        model_cfg={"embed_dim": 16, "attn_dim": 8, "dropout": 0.0},
        lr=1.0e-3,
        max_epochs=2,
        canary_interval=0,
        collect_embeddings=False,
    )


def test_training_step_backpropagates_through_classifier(monkeypatch):
    module = _tiny_module()
    monkeypatch.setattr(module, "log", lambda *_args, **_kwargs: None)
    batch = {
        "features": torch.randn(2, 5, 8),
        "mask": torch.ones(2, 5, dtype=torch.bool),
        "labels": torch.tensor([0, 1]),
        "slide_ids": ["a", "b"],
    }

    loss = module.training_step(batch, batch_idx=0)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert module.model.head.weight.grad is not None
    assert torch.isfinite(module.model.head.weight.grad).all()


def _batch(n=5, b=2):
    return {
        "features": torch.randn(b, n, 8),
        "mask": torch.ones(b, n),
        "labels": torch.tensor([0, 1][:b]),
        "slide_ids": ["a", "b"][:b],
    }


def test_training_step_does_not_sync_the_device_per_step(monkeypatch):
    """The NaN guard must not read a GPU tensor on the host every step.

    A per-step `.item()` blocks the CPU on the GPU, so the H2D copy of the
    next batch cannot overlap the current step's compute — the dominant cost
    at small batch sizes. The tally lives on-device and is read once per epoch.
    """
    module = _tiny_module()
    monkeypatch.setattr(module, "log", lambda *_a, **_k: None)

    calls = []
    original = torch.Tensor.item

    def spy(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(torch.Tensor, "item", spy)
    module.training_step(_batch(), batch_idx=0)
    assert calls == [], f"training_step performed {len(calls)} host sync(s)"


def test_non_finite_loss_becomes_a_zero_gradient_step(monkeypatch):
    """A poisoned batch must yield a no-op update, not a crash or a NaN param.

    nan_to_num keeps the tensor in the graph, so backward still records inf
    checks for the GradScaler — returning a detached zero would crash it.
    """
    module = _tiny_module()
    monkeypatch.setattr(module, "log", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "_compute_loss",
        lambda logits, _labels, _weights=None: logits.sum() + torch.tensor(float("nan")),
    )

    loss = module.training_step(_batch(), batch_idx=0)
    loss.backward()

    assert torch.isfinite(loss) and float(loss.detach()) == 0.0
    # Gradients EXIST (so GradScaler records its inf checks) and are all zero,
    # so the optimizer step is a harmless no-op.
    assert module.model.head.weight.grad is not None
    assert torch.isfinite(module.model.head.weight.grad).all()
    assert float(module.model.head.weight.grad.abs().sum()) == 0.0


def test_nan_with_nan_local_derivative_still_produces_nan_grads(monkeypatch):
    """Pins a known limit of the nan_to_num guard, unchanged by the fast path.

    `nan_to_num`'s own derivative is `isfinite(input)`, so it zeroes the
    incoming gradient — but only *after* the upstream chain has been formed.
    When the NaN enters through a multiplication (0 * nan == nan) the parameter
    gradients are NaN regardless. Under `16-mixed` the GradScaler detects this
    and skips the step; under `32-true` there is no scaler, so such a step
    would poison the weights.
    """
    module = _tiny_module()
    monkeypatch.setattr(module, "log", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "_compute_loss",
        lambda logits, _labels, _weights=None: logits.sum() * torch.tensor(float("nan")),
    )

    module.training_step(_batch(), batch_idx=0).backward()
    assert not torch.isfinite(module.model.head.weight.grad).all()


def test_finite_but_saturating_logits_are_not_counted_as_nan(monkeypatch):
    """`_compute_loss` clamps logits, so huge-but-finite loss is not a repair."""
    module = _tiny_module()
    monkeypatch.setattr(module, "log", lambda *_a, **_k: None)
    batch = _batch()
    batch["features"] *= 1e4
    module.on_train_epoch_start()
    module.training_step(batch, batch_idx=0)
    assert int(module._nan_steps.item()) == 0


def test_nan_tally_reports_once_per_epoch(monkeypatch, caplog):
    module = _tiny_module()
    monkeypatch.setattr(module, "log", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "_compute_loss",
        lambda logits, _labels, _weights=None: logits.sum() * torch.tensor(float("nan")),
    )
    module.on_train_epoch_start()
    module.training_step(_batch(), batch_idx=0)
    module.training_step(_batch(), batch_idx=1)
    assert int(module._nan_steps.item()) == 2

    with caplog.at_level("WARNING"):
        module.on_train_epoch_end()
    assert "2 non-finite training step" in caplog.text

    module.on_train_epoch_start()
    assert int(module._nan_steps.item()) == 0


def test_fp32_escape_hatch_is_scoped_to_float16():
    """bf16 has fp32's exponent range, so forcing fp32 there costs speed for nothing."""
    module = _tiny_module()

    class _Plugin:
        def __init__(self, precision):
            self.precision = precision

    class _Trainer:
        def __init__(self, precision):
            self.precision_plugin = _Plugin(precision)

    assert module._autocast_dtype() is None  # no trainer attached

    module._trainer = _Trainer("16-mixed")
    assert module._autocast_dtype() is torch.float16

    module._trainer = _Trainer("bf16-mixed")
    assert module._autocast_dtype() is torch.bfloat16

    module._trainer = _Trainer("32-true")
    assert module._autocast_dtype() is None
    module._trainer = None


def test_forward_accepts_bool_and_float_masks():
    """Collators emit float masks; some callers pass bool. Both must work."""
    module = _tiny_module()
    for mask in (torch.ones(2, 5), torch.ones(2, 5, dtype=torch.bool)):
        batch = _batch()
        batch["mask"] = mask
        out = module._forward_model(batch)
        assert torch.isfinite(out.logits).all()


def test_float16_batches_run_under_full_precision():
    """A float16 packed store must work under `32-true`, not raise Half-vs-Float.

    Half batches are the point of the packed store (half the IPC and PCIe
    bytes), but with no autocast active nothing promotes them, so the forward
    boundary reconciles the dtype against the model's own weights.
    """
    module = _tiny_module()
    batch = _batch()
    batch["features"] = batch["features"].half()
    out = module._forward_model(batch)
    assert out.logits.dtype == torch.float32
    assert torch.isfinite(out.logits).all()


def test_masked_padding_does_not_influence_the_slide_embedding():
    module = _tiny_module()
    module.eval()
    feats = torch.randn(1, 4, 8)
    padded = torch.cat([feats, torch.randn(1, 6, 8) * 1e3], dim=1)
    mask = torch.cat([torch.ones(1, 4), torch.zeros(1, 6)], dim=1)

    with torch.no_grad():
        tight = module.model(feats, mask=torch.ones(1, 4))
        loose = module.model(padded, mask=mask)
    torch.testing.assert_close(tight.slide_embedding, loose.slide_embedding)


def test_validation_step_does_not_materialise_attention(monkeypatch):
    module = _tiny_module()
    monkeypatch.setattr(module, "log", lambda *_a, **_k: None)
    seen = {}
    original = module._forward_model
    monkeypatch.setattr(
        module,
        "_forward_model",
        lambda batch, return_attention=False: (
            seen.setdefault("attn", return_attention)
            or original(batch, return_attention=return_attention)
        ),
    )
    module.validation_step(_batch(), batch_idx=0)
    assert seen["attn"] is False


def test_training_module_builds_optimizer_and_scheduler_contract():
    configured = _tiny_module().configure_optimizers()

    assert configured["optimizer"].param_groups
    assert hasattr(configured["lr_scheduler"], "step")


def _tiny_cohort(tmp_path, n_slides=10, feat_dim=1024):
    """Feature dir + manifest + splits shared by the CLI integration tests."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    rng = np.random.default_rng(0)
    rows, split_rows = [], []
    for index in range(n_slides):
        slide_id = f"slide_{index}"
        n_patches = 4 + index * 3
        with h5py.File(feature_dir / f"{slide_id}.h5", "w") as handle:
            handle.create_dataset(
                "features",
                data=rng.standard_normal((n_patches, feat_dim)).astype(np.float32),
            )
            handle.create_dataset(
                "coords",
                data=rng.integers(0, 1000, (n_patches, 2)).astype(np.int64),
            )
        rows.append(
            {
                "slide_id": f"{slide_id}.svs",
                "patient_id": f"P{index}",
                "target_label": index % 2,
            }
        )
        split_rows.append({"slide_id": slide_id, "fold": index % 5})

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    pd.DataFrame(split_rows).to_parquet(splits_dir / "splits.parquet", index=False)
    return feature_dir, manifest, splits_dir


def test_training_cli_trains_end_to_end_from_a_packed_store(tmp_path):
    """The fast path must survive the real CLI: Hydra -> DataModule -> Lightning.

    Exercises packed reads, float16 batches under `32-true`, fixed-size bags
    and batch_size > 1 together, which is the configuration the speed work is
    actually for.
    """
    from oceanpath.datasets.packed import pack_features

    repo_root = Path(__file__).resolve().parents[2]
    feature_dir, manifest, splits_dir = _tiny_cohort(tmp_path)
    pack_dir = tmp_path / "packed"
    pack_features(feature_dir, pack_dir)

    train_dir = tmp_path / "train-output"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "train.py"),
        "runtime.verify_environment=false",
        "platform.num_workers=0",
        "platform.accelerator=cpu",
        f"data.csv_path={manifest}",
        f"data.feature_h5_dir={feature_dir}",
        f"+splits.output_dir={splits_dir}",
        "splits.scheme=kfold",
        "splits.n_folds=2",
        "training.verify_splits=false",
        "training.max_epochs=1",
        "training.warmup_epochs=0",
        "training.batch_size=4",
        "training.eval_batch_size=2",
        "training.fixed_bag_size=8",
        "training.force_float32=false",
        f"training.packed_dir={pack_dir}",
        "training.skip_finalize=true",
        "training.collect_embeddings=false",
        f"train_dir={train_dir}",
    ]
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib")}
    completed = subprocess.run(
        command, cwd=repo_root, env=environment, text=True, capture_output=True, check=False
    )

    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    assert "Using packed feature store" in (completed.stdout + completed.stderr)
    assert list(train_dir.rglob("*.ckpt")), "training produced no checkpoint"


def test_training_cli_refuses_a_stale_pack(tmp_path):
    """A re-extracted slide must not be silently trained against an old pack."""
    from oceanpath.datasets.packed import pack_features

    repo_root = Path(__file__).resolve().parents[2]
    feature_dir, manifest, splits_dir = _tiny_cohort(tmp_path, n_slides=6, feat_dim=1024)
    pack_dir = tmp_path / "packed"
    pack_features(feature_dir, pack_dir)

    # Re-extraction: same slide, different content.
    with h5py.File(feature_dir / "slide_0.h5", "w") as handle:
        handle.create_dataset("features", data=np.zeros((17, 1024), dtype=np.float32))
        handle.create_dataset("coords", data=np.zeros((17, 2), dtype=np.int64))

    command = [
        sys.executable,
        str(repo_root / "scripts" / "train.py"),
        "runtime.verify_environment=false",
        "platform.num_workers=0",
        "platform.accelerator=cpu",
        f"data.csv_path={manifest}",
        f"data.feature_h5_dir={feature_dir}",
        f"+splits.output_dir={splits_dir}",
        "splits.scheme=kfold",
        "splits.n_folds=2",
        "training.verify_splits=false",
        "training.max_epochs=1",
        f"training.packed_dir={pack_dir}",
        f"train_dir={tmp_path / 'out'}",
    ]
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib")}
    completed = subprocess.run(
        command, cwd=repo_root, env=environment, text=True, capture_output=True, check=False
    )
    assert completed.returncode != 0
    assert "STALE" in completed.stdout + completed.stderr


def test_training_cli_dry_run_reads_per_slide_h5_contract(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    rows = []
    split_rows = []
    for index in range(10):
        slide_id = f"slide_{index}"
        with h5py.File(feature_dir / f"{slide_id}.h5", "w") as handle:
            handle.create_dataset(
                "features",
                data=np.full((4 + index, 1024), index, dtype=np.float32),
            )
        rows.append(
            {
                "slide_id": f"{slide_id}.svs",
                "patient_id": f"P{index}",
                "target_label": index % 2,
            }
        )
        split_rows.append({"slide_id": slide_id, "fold": index % 5})

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    pd.DataFrame(split_rows).to_parquet(splits_dir / "splits.parquet", index=False)
    train_dir = tmp_path / "train-output"

    command = [
        sys.executable,
        str(repo_root / "scripts" / "train.py"),
        "dry_run=true",
        "runtime.verify_environment=false",
        "platform.num_workers=0",
        f"data.csv_path={manifest}",
        f"data.feature_h5_dir={feature_dir}",
        f"+splits.output_dir={splits_dir}",
        "splits.scheme=kfold",
        "training.verify_splits=false",
        f"train_dir={train_dir}",
    ]
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "matplotlib")}
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "DRY RUN — train.py" in completed.stdout
    assert "Train size:  8" in completed.stdout
    assert "Val size:    2" in completed.stdout
    assert not train_dir.exists()


def test_accumulates_bfloat16_embeddings(tmp_path):
    """Regression: bf16-mixed produces a BFloat16 slide embedding and NumPy has
    no bfloat16 dtype, so an unguarded .numpy() raised
    'Got unsupported ScalarType BFloat16' on the first validation batch."""
    from oceanpath.models import MILOutput

    module = MILTrainModule(
        arch="abmil",
        in_dim=8,
        num_classes=2,
        model_cfg={"embed_dim": 16, "attn_dim": 8, "dropout": 0.0},
        canary_interval=0,
        collect_embeddings=True,
    )
    output = MILOutput(
        logits=torch.randn(2, 2),
        slide_embedding=torch.randn(2, 16, dtype=torch.bfloat16),
        extras={},
    )
    batch = {"labels": torch.tensor([0, 1]), "slide_ids": ["a", "b"]}

    module._accumulate_embeddings(module._val_embeddings, output, batch)

    assert len(module._val_embeddings) == 2
    assert module._val_embeddings[0]["embedding"].dtype == np.float32
    path = module.save_embeddings(str(tmp_path), prefix="val")
    assert path is not None and Path(path).is_file()
