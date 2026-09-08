"""
Model finalization after cross-validation (Phase 2 of Stage 3).

After all CV folds complete, produces configured final model artifacts:

  1. best_fold — single fold checkpoint with the best validation score
  2. ensemble  — optional K-checkpoint ensemble for project branches
  3. refit     — optional model retrained on all training data

The main-branch default is ``best_fold`` only, selected with validation evidence
before held-out evaluation. Projects can explicitly enable ensemble or refit.

Output structure
════════════════════════════════════════════════════════════════════════════
  output_dir/final/
  ├── best_fold/
  │   ├── model.ckpt          # Lightning checkpoint
  │   └── info.json           # selection metadata
  ├── ensemble/                # optional
  │   ├── fold_0.ckpt ... fold_{K-1}.ckpt
  │   └── info.json           # model config + fold scores
  ├── refit/                   # optional
  │   ├── model.ckpt
  │   └── info.json           # epoch rule, training stats
  └── finalize_summary.json   # top-level summary of requested strategies

Stage 4 contract
════════════════════════════════════════════════════════════════════════════
  Each info.json has a 'strategy' field (best_fold | ensemble | refit)
  and a 'model_path' (or list of paths for ensemble).
  Stage 4 reads info.json to know HOW to load and run inference:
    best_fold / refit → MILTrainModule.load_from_checkpoint(model_path)
    ensemble          → load each fold_*.ckpt, average softmax probs
"""

import contextlib
import gc
import json
import logging
import re
import shutil
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import RichProgressBar
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def finalize_models(
    cfg: DictConfig,
    output_dir: Path,
    n_folds: int,
    all_fold_metrics: list[dict],
) -> dict:
    """
    Phase 2 entry point: produce the configured final model artifacts.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config (same as used for fold training).
    output_dir : Path
        Experiment output directory (contains fold_0/, fold_1/, ...).
    n_folds : int
        Number of CV folds.
    all_fold_metrics : list[dict]
        Per-fold metrics dicts from Phase 1. Each must have
        'best_checkpoint' (str path) and 'best_epoch' (int) keys.

    Returns
    -------
    dict keyed by requested strategies (best_fold, ensemble, refit).
    Each info dict has 'error' key if that strategy failed.
    """
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Phase 2: Finalizing models")
    logger.info("=" * 60)

    requested = [
        str(value)
        for value in OmegaConf.select(
            cfg,
            "training.final_strategies",
            default=["best_fold"],
        )
    ]
    supported = {"best_fold", "ensemble", "refit"}
    invalid = sorted(set(requested) - supported)
    if not requested or invalid:
        raise ValueError(
            "training.final_strategies must be a non-empty list drawn from "
            f"{sorted(supported)}; invalid={invalid}"
        )
    if "best_fold" not in requested:
        raise ValueError("training.final_strategies must include best_fold")

    results = {}

    # ── 1. Best fold (fast — just copies a checkpoint) ───────────────────
    results["best_fold"] = _safe_run(
        "best_fold",
        _save_best_fold,
        cfg,
        final_dir,
        all_fold_metrics,
    )

    # ── 2. Ensemble (fast — copies K checkpoints) ────────────────────────
    if "ensemble" in requested:
        results["ensemble"] = _safe_run(
            "ensemble",
            _save_ensemble,
            cfg,
            final_dir,
            n_folds,
            all_fold_metrics,
        )

    # ── 3. Refit (slow — full training run) ──────────────────────────────
    if "refit" in requested:
        results["refit"] = _safe_run(
            "refit",
            _run_refit,
            cfg,
            final_dir,
            all_fold_metrics,
        )

    # ── Save summary ─────────────────────────────────────────────────────
    summary_path = final_dir / "finalize_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Finalization summary → {summary_path}")

    _print_summary(results)

    return results


def _safe_run(name: str, fn, *args) -> dict:
    """Run a finalization step, catching and logging errors."""
    try:
        return fn(*args)
    except Exception as e:
        logger.error(f"{name} failed: {e}", exc_info=True)
        return {"strategy": name, "error": str(e)}


# ═════════════════════════════════════════════════════════════════════════════
# 1. Best fold
# ═════════════════════════════════════════════════════════════════════════════


def _save_best_fold(
    cfg: DictConfig,
    final_dir: Path,
    all_fold_metrics: list[dict],
) -> dict:
    """
    Copy the single best fold checkpoint.

    Selection uses the same monitor_metric / monitor_mode as training.
    """
    t = cfg.training
    metric_key = t.monitor_metric
    mode = t.monitor_mode

    # ── Find best fold ───────────────────────────────────────────────────
    best_idx = None
    best_score = float("inf") if mode == "min" else float("-inf")

    for i, fm in enumerate(all_fold_metrics):
        score = fm.get(metric_key)
        if score is None:
            logger.warning(f"Fold {i} missing metric '{metric_key}' — skipping")
            continue
        is_better = (score < best_score) if mode == "min" else (score > best_score)
        if is_better:
            best_score = score
            best_idx = i

    if best_idx is None:
        raise ValueError(
            f"No fold has metric '{metric_key}'. "
            f"Available keys: {list(all_fold_metrics[0].keys()) if all_fold_metrics else '(empty)'}"
        )

    # ── Copy checkpoint ──────────────────────────────────────────────────
    src_ckpt = all_fold_metrics[best_idx].get("best_checkpoint", "")
    if not src_ckpt or not Path(src_ckpt).is_file():
        raise FileNotFoundError(f"Fold {best_idx} checkpoint not found: '{src_ckpt}'")

    dest_dir = final_dir / "best_fold"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_ckpt = dest_dir / "model.ckpt"
    shutil.copy2(src_ckpt, dest_ckpt)

    info = {
        "strategy": "best_fold",
        "source_fold": best_idx,
        "monitor_metric": metric_key,
        "monitor_mode": mode,
        "best_score": float(best_score),
        "best_epoch": all_fold_metrics[best_idx].get("best_epoch"),
        "source_checkpoint": str(src_ckpt),
        "model_path": str(dest_ckpt),
    }
    _write_info(dest_dir, info)
    logger.info(
        f"best_fold: fold {best_idx} "
        f"({metric_key}={best_score:.4f}, epoch={info['best_epoch']}) → {dest_ckpt}"
    )
    return info


# ═════════════════════════════════════════════════════════════════════════════
# 2. Ensemble
# ═════════════════════════════════════════════════════════════════════════════


def _save_ensemble(
    cfg: DictConfig,
    final_dir: Path,
    n_folds: int,
    all_fold_metrics: list[dict],
) -> dict:
    """
    Copy all fold checkpoints into an ensemble directory.

    Stage 4 loads each, runs inference, and averages softmax probabilities.
    """
    t = cfg.training
    dest_dir = final_dir / "ensemble"
    dest_dir.mkdir(parents=True, exist_ok=True)

    fold_details = []
    copied = 0

    for i, fm in enumerate(all_fold_metrics):
        src_ckpt = fm.get("best_checkpoint", "")
        if not src_ckpt or not Path(src_ckpt).is_file():
            logger.warning(f"Fold {i} checkpoint missing — skipping in ensemble")
            fold_details.append({"fold": i, "error": "checkpoint not found"})
            continue

        dest_ckpt = dest_dir / f"fold_{i}.ckpt"
        shutil.copy2(src_ckpt, dest_ckpt)
        copied += 1

        fold_details.append(
            {
                "fold": i,
                "model_path": str(dest_ckpt),
                "best_epoch": fm.get("best_epoch"),
                "monitor_score": fm.get(t.monitor_metric, float("inf")),
            }
        )

    if copied == 0:
        raise FileNotFoundError("No fold checkpoints found — cannot create ensemble")

    info = {
        "strategy": "ensemble",
        "n_folds": n_folds,
        "n_checkpoints": copied,
        "monitor_metric": t.monitor_metric,
        "ensemble_method": "mean_prob",
        "folds": fold_details,
        # Stage 4 needs these to reconstruct model architecture
        "model_arch": cfg.model.arch,
        "model_cfg": OmegaConf.to_container(cfg.model, resolve=True),
        "in_dim": cfg.encoder.feature_dim,
    }
    _write_info(dest_dir, info)
    logger.info(f"ensemble: {copied}/{n_folds} fold checkpoints → {dest_dir}")
    return info


# ═════════════════════════════════════════════════════════════════════════════
# 3. Refit on full training data
# ═════════════════════════════════════════════════════════════════════════════


def _refit_class_weights(cfg: DictConfig, datamodule, num_classes: int) -> list[float] | None:
    """Class weights for the refit, resolved the same way a fold resolves them.

    ``training.auto_class_weights`` is honoured here as well as in run_fold.
    Reading only the explicit ``training.class_weights`` would leave a refit
    unweighted while every fold it is meant to summarize was trained with
    inverse-prevalence weights — the same recipe on more data, except not.
    """
    from oceanpath.workflows.training import _inverse_prevalence_weights

    explicit = OmegaConf.select(cfg, "training.class_weights", default=None)
    auto = OmegaConf.select(cfg, "training.auto_class_weights", default=None)
    if bool(
        OmegaConf.select(
            cfg, "training.training_class_weighted_loss", default=False
        )
    ) and auto in (None, "null", False):
        auto = "inverse_prevalence"
    if auto in (None, "null", False):
        return explicit
    if str(auto) != "inverse_prevalence":
        raise ValueError(
            f"training.auto_class_weights={auto!r}; only 'inverse_prevalence' is supported"
        )
    if explicit not in (None, "null"):
        raise ValueError(
            "training.class_weights and training.auto_class_weights are mutually exclusive"
        )
    weights = _inverse_prevalence_weights(datamodule.train_dataset, num_classes)
    logger.info(
        "Refit class weights (inverse prevalence, full training set): %s",
        [round(w, 4) for w in weights],
    )
    return weights


def _run_refit(
    cfg: DictConfig,
    final_dir: Path,
    all_fold_metrics: list[dict],
) -> dict:
    """
    Train a fresh model on ALL training data for a fixed number of epochs.

    Key design choices:
      - NO validation set → no early stopping, no model selection
      - Epoch count derived from CV fold stopping points (p75 by default), or
        an exact optimizer-step budget when ``training.refit_max_steps`` is set
      - plateau scheduler falls back to cosine (plateau needs val loss)
      - Final model = last epoch state (not "best" — there's nothing to select on)
    """
    from oceanpath.datasets import MILDataModule
    from oceanpath.training.lightning import MILTrainModule

    t = cfg.training
    start_time = time.monotonic()

    # Seed before *anything* that can consume randomness.  In particular both
    # MILDataModule.setup() (samplers/workers) and MILTrainModule construction
    # (parameter initialisation) must be downstream of the nominal run seed.
    # Seeding after module construction makes a directory named ``seed43``
    # impossible to reproduce even when later dropout RNG happens to be stable.
    L.seed_everything(int(t.seed), workers=True)

    # ── Compute refit epochs ─────────────────────────────────────────────
    refit_epoch_rule = OmegaConf.select(
        cfg,
        "training.refit_epoch_rule",
        default="p75",
    )
    refit_epochs = _compute_refit_epochs(
        all_fold_metrics,
        rule=refit_epoch_rule,
        fallback_epochs=t.max_epochs,
    )

    # ── Build DataModule in refit mode ───────────────────────────────────
    from oceanpath.config import FoundationPaths

    paths = FoundationPaths.from_config(cfg)
    configured_workers = OmegaConf.select(cfg, "training.num_workers", default=None)
    num_workers = (
        int(cfg.platform.num_workers)
        if configured_workers in (None, "null")
        else int(configured_workers)
    )
    configured_prefetch = OmegaConf.select(cfg, "training.prefetch_factor", default=None)
    prefetch_factor = (
        int(OmegaConf.select(cfg, "platform.prefetch_factor", default=2))
        if configured_prefetch in (None, "null")
        else int(configured_prefetch)
    )

    datamodule = MILDataModule(
        feature_dir=str(paths.feature_h5_dir),
        splits_dir=str(paths.splits_dir),
        csv_path=str(paths.manifest_path),
        label_column=cfg.data.label_columns[0],
        filename_column=cfg.data.filename_column,
        scheme=cfg.splits.scheme,
        fold=0,  # ignored in refit_mode
        batch_size=t.batch_size,
        max_instances=t.max_instances,
        dataset_max_instances=OmegaConf.select(
            cfg,
            "training.dataset_max_instances",
            default=None,
        ),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=OmegaConf.select(cfg, "training.pin_memory", default=None),
        persistent_workers=OmegaConf.select(
            cfg, "training.persistent_workers", default=None
        ),
        class_weighted_sampling=t.class_weighted_sampling,
        instance_dropout=t.instance_dropout,
        feature_noise_std=t.feature_noise_std,
        cache_size_mb=0,  # never cache training data
        return_coords=t.return_coords,
        verify_splits=False,  # already verified during CV
        use_preallocated_collator=t.use_preallocated_collator,
        cap_strategy=OmegaConf.select(cfg, "training.cap_strategy", default="random"),
        cap_grid_size=OmegaConf.select(cfg, "training.cap_grid_size", default=32),
        length_bucket=OmegaConf.select(cfg, "training.length_bucket", default=False),
        length_bucket_size=OmegaConf.select(cfg, "training.length_bucket_size", default=64),
        force_float32=OmegaConf.select(cfg, "training.force_float32", default=True),
        packed_dir=OmegaConf.select(cfg, "training.packed_dir", default=None),
        verify_packed_source=OmegaConf.select(
            cfg, "training.verify_packed_source", default=True
        ),
        resident_device=OmegaConf.select(cfg, "training.resident_device", default=None),
        fixed_bag_size=OmegaConf.select(cfg, "training.fixed_bag_size", default=None),
        short_bag_policy=OmegaConf.select(cfg, "training.short_bag_policy", default="repeat"),
        drop_last=OmegaConf.select(cfg, "training.drop_last", default=False),
        # Must match the folds: a refit trained unweighted while every fold was
        # patient-weighted would not be the same recipe on more data.
        sample_weight_column=OmegaConf.select(
            cfg, "training.sample_weight_column", default=None
        ),
        # The TRAIN SAMPLER must match the folds too, and previously did not:
        # these four were never forwarded, so a study configured for
        # patient-level sampling silently refit under slide_uniform. That
        # reintroduces exactly the per-patient imbalance the setting exists to
        # remove — and does so only in the refit, which is the deployed object.
        train_sampling_strategy=OmegaConf.select(
            cfg, "training.train_sampling_strategy", default="slide_uniform"
        ),
        patient_column=OmegaConf.select(cfg, "data.patient_id_column", default=None),
        cohort_column=OmegaConf.select(cfg, "data.cohort_column", default=None),
        sampling_target_positive_prevalence=OmegaConf.select(
            cfg, "training.sampling_target_positive_prevalence", default=0.4
        ),
        sampling_seed=int(t.seed),
        refit_mode=True,  # <-- ALL slides → train, no val
    )
    datamodule.setup(stage="fit")
    if datamodule.feat_dim != int(cfg.encoder.feature_dim):
        raise ValueError(
            f"Feature dimension mismatch: H5 files contain D={datamodule.feat_dim}, "
            f"but encoder.feature_dim={cfg.encoder.feature_dim}"
        )
    logger.info(
        f"Refit dataset: {len(datamodule.train_dataset)} slides "
        f"({datamodule.train_dataset.get_label_counts()})"
    )

    configured_max_steps = OmegaConf.select(
        cfg, "training.refit_max_steps", default=None
    )
    refit_max_steps = (
        None
        if configured_max_steps in (None, "null")
        else int(configured_max_steps)
    )
    if refit_max_steps is not None and refit_max_steps <= 0:
        raise ValueError("training.refit_max_steps must be a positive integer")

    # Lightning's max_steps is the authoritative exact optimizer budget.  Give
    # max_epochs enough room to reach it; max_steps stops a final partial epoch
    # without the +/- one-epoch drift caused by round(budget / population).
    trainer_max_epochs = refit_epochs
    train_batches_per_epoch: int | None = None
    if refit_max_steps is not None:
        accumulate = int(t.accumulate_grad_batches)
        if accumulate <= 0:
            raise ValueError("training.accumulate_grad_batches must be positive")
        train_batches_per_epoch = len(datamodule.train_dataloader())
        optimizer_steps_per_epoch = int(np.ceil(train_batches_per_epoch / accumulate))
        if optimizer_steps_per_epoch <= 0:
            raise ValueError("refit train dataloader is empty")
        trainer_max_epochs = max(
            refit_epochs,
            int(np.ceil(refit_max_steps / optimizer_steps_per_epoch)),
        )

    # ── Build fresh model ────────────────────────────────────────────────
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    num_classes = datamodule.num_classes
    configured_classes = OmegaConf.select(cfg, "data.num_classes", default=None)
    if configured_classes not in (None, "null") and int(configured_classes) != num_classes:
        raise ValueError(
            f"Class-count mismatch: manifest has {num_classes} classes, "
            f"but data.num_classes={configured_classes}"
        )

    # Same head contract the folds use: loss_type=bce trains a SINGLE-logit
    # head, so a 2-class manifest still builds num_classes=1. Passing the raw
    # class count here made every bce refit fail at construction while the
    # folds themselves trained fine — the failure was swallowed by _safe_run,
    # so the run still reported "finalized".
    head_num_classes = 1 if str(t.loss_type) == "bce" and num_classes == 2 else num_classes

    # plateau needs val loss → fall back to cosine
    refit_scheduler = t.lr_scheduler
    if refit_scheduler == "plateau":
        refit_scheduler = "cosine"
        logger.info("Refit: plateau scheduler → cosine (no validation set)")

    refit_class_weights = _refit_class_weights(cfg, datamodule, num_classes)
    module = MILTrainModule(
        arch=cfg.model.arch,
        in_dim=cfg.encoder.feature_dim,
        num_classes=head_num_classes,
        model_cfg=model_cfg,
        lr=t.lr,
        weight_decay=t.weight_decay,
        lr_scheduler=refit_scheduler,
        warmup_epochs=t.warmup_epochs,
        max_epochs=trainer_max_epochs,  # scheduler horizon; max_steps is authoritative
        # When the exposure contract is in optimizer steps, the learning-rate
        # trajectory must use that same clock.  An epoch-wise cosine otherwise
        # changes phase at different optimizer steps as source N changes.
        lr_scheduler_interval="step" if refit_max_steps is not None else "epoch",
        lr_scheduler_total_steps=refit_max_steps,
        loss_type=t.loss_type,
        class_weights=refit_class_weights,
        focal_gamma=t.focal_gamma,
        monitor_metric=t.monitor_metric,
        monitor_mode=t.monitor_mode,
        canary_interval=t.canary_interval,
        compile_model=t.compile_model,
        freeze_aggregator=t.freeze_aggregator,
        collect_embeddings=False,  # nothing to collect
        aggregator_weights_path=OmegaConf.select(
            cfg, "training.aggregator_weights_path", default=None
        ),  # ← ADD
    )

    # ── Trainer — NO early stopping, NO val monitoring ───────────────────
    refit_dir = final_dir / "refit"
    refit_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    # Exact-step study refits are normally launched as captured subprocesses.
    # Rich's per-batch redraws make those provenance logs enormous and obscure
    # the useful completion record, so keep the progress UI only for ordinary
    # interactive epoch-based refits.  This does not change trainer semantics.
    if refit_max_steps is None:
        with contextlib.suppress(Exception):
            callbacks.append(RichProgressBar())

    trainer = L.Trainer(
        max_epochs=trainer_max_epochs,
        max_steps=refit_max_steps if refit_max_steps is not None else -1,
        accelerator=cfg.platform.accelerator,
        devices=cfg.platform.devices,
        strategy=cfg.platform.strategy,
        precision=cfg.platform.precision,
        callbacks=callbacks,
        logger=False,  # no W&B for refit
        gradient_clip_val=t.gradient_clip_val,
        accumulate_grad_batches=t.accumulate_grad_batches,
        deterministic=t.deterministic,
        default_root_dir=str(refit_dir),
        enable_checkpointing=False,  # we save manually at the end
        enable_progress_bar=refit_max_steps is None,
        log_every_n_steps=1,
    )

    # ── Train ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        f"  REFIT — {len(datamodule.train_dataset)} slides, "
        + (
            f"exactly {refit_max_steps} optimizer steps "
            f"(epoch ceiling={trainer_max_epochs})"
            if refit_max_steps is not None
            else f"{refit_epochs} epochs (rule={refit_epoch_rule})"
        )
    )
    logger.info("=" * 60)

    trainer.fit(module, train_dataloaders=datamodule.train_dataloader())
    actual_optimizer_steps = int(trainer.global_step)
    if refit_max_steps is not None and actual_optimizer_steps != refit_max_steps:
        raise RuntimeError(
            "Refit optimizer-step budget mismatch: "
            f"expected {refit_max_steps}, observed {actual_optimizer_steps}"
        )
    final_learning_rates = [
        float(group["lr"])
        for optimizer in trainer.optimizers
        for group in optimizer.param_groups
    ]

    # ── Save final-epoch checkpoint ──────────────────────────────────────
    final_ckpt = refit_dir / "model.ckpt"
    trainer.save_checkpoint(str(final_ckpt))

    elapsed = time.monotonic() - start_time

    info = {
        "strategy": "refit",
        "refit_epochs": refit_epochs,
        "refit_epoch_rule": refit_epoch_rule,
        "trainer_max_epochs": trainer_max_epochs,
        "refit_max_steps": refit_max_steps,
        "actual_optimizer_steps": actual_optimizer_steps,
        "train_batches_per_epoch": train_batches_per_epoch,
        "batch_size": int(t.batch_size),
        "accumulate_grad_batches": int(t.accumulate_grad_batches),
        "seed": int(t.seed),
        "sampling_seed": int(t.seed),
        "train_sampling_strategy": str(
            OmegaConf.select(
                cfg, "training.train_sampling_strategy", default="slide_uniform"
            )
        ),
        "sample_weight_column": OmegaConf.select(
            cfg, "training.sample_weight_column", default=None
        ),
        "class_weights": refit_class_weights,
        "dataset_max_instances": OmegaConf.select(
            cfg, "training.dataset_max_instances", default=None
        ),
        "max_instances": OmegaConf.select(
            cfg, "training.max_instances", default=None
        ),
        "eval_full_bags": bool(
            OmegaConf.select(cfg, "training.eval_full_bags", default=False)
        ),
        "training_sampling": datamodule.training_sampling_summary,
        "fold_best_epochs": [fm.get("best_epoch") for fm in all_fold_metrics],
        "n_train_slides": len(datamodule.train_dataset),
        "label_counts": datamodule.train_dataset.get_label_counts(),
        "lr_scheduler": refit_scheduler,
        "lr_scheduler_interval": (
            "step" if refit_max_steps is not None else "epoch"
        ),
        "lr_scheduler_total_steps": refit_max_steps,
        "final_learning_rates": final_learning_rates,
        "elapsed_seconds": round(elapsed, 1),
        "model_path": str(final_ckpt),
    }
    _write_info(refit_dir, info)
    logger.info(
        "refit: %s, %.0fs → %s",
        (
            f"{actual_optimizer_steps} optimizer steps"
            if refit_max_steps is not None
            else f"{refit_epochs} epochs"
        ),
        elapsed,
        final_ckpt,
    )

    # Cleanup
    del module, trainer, datamodule
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return info


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _compute_refit_epochs(
    all_fold_metrics: list[dict],
    rule: str = "p75",
    fallback_epochs: int = 30,
) -> int:
    """
    Determine how many epochs to train the refit model.

    Uses the best_epoch from each fold (epoch at which the best checkpoint
    was saved — NOT the stopped_epoch).

    Rules:
      p75    — 75th percentile of fold best_epochs (default, matches original)
      median — median
      max    — maximum across folds
      mean   — mean (rounded up)
    """
    epochs = [
        fm["best_epoch"]
        for fm in all_fold_metrics
        if fm.get("best_epoch") is not None
        and isinstance(fm["best_epoch"], (int, float))
        and fm["best_epoch"] > 0
    ]
    if not epochs:
        logger.warning(
            f"No valid best_epoch in fold metrics — falling back to {fallback_epochs} epochs"
        )
        return fallback_epochs

    rules = {
        "p75": lambda e: int(np.ceil(np.percentile(e, 75))),
        "median": lambda e: int(np.ceil(np.median(e))),
        "max": lambda e: max(e),
        "mean": lambda e: int(np.ceil(np.mean(e))),
    }
    if rule not in rules:
        raise ValueError(f"Unknown refit_epoch_rule '{rule}'. Choose from {list(rules)}")

    n = max(1, rules[rule](epochs))
    logger.info(f"Refit epochs: {n} (rule={rule}, fold best_epochs={epochs})")
    return n


def _get_all_train_slide_ids(cfg: DictConfig) -> list[str]:
    """
    Collect ALL non-test slide IDs from the splits (union across all folds).

    For kfold:         all slides
    For oof_kfold:     all slides (every fold is the test fold exactly once,
                       so nothing is globally held out)
    For holdout:       train + val (excludes test)
    For custom_kfold:  fold >= 0 (fold == -1 is test)
    For monte_carlo:   union of train+val from repeat 0
    """
    from oceanpath.config import FoundationPaths
    from oceanpath.splitting import SUPPORTED_SPLIT_SCHEMES, load_splits

    splits_dir = FoundationPaths.from_config(cfg).splits_dir
    splits_df = load_splits(str(splits_dir), verify=False)
    scheme = cfg.splits.scheme

    if scheme in ("kfold", "oof_kfold", "predefined_oof_kfold"):
        return splits_df["slide_id"].tolist()

    if scheme in ("holdout", "custom_holdout"):
        return splits_df.loc[splits_df["split"].isin(["train", "val"]), "slide_id"].tolist()

    if scheme in ("custom_kfold",):
        return splits_df.loc[splits_df["fold"] >= 0, "slide_id"].tolist()

    if scheme == "monte_carlo":
        r0 = splits_df[splits_df["repeat"] == 0]
        return r0.loc[r0["split"].isin(["train", "val"]), "slide_id"].tolist()

    raise ValueError(
        f"Unknown split scheme '{scheme}'. Must be one of: {list(SUPPORTED_SPLIT_SCHEMES)}"
    )


def parse_best_epoch_from_checkpoint(ckpt_path: str) -> int | None:
    """
    Extract epoch number from a Lightning checkpoint filename.

    Handles:  best-epoch=5-val/loss=0.1234.ckpt  →  5
              epoch=12-step=500.ckpt              →  12
    """
    if not ckpt_path:
        return None
    match = re.search(r"epoch=(\d+)", Path(ckpt_path).stem)
    return int(match.group(1)) + 1 if match else None


def _write_info(dest_dir: Path, info: dict) -> None:
    """Write info.json atomically."""
    path = dest_dir / "info.json"
    path.write_text(json.dumps(info, indent=2, default=str))


def _print_summary(results: dict) -> None:
    """Print a human-readable summary of finalization results."""
    print(f"\n{'=' * 60}")
    print("  Finalization Summary")
    print(f"{'=' * 60}")

    for name in ("best_fold", "ensemble", "refit"):
        info = results.get(name, {})
        if "error" in info:
            print(f"  {name:12s}: FAILED — {info['error']}")
        elif name == "best_fold":
            print(
                f"  {name:12s}: fold {info.get('source_fold')} "
                f"({info.get('monitor_metric')}={info.get('best_score', 0):.4f}, "
                f"epoch={info.get('best_epoch')})"
            )
        elif name == "ensemble":
            print(f"  {name:12s}: {info.get('n_checkpoints')}/{info.get('n_folds')} folds")
        elif name == "refit":
            print(
                f"  {name:12s}: {info.get('refit_epochs')} epochs, "
                f"{info.get('n_train_slides')} slides, "
                f"{info.get('elapsed_seconds', 0):.0f}s"
            )

    print(f"{'=' * 60}\n")
