"""Masked multi-target training for the flat shared-trunk ABMIL model.

Batch contract
--------------
``labels`` and ``label_mask`` are both ``[B, T]`` in ``target_names`` order.
Known labels must be binary.  Unknown entries may contain any placeholder
(including NaN) because they are removed by ``label_mask`` before BCE.

The loss first averages within every active target, then averages those target
losses equally.  A well-measured endpoint therefore cannot dominate merely by
having fewer missing labels than another endpoint.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from oceanpath.models.base import MILOutput
from oceanpath.models.flat_multitarget import FlatMultiTargetClassifier

logger = logging.getLogger(__name__)


def derive_pos_weights(
    labels: torch.Tensor | np.ndarray,
    label_mask: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Derive ``N_negative / N_positive`` independently for every target.

    Call this only on the concrete training fold.  A target without both
    classes raises rather than leaking a weight from validation/test data or
    silently producing an infinite weight.
    """
    y = torch.as_tensor(labels, dtype=torch.float32)
    known = torch.as_tensor(label_mask, dtype=torch.bool)
    if y.ndim != 2 or known.shape != y.shape:
        raise ValueError(
            f"labels and label_mask must have the same [N, T] shape; "
            f"got {tuple(y.shape)} and {tuple(known.shape)}"
        )
    known = known & torch.isfinite(y)
    invalid = known & (y != 0) & (y != 1)
    if bool(invalid.any()):
        raise ValueError("known multi-target labels must be binary 0/1")
    positives = ((y == 1) & known).sum(dim=0)
    negatives = ((y == 0) & known).sum(dim=0)
    bad = torch.nonzero((positives == 0) | (negatives == 0), as_tuple=False).flatten()
    if len(bad):
        raise ValueError(
            "every training-fold target needs at least one positive and one negative; "
            f"invalid target indices={bad.tolist()}"
        )
    return negatives.float() / positives.float()


def masked_multitarget_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Masked, class-weighted BCE with equal contribution from active heads."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, T], got {tuple(logits.shape)}")
    if labels.shape != logits.shape or label_mask.shape != logits.shape:
        raise ValueError(
            "logits, labels, and label_mask must have identical [B, T] shapes; "
            f"got {tuple(logits.shape)}, {tuple(labels.shape)}, "
            f"{tuple(label_mask.shape)}"
        )
    if pos_weight.ndim != 1 or pos_weight.shape[0] != logits.shape[1]:
        raise ValueError(
            f"pos_weight must have shape [{logits.shape[1]}], got {tuple(pos_weight.shape)}"
        )

    logits_fp32 = logits.float().clamp(-100, 100)
    labels_fp32 = labels.float()
    known = label_mask.bool() & torch.isfinite(labels_fp32)
    invalid = known & (labels_fp32 != 0) & (labels_fp32 != 1)
    # Labels are validated once while building each fold. Retain the useful
    # direct-call guard on CPU without forcing a GPU->CPU synchronization on
    # every training step.
    if labels_fp32.device.type == "cpu" and bool(invalid.any()):
        raise ValueError("known multi-target labels must be binary 0/1")

    # Replacing unknown values before BCE is essential: NaN * 0 is still NaN.
    safe_labels = torch.where(known, labels_fp32, torch.zeros_like(labels_fp32))
    elementwise = F.binary_cross_entropy_with_logits(
        logits_fp32,
        safe_labels,
        pos_weight=pos_weight.to(device=logits.device, dtype=torch.float32),
        reduction="none",
    )
    known_float = known.to(elementwise.dtype)
    counts = known_float.sum(dim=0)
    active_heads = counts > 0
    per_head = (elementwise * known_float).sum(dim=0) / counts.clamp_min(1)
    # The clamp keeps an accidental all-unknown batch graph-connected but a
    # no-op. Production manifests reject such rows before training.
    active_float = active_heads.to(per_head.dtype)
    return (per_head * active_float).sum() / active_float.sum().clamp_min(1)


def compute_patient_level_aurocs(
    rows: Sequence[Mapping[str, Any]],
    target_names: Sequence[str],
    patient_map: Mapping[str, str] | None = None,
    score_aggregation: str = "mean_logit",
) -> dict[str, Any]:
    """Aggregate slide scores by patient and compute per-target AUROC.

    Only slides carrying a known ground-truth label for a target enter that
    target's patient aggregation.  Patient label conflicts fail closed.
    ``macro`` is the unweighted mean over targets whose patient set contains
    both classes; it is the intended early-stopping metric.  Legacy callers
    retain ``mean_logit``.  Protocols that define a patient's prediction as
    the arithmetic mean of slide probabilities must request
    ``mean_probability`` so checkpoint selection matches final evaluation.
    """
    if score_aggregation not in {"mean_logit", "mean_probability"}:
        raise ValueError(
            "score_aggregation must be 'mean_logit' or 'mean_probability', "
            f"got {score_aggregation!r}"
        )
    names = tuple(str(name) for name in target_names)
    frame = pd.DataFrame(rows)
    per_target: dict[str, dict[str, int | float]] = {}
    finite_aurocs: list[float] = []
    mapping = patient_map or {}

    if frame.empty:
        return {
            "per_target": {
                name: {"auroc": float("nan"), "n_patients": 0, "n_positive": 0}
                for name in names
            },
            "macro": 0.5,
            "n_valid_targets": 0,
        }

    if "patient_id" not in frame:
        frame["patient_id"] = frame["slide_id"].map(
            lambda slide_id: mapping.get(str(slide_id), str(slide_id))
        )

    for name in names:
        required = {f"label_{name}", f"label_mask_{name}", f"logit_{name}"}
        if score_aggregation == "mean_probability":
            required.add(f"prob_{name}")
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"prediction rows are missing columns for {name}: {missing}")
        selected = frame[frame[f"label_mask_{name}"].astype(bool)].copy()
        selected = selected[np.isfinite(selected[f"label_{name}"])]
        if selected.empty:
            per_target[name] = {"auroc": float("nan"), "n_patients": 0, "n_positive": 0}
            continue

        conflicts = selected.groupby("patient_id")[f"label_{name}"].nunique()
        bad = conflicts[conflicts > 1]
        if not bad.empty:
            raise ValueError(
                f"target {name} has conflicting labels for {len(bad)} patient(s); "
                f"examples={bad.index.astype(str).tolist()[:5]}"
            )
        score_column = f"logit_{name}"
        score_name = "mean_logit"
        if score_aggregation == "mean_probability":
            score_column = f"prob_{name}"
            score_name = "mean_probability"
        patients = selected.groupby("patient_id").agg(
            label=(f"label_{name}", "first"),
            **{score_name: (score_column, "mean")},
        )
        y = patients["label"].to_numpy(dtype=int)
        auroc = (
            float(roc_auc_score(y, patients[score_name].to_numpy()))
            if len(np.unique(y)) == 2
            else float("nan")
        )
        if np.isfinite(auroc):
            finite_aurocs.append(auroc)
        per_target[name] = {
            "auroc": auroc,
            "n_patients": int(len(patients)),
            "n_positive": int(y.sum()),
        }

    return {
        "per_target": per_target,
        "macro": float(np.mean(finite_aurocs)) if finite_aurocs else 0.5,
        "n_valid_targets": len(finite_aurocs),
    }


class FlatMultiTargetTrainModule(L.LightningModule):
    """Lightning module for masked multi-target binary ABMIL training."""

    def __init__(
        self,
        target_names: Sequence[str],
        pos_weights: Sequence[float],
        in_dim: int = 1024,
        model_cfg: Mapping[str, Any] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        lr_scheduler: str | None = "cosine",
        warmup_epochs: int = 0,
        max_epochs: int = 40,
        monitor_metric: str = "val/patient_auroc_macro",
        monitor_mode: str = "max",
        aggregator_lr: float | None = None,
        head_lr: float | None = None,
        patient_map: Mapping[str, str] | None = None,
        patient_score_aggregation: str = "mean_logit",
    ) -> None:
        super().__init__()
        names = tuple(str(name) for name in target_names)
        weights = tuple(float(value) for value in pos_weights)
        if len(weights) != len(names):
            raise ValueError(
                f"pos_weights length {len(weights)} does not match "
                f"target_names length {len(names)}"
            )
        if any(not np.isfinite(value) or value <= 0 for value in weights):
            raise ValueError("all pos_weights must be finite and positive")
        if patient_score_aggregation not in {"mean_logit", "mean_probability"}:
            raise ValueError(
                "patient_score_aggregation must be 'mean_logit' or "
                f"'mean_probability', got {patient_score_aggregation!r}"
            )

        cfg = dict(model_cfg or {})
        valid_model_keys = {
            "embed_dim",
            "num_fc_layers",
            "attn_dim",
            "gate",
            "dropout",
            "gradient_checkpointing",
        }
        model_kwargs = {key: value for key, value in cfg.items() if key in valid_model_keys}
        self.model = FlatMultiTargetClassifier(
            in_dim=in_dim,
            target_names=names,
            **model_kwargs,
        )
        self.target_names = self.model.target_names
        self.patient_map = dict(patient_map or {})
        self.register_buffer(
            "pos_weight",
            torch.tensor(weights, dtype=torch.float32),
            persistent=True,
        )

        self.save_hyperparameters(
            {
                "target_names": list(self.target_names),
                "pos_weights": list(weights),
                "in_dim": int(in_dim),
                "model_cfg": cfg,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "lr_scheduler": lr_scheduler,
                "warmup_epochs": int(warmup_epochs),
                "max_epochs": int(max_epochs),
                "monitor_metric": str(monitor_metric),
                "monitor_mode": str(monitor_mode),
                "aggregator_lr": aggregator_lr,
                "head_lr": head_lr,
                "patient_score_aggregation": patient_score_aggregation,
            }
        )

        self._val_predictions: list[dict[str, Any]] = []
        self._test_predictions: list[dict[str, Any]] = []
        self._last_val_patient_metrics: dict[str, Any] | None = None
        self._last_test_patient_metrics: dict[str, Any] | None = None

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> MILOutput:
        return self.model(
            features,
            mask=mask,
            coords=coords,
            return_attention=return_attention,
        )

    def _param_dtype(self) -> torch.dtype:
        for parameter in self.model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return torch.float32

    def _autocast_dtype(self) -> torch.dtype | None:
        trainer = self._trainer
        if trainer is None:
            return None
        precision = str(getattr(trainer.precision_plugin, "precision", "32-true"))
        if precision.startswith("16"):
            return torch.float16
        if precision.startswith("bf16"):
            return torch.bfloat16
        return None

    def _forward_model(self, batch: Mapping[str, Any]) -> MILOutput:
        """Preserve bf16 throughput and retain the existing fp16 ABMIL guard."""
        autocast_dtype = self._autocast_dtype()
        if autocast_dtype is torch.float16:
            device_type = batch["features"].device.type
            with torch.amp.autocast(device_type=device_type, enabled=False):
                return self.model(
                    batch["features"].float(),
                    mask=(
                        batch["mask"].bool() if batch.get("mask") is not None else None
                    ),
                    coords=batch.get("coords"),
                )

        features = batch["features"]
        if autocast_dtype is None:
            features = features.to(self._param_dtype())
        return self.model(
            features,
            mask=batch.get("mask"),
            coords=batch.get("coords"),
        )

    def _loss(self, logits: torch.Tensor, batch: Mapping[str, Any]) -> torch.Tensor:
        if "label_mask" not in batch:
            raise KeyError("multi-target batches require label_mask")
        return masked_multitarget_bce(
            logits,
            batch["labels"],
            batch["label_mask"],
            self.pos_weight,
        )

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        output = self._forward_model(batch)
        loss = self._loss(output.logits, batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(batch["labels"].shape[0]),
        )
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        output = self._forward_model(batch)
        loss = self._loss(output.logits, batch)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=int(batch["labels"].shape[0]),
        )
        self._accumulate_predictions(self._val_predictions, output, batch)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        output = self._forward_model(batch)
        loss = self._loss(output.logits, batch)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=int(batch["labels"].shape[0]),
        )
        self._accumulate_predictions(self._test_predictions, output, batch)

    def _accumulate_predictions(
        self,
        accumulator: list[dict[str, Any]],
        output: MILOutput,
        batch: Mapping[str, Any],
    ) -> None:
        logits = output.logits.detach().float().cpu().numpy()
        probabilities = torch.sigmoid(output.logits.detach().float()).cpu().numpy()
        labels = batch["labels"].detach().float().cpu().numpy()
        known = batch["label_mask"].detach().bool().cpu().numpy()
        if logits.shape != labels.shape or known.shape != labels.shape:
            raise ValueError("prediction logits, labels, and masks have different shapes")

        for row_index, slide_id_value in enumerate(batch["slide_ids"]):
            slide_id = str(slide_id_value)
            row: dict[str, Any] = {
                "slide_id": slide_id,
                "patient_id": self.patient_map.get(slide_id, slide_id),
            }
            for target_index, name in enumerate(self.target_names):
                is_known = bool(known[row_index, target_index]) and bool(
                    np.isfinite(labels[row_index, target_index])
                )
                row[f"label_mask_{name}"] = is_known
                row[f"label_{name}"] = (
                    float(labels[row_index, target_index]) if is_known else float("nan")
                )
                # Store the native model score directly. Never reconstruct it
                # from sigmoid probabilities, which saturate for extreme logits.
                row[f"logit_{name}"] = float(logits[row_index, target_index])
                row[f"prob_{name}"] = float(probabilities[row_index, target_index])
            accumulator.append(row)

    def on_validation_epoch_start(self) -> None:
        self._val_predictions.clear()

    def on_test_epoch_start(self) -> None:
        self._test_predictions.clear()

    def on_validation_epoch_end(self) -> None:
        self._val_predictions = self._gather_objects_across_ranks(self._val_predictions)
        summary = compute_patient_level_aurocs(
            self._val_predictions,
            self.target_names,
            self.patient_map,
            score_aggregation=str(self.hparams.patient_score_aggregation),
        )
        self._last_val_patient_metrics = summary
        self._log_patient_aurocs("val", summary, progress_bar=True)

    def on_test_epoch_end(self) -> None:
        self._test_predictions = self._gather_objects_across_ranks(self._test_predictions)
        summary = compute_patient_level_aurocs(
            self._test_predictions,
            self.target_names,
            self.patient_map,
            score_aggregation=str(self.hparams.patient_score_aggregation),
        )
        self._last_test_patient_metrics = summary
        self._log_patient_aurocs("test", summary, progress_bar=False)

    def _log_patient_aurocs(
        self,
        prefix: str,
        summary: Mapping[str, Any],
        progress_bar: bool,
    ) -> None:
        for name in self.target_names:
            value = float(summary["per_target"][name]["auroc"])
            self.log(
                f"{prefix}/patient_auroc/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        self.log(
            f"{prefix}/patient_auroc_macro",
            float(summary["macro"]),
            on_step=False,
            on_epoch=True,
            prog_bar=progress_bar,
            sync_dist=False,
        )

    @staticmethod
    def _gather_objects_across_ranks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return items
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return items
        gathered: list[list[dict[str, Any]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, items)
        flat = [row for shard in gathered if shard is not None for row in shard]
        deduplicated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in flat:
            slide_id = str(row["slide_id"])
            if slide_id in seen:
                continue
            seen.add(slide_id)
            deduplicated.append(row)
        return deduplicated

    def save_predictions(self, output_dir: str | Path, prefix: str = "val") -> Path | None:
        rows = self._val_predictions if prefix == "val" else self._test_predictions
        if not rows:
            return None
        output_path = Path(output_dir) / f"preds_{prefix}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(output_path, index=False, engine="pyarrow")
        return output_path

    def configure_optimizers(self) -> Any:
        base_lr = float(self.hparams.lr)
        aggregator_lr = self.hparams.aggregator_lr or base_lr
        head_lr = self.hparams.head_lr or base_lr
        if aggregator_lr != head_lr:
            parameters: list[dict[str, Any]] = [
                {"params": self.model.aggregator.parameters(), "lr": aggregator_lr},
                {"params": self.model.heads.parameters(), "lr": head_lr},
            ]
        else:
            parameters = [{"params": self.parameters(), "lr": base_lr}]
        optimizer = torch.optim.AdamW(parameters, weight_decay=float(self.hparams.weight_decay))

        scheduler_name = self.hparams.lr_scheduler
        if scheduler_name in (None, "none"):
            return optimizer
        if scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(self.hparams.monitor_mode),
                factor=0.5,
                patience=5,
                min_lr=1e-7,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": str(self.hparams.monitor_metric),
                    "interval": "epoch",
                },
            }
        if scheduler_name != "cosine":
            raise ValueError(f"unsupported lr_scheduler={scheduler_name!r}")

        warmup_epochs = int(self.hparams.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(self.hparams.max_epochs) - warmup_epochs),
            eta_min=1e-7,
        )
        if warmup_epochs <= 0:
            return {"optimizer": optimizer, "lr_scheduler": cosine}
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(1, warmup_epochs),
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


__all__ = [
    "FlatMultiTargetTrainModule",
    "compute_patient_level_aurocs",
    "derive_pos_weights",
    "masked_multitarget_bce",
]
