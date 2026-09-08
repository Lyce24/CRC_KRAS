"""Stage-wise losses, metrics, and Lightning support for PHARE.

PHARE models the deterministic molecular hierarchy

``V600E ⊆ BRAF`` and ``{G12D, G12C} ⊆ KRAS``.

The child heads estimate conditional probabilities only inside their
ground-truth parent-positive risk sets.  In particular, a wild-type parent is
*not* a negative example for a conditional child head.  Marginal child
probabilities are produced by the model's explicit factorisation
``P(child|x) = P(parent|x) P(child|parent,x)``.

Batch contract
--------------
``labels`` and ``label_mask`` are ``[B, 6]`` in :data:`TARGET_NAMES` order.
Known entries must be binary.  Unknown labels may be NaN, but their mask must
be false.  ``slide_ids`` is a length-B sequence; ``patient_ids`` is optional.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from oceanpath.models.phare import PHAREClassifier

PARENT_TARGETS = ("msi", "braf", "kras")
CHILD_TARGETS = ("v600e", "g12d", "g12c")
CHILD_TO_PARENT = {"v600e": "braf", "g12d": "kras", "g12c": "kras"}
TARGET_NAMES = PARENT_TARGETS + CHILD_TARGETS

Stage = Literal["parent", "child"]


@dataclass(frozen=True)
class PHAREMasks:
    """Validated known-label masks and parent-positive child risk sets."""

    known: torch.Tensor
    parent: torch.Tensor
    conditional: torch.Tensor


@dataclass(frozen=True)
class PHAREPosWeights:
    """Training-fold-only positive weights in fixed head order."""

    parent: torch.Tensor
    conditional: torch.Tensor


@dataclass(frozen=True)
class PHARELoss:
    """Named loss components used for logging and tests."""

    total: torch.Tensor
    parent: torch.Tensor
    conditional: torch.Tensor
    residual_kl: torch.Tensor


def _as_label_tensors(
    labels: torch.Tensor | np.ndarray,
    label_mask: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.as_tensor(labels, dtype=torch.float32)
    declared_known = torch.as_tensor(label_mask, dtype=torch.bool, device=y.device)
    if y.ndim != 2 or y.shape[1] != len(TARGET_NAMES):
        raise ValueError(
            f"labels must be [N, {len(TARGET_NAMES)}] in TARGET_NAMES order; "
            f"got {tuple(y.shape)}"
        )
    if declared_known.shape != y.shape:
        raise ValueError(
            "labels and label_mask must have identical shapes; "
            f"got {tuple(y.shape)} and {tuple(declared_known.shape)}"
        )
    nonfinite_known = declared_known & ~torch.isfinite(y)
    if bool(nonfinite_known.any()):
        raise ValueError("label_mask=True requires a finite PHARE label")
    known = declared_known
    invalid = known & (y != 0) & (y != 1)
    if bool(invalid.any()):
        raise ValueError("known PHARE labels must be binary 0/1")
    return y, known


def derive_phare_masks(
    labels: torch.Tensor | np.ndarray,
    label_mask: torch.Tensor | np.ndarray,
) -> PHAREMasks:
    """Validate hierarchy and derive strict parent-positive child masks.

    A child is supervised if and only if both child and parent are known and
    the parent is positive.  A known child-positive/parent-negative pair is a
    label-contract violation and fails closed.
    """

    y, known = _as_label_tensors(labels, label_mask)
    parent_mask = known[:, : len(PARENT_TARGETS)]
    conditional_columns: list[torch.Tensor] = []
    for child_index, child in enumerate(CHILD_TARGETS, start=len(PARENT_TARGETS)):
        parent = CHILD_TO_PARENT[child]
        parent_index = TARGET_NAMES.index(parent)
        parent_known = known[:, parent_index]
        child_known = known[:, child_index]
        contradiction = (
            parent_known
            & child_known
            & y[:, parent_index].eq(0)
            & y[:, child_index].eq(1)
        )
        if bool(contradiction.any()):
            rows = torch.nonzero(contradiction, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"{child}-positive/{parent}-negative hierarchy violation at rows={rows[:10]}"
            )
        conditional_columns.append(
            child_known & parent_known & y[:, parent_index].eq(1)
        )
    return PHAREMasks(
        known=known,
        parent=parent_mask,
        conditional=torch.stack(conditional_columns, dim=1),
    )


def _derive_binary_pos_weights(
    labels: torch.Tensor,
    masks: torch.Tensor,
    names: Sequence[str],
) -> torch.Tensor:
    selected_labels = torch.where(masks, labels, torch.zeros_like(labels))
    positives = (selected_labels.eq(1) & masks).sum(dim=0)
    negatives = (selected_labels.eq(0) & masks).sum(dim=0)
    bad = torch.nonzero((positives == 0) | (negatives == 0), as_tuple=False).flatten()
    if len(bad):
        invalid_names = [str(names[index]) for index in bad.tolist()]
        raise ValueError(
            "every training-fold PHARE head needs at least one positive and one "
            f"negative in its risk set; invalid={invalid_names}"
        )
    return negatives.float() / positives.float()


def derive_phare_pos_weights(
    labels: torch.Tensor | np.ndarray,
    label_mask: torch.Tensor | np.ndarray,
) -> PHAREPosWeights:
    """Return ``N_negative / N_positive`` using only the training fold.

    Conditional weights use only parent-positive patients.  This deliberately
    excludes the many easy wild-type-parent rows from child class counts.
    """

    y, _known = _as_label_tensors(labels, label_mask)
    masks = derive_phare_masks(y, label_mask)
    return PHAREPosWeights(
        parent=_derive_binary_pos_weights(
            y[:, : len(PARENT_TARGETS)], masks.parent, PARENT_TARGETS
        ),
        conditional=_derive_binary_pos_weights(
            y[:, len(PARENT_TARGETS) :], masks.conditional, CHILD_TARGETS
        ),
    )


def _column(vector: torch.Tensor, *, name: str) -> torch.Tensor:
    if vector.ndim == 2 and vector.shape[1] == 1:
        return vector[:, 0]
    if vector.ndim != 1:
        raise ValueError(f"{name} must be [B] or [B, 1], got {tuple(vector.shape)}")
    return vector


def masked_equal_head_bce(
    logits: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    pos_weights: Mapping[str, torch.Tensor | float],
) -> torch.Tensor:
    """Compute masked BCE per head, then average active heads equally."""

    names = tuple(logits)
    if not names:
        raise ValueError("masked_equal_head_bce requires at least one logit head")
    if set(labels) != set(names) or set(masks) != set(names) or set(pos_weights) != set(names):
        raise ValueError("logits, labels, masks, and pos_weights need identical head names")

    losses: list[torch.Tensor] = []
    graph_anchor: torch.Tensor | None = None
    for name in names:
        head_logits = _column(logits[name], name=f"logits[{name}]").float().clamp(-100, 100)
        head_labels = _column(labels[name], name=f"labels[{name}]").to(
            device=head_logits.device, dtype=torch.float32
        )
        head_mask = _column(masks[name], name=f"masks[{name}]").to(
            device=head_logits.device, dtype=torch.bool
        )
        head_mask = head_mask & torch.isfinite(head_labels)
        invalid = head_mask & (head_labels != 0) & (head_labels != 1)
        if bool(invalid.any()):
            raise ValueError(f"known labels for {name} must be binary 0/1")
        if graph_anchor is None:
            graph_anchor = head_logits.sum() * 0.0
        if not bool(head_mask.any()):
            continue
        safe_labels = torch.where(head_mask, head_labels, torch.zeros_like(head_labels))
        weight = torch.as_tensor(
            pos_weights[name], device=head_logits.device, dtype=torch.float32
        ).reshape(())
        if not bool(torch.isfinite(weight)) or bool(weight <= 0):
            raise ValueError(f"pos_weight for {name} must be finite and positive")
        elementwise = F.binary_cross_entropy_with_logits(
            head_logits,
            safe_labels,
            pos_weight=weight,
            reduction="none",
        )
        losses.append(elementwise[head_mask].mean())
    if not losses:
        assert graph_anchor is not None
        return graph_anchor
    return torch.stack(losses).mean()


def _named_columns(
    tensor: torch.Tensor,
    names: Sequence[str],
) -> dict[str, torch.Tensor]:
    if tensor.ndim != 2 or tensor.shape[1] != len(names):
        raise ValueError(f"expected [B, {len(names)}], got {tuple(tensor.shape)}")
    return {name: tensor[:, index] for index, name in enumerate(names)}


def compute_patient_level_phare_aurocs(
    rows: Sequence[Mapping[str, Any]],
    patient_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate slide logits by patient and report parent/q/marginal AUROCs."""

    frame = pd.DataFrame(rows)
    mapping = patient_map or {}
    if frame.empty:
        empty_parent = {name: _empty_metric() for name in PARENT_TARGETS}
        empty_child = {name: _empty_metric() for name in CHILD_TARGETS}
        return {
            "parent": empty_parent,
            "conditional": empty_child,
            "marginal": {name: _empty_metric() for name in CHILD_TARGETS},
            "parent_macro": 0.5,
            "conditional_macro": 0.5,
        }
    if "patient_id" not in frame:
        frame["patient_id"] = frame["slide_id"].map(
            lambda slide_id: mapping.get(str(slide_id), str(slide_id))
        )

    parent: dict[str, dict[str, int | float]] = {}
    conditional: dict[str, dict[str, int | float]] = {}
    marginal: dict[str, dict[str, int | float]] = {}
    for name in PARENT_TARGETS:
        parent[name] = _patient_auroc(
            frame,
            label_column=f"label_{name}",
            mask_column=f"label_mask_{name}",
            logit_column=f"logit_{name}",
            metric_name=name,
        )
    for name in CHILD_TARGETS:
        conditional[name] = _patient_auroc(
            frame,
            label_column=f"label_{name}",
            mask_column=f"conditional_mask_{name}",
            logit_column=f"conditional_logit_{name}",
            metric_name=f"{name}_conditional",
        )
        marginal[name] = _patient_auroc(
            frame,
            label_column=f"label_{name}",
            mask_column=f"label_mask_{name}",
            logit_column=f"marginal_logit_{name}",
            metric_name=f"{name}_marginal",
        )
    return {
        "parent": parent,
        "conditional": conditional,
        "marginal": marginal,
        "parent_macro": _finite_macro(parent),
        "conditional_macro": _finite_macro(conditional),
    }


def _empty_metric() -> dict[str, int | float]:
    return {"auroc": float("nan"), "n_patients": 0, "n_positive": 0}


def _patient_auroc(
    frame: pd.DataFrame,
    *,
    label_column: str,
    mask_column: str,
    logit_column: str,
    metric_name: str,
) -> dict[str, int | float]:
    required = {"patient_id", label_column, mask_column, logit_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction rows are missing columns for {metric_name}: {missing}")
    selected = frame[frame[mask_column].astype(bool)].copy()
    selected = selected[
        np.isfinite(selected[label_column]) & np.isfinite(selected[logit_column])
    ]
    if selected.empty:
        return _empty_metric()
    conflicts = selected.groupby("patient_id")[label_column].nunique()
    bad = conflicts[conflicts > 1]
    if not bad.empty:
        raise ValueError(
            f"{metric_name} has conflicting labels for {len(bad)} patient(s); "
            f"examples={bad.index.astype(str).tolist()[:5]}"
        )
    patients = selected.groupby("patient_id").agg(
        label=(label_column, "first"),
        mean_logit=(logit_column, "mean"),
    )
    y = patients["label"].to_numpy(dtype=int)
    score = patients["mean_logit"].to_numpy(dtype=float)
    auroc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")
    return {"auroc": auroc, "n_patients": int(len(y)), "n_positive": int(y.sum())}


def _finite_macro(metrics: Mapping[str, Mapping[str, int | float]]) -> float:
    values = [float(block["auroc"]) for block in metrics.values()]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.5


class PHARETrainModule(L.LightningModule):
    """Two-stage PHARE Lightning module with hierarchy-safe supervision."""

    def __init__(
        self,
        stage: Stage,
        parent_pos_weights: Sequence[float],
        conditional_pos_weights: Sequence[float],
        in_dim: int = 1024,
        model_cfg: Mapping[str, Any] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        lr_scheduler: str | None = "cosine",
        warmup_epochs: int = 0,
        max_epochs: int = 40,
        monitor_metric: str | None = None,
        monitor_mode: str = "max",
        residual_kl_weight: float = 0.01,
        patient_map: Mapping[str, str] | None = None,
        parent_checkpoint: str | Path | None = None,
        model: PHAREClassifier | None = None,
    ) -> None:
        super().__init__()
        if stage not in ("parent", "child"):
            raise ValueError(f"stage must be 'parent' or 'child', got {stage!r}")
        parent_weights = tuple(float(value) for value in parent_pos_weights)
        child_weights = tuple(float(value) for value in conditional_pos_weights)
        _validate_weight_sequence(parent_weights, PARENT_TARGETS)
        _validate_weight_sequence(child_weights, CHILD_TARGETS)
        if not np.isfinite(residual_kl_weight) or residual_kl_weight < 0:
            raise ValueError("residual_kl_weight must be finite and non-negative")

        cfg = dict(model_cfg or {})
        self.model = model if model is not None else PHAREClassifier(in_dim=in_dim, **cfg)
        self.stage: Stage = stage
        self.patient_map = dict(patient_map or {})
        parent_checkpoint_string = (
            str(Path(parent_checkpoint).expanduser().resolve())
            if parent_checkpoint is not None
            else None
        )
        if stage == "child" and parent_checkpoint_string is not None:
            self._load_parent_checkpoint(Path(parent_checkpoint_string))
        self.register_buffer(
            "parent_pos_weight",
            torch.tensor(parent_weights, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "conditional_pos_weight",
            torch.tensor(child_weights, dtype=torch.float32),
            persistent=True,
        )
        default_monitor = (
            "val/patient_auroc_macro_parent"
            if stage == "parent"
            else "val/patient_auroc/g12d_conditional"
        )
        chosen_monitor = str(monitor_metric or default_monitor)
        self.save_hyperparameters(
            {
                "stage": stage,
                "parent_pos_weights": list(parent_weights),
                "conditional_pos_weights": list(child_weights),
                "in_dim": int(in_dim),
                "model_cfg": cfg,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "lr_scheduler": lr_scheduler,
                "warmup_epochs": int(warmup_epochs),
                "max_epochs": int(max_epochs),
                "monitor_metric": chosen_monitor,
                "monitor_mode": str(monitor_mode),
                "residual_kl_weight": float(residual_kl_weight),
                "parent_checkpoint": parent_checkpoint_string,
            }
        )
        self.monitor_metric = chosen_monitor
        self._configure_stage()

        self._val_predictions: list[dict[str, Any]] = []
        self._test_predictions: list[dict[str, Any]] = []
        self._last_val_patient_metrics: dict[str, Any] | None = None
        self._last_test_patient_metrics: dict[str, Any] | None = None

    def _load_parent_checkpoint(self, checkpoint_path: Path) -> None:
        """Load exactly the projection and parent heads from a stage-1 checkpoint."""

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PHARE parent checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("PHARE parent checkpoint must contain a mapping")
        raw_state = payload.get("state_dict", payload)
        if not isinstance(raw_state, Mapping):
            raise ValueError("PHARE parent checkpoint state_dict must be a mapping")

        prefixes = ("projection.", "parent_heads.")
        parent_state: dict[str, torch.Tensor] = {}
        for raw_name, value in raw_state.items():
            name = str(raw_name)
            if name.startswith("model."):
                name = name.removeprefix("model.")
            if name.startswith(prefixes):
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"checkpoint parameter {raw_name!r} is not a tensor")
                parent_state[name] = value

        expected = {
            name
            for name in self.model.state_dict()
            if name.startswith(("projection.", "parent_heads."))
        }
        missing = sorted(expected - set(parent_state))
        unexpected = sorted(set(parent_state) - expected)
        if missing or unexpected:
            raise ValueError(
                "parent checkpoint does not exactly cover PHARE projection/parent heads; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        incompatible = self.model.load_state_dict(parent_state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"unexpected keys while loading PHARE parents: {incompatible.unexpected_keys}"
            )

    def _configure_stage(self) -> None:
        configure = getattr(self.model, "set_training_stage", None)
        if not callable(configure):
            raise TypeError("PHAREClassifier must implement set_training_stage(stage)")
        configure(self.stage)
        if self.stage == "child":
            assertion = getattr(self.model, "assert_child_stage_frozen", None)
            if callable(assertion):
                assertion()
            else:
                self._assert_parent_projection_frozen()

    def _assert_parent_projection_frozen(self) -> None:
        found = False
        for attribute in ("projection", "shared_projection", "parent_heads"):
            module = getattr(self.model, attribute, None)
            if module is None:
                continue
            found = True
            trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
            if trainable:
                raise AssertionError(
                    f"child stage requires frozen {attribute}; trainable={trainable[:10]}"
                )
        if not found:
            raise AssertionError("could not locate PHARE parent/projection modules to verify freezing")

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> Any:
        return self.model(
            features,
            mask=mask,
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

    def _forward_model(self, batch: Mapping[str, Any]) -> Any:
        autocast_dtype = self._autocast_dtype()
        if autocast_dtype is torch.float16:
            with torch.amp.autocast(device_type=batch["features"].device.type, enabled=False):
                return self.model(
                    batch["features"].float(),
                    mask=batch.get("mask"),
                )
        features = batch["features"]
        if autocast_dtype is None:
            features = features.to(self._param_dtype())
        return self.model(features, mask=batch.get("mask"))

    def _loss(self, output: Any, batch: Mapping[str, Any]) -> PHARELoss:
        if "label_mask" not in batch:
            raise KeyError("PHARE batches require label_mask")
        labels = batch["labels"]
        masks = derive_phare_masks(labels, batch["label_mask"])
        y = labels.float()
        parent_labels = _named_columns(y[:, : len(PARENT_TARGETS)], PARENT_TARGETS)
        child_labels = _named_columns(y[:, len(PARENT_TARGETS) :], CHILD_TARGETS)
        parent_masks = _named_columns(masks.parent, PARENT_TARGETS)
        child_masks = _named_columns(masks.conditional, CHILD_TARGETS)
        parent_weights = {
            name: self.parent_pos_weight[index] for index, name in enumerate(PARENT_TARGETS)
        }
        child_weights = {
            name: self.conditional_pos_weight[index] for index, name in enumerate(CHILD_TARGETS)
        }

        zero = sum(value.sum() * 0.0 for value in output.parent_logits.values())
        if self.stage == "parent":
            parent_loss = masked_equal_head_bce(
                output.parent_logits, parent_labels, parent_masks, parent_weights
            )
            conditional_loss = zero
            residual_kl = zero
            total = parent_loss
        else:
            parent_loss = zero
            conditional_loss = masked_equal_head_bce(
                output.conditional_logits,
                child_labels,
                child_masks,
                child_weights,
            )
            values = getattr(output, "residual_kl", getattr(output, "child_kl", None))
            if values is None:
                raise ValueError("residual PHARE output is missing residual_kl")
            residual_kl = self._mean_residual_kl(values, child_masks)
            total = conditional_loss + float(self.hparams.residual_kl_weight) * residual_kl
        return PHARELoss(
            total=total,
            parent=parent_loss,
            conditional=conditional_loss,
            residual_kl=residual_kl,
        )

    def _mean_residual_kl(
        self,
        values: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        residual_mode = str(getattr(self.model, "child_mode", "residual")) == "residual"
        if not residual_mode or float(self.hparams.residual_kl_weight) == 0:
            anchor = next(iter(self.model.parameters()))
            return anchor.sum() * 0.0
        if set(CHILD_TARGETS) - set(values):
            missing = sorted(set(CHILD_TARGETS) - set(values))
            raise ValueError(f"residual PHARE output is missing KL terms: {missing}")
        if set(CHILD_TARGETS) - set(masks):
            missing = sorted(set(CHILD_TARGETS) - set(masks))
            raise ValueError(f"residual PHARE KL masks are missing children: {missing}")

        active: list[torch.Tensor] = []
        for name in CHILD_TARGETS:
            value = values[name].float()
            mask = masks[name].to(device=value.device, dtype=torch.bool)
            if value.shape != mask.shape:
                raise ValueError(
                    f"residual KL/mask shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(mask.shape)}"
                )
            if mask.any():
                active.append(value[mask].mean())
        if not active:
            anchor = next(iter(self.model.parameters()))
            return anchor.sum() * 0.0
        return torch.stack(active).mean()

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self._loss(self._forward_model(batch), batch)
        self._log_losses("train", loss, on_step=True)
        return loss.total

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        output = self._forward_model(batch)
        loss = self._loss(output, batch)
        self._log_losses("val", loss, on_step=False)
        self._accumulate_predictions(self._val_predictions, output, batch)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        output = self._forward_model(batch)
        loss = self._loss(output, batch)
        self._log_losses("test", loss, on_step=False)
        self._accumulate_predictions(self._test_predictions, output, batch)

    def _log_losses(self, prefix: str, loss: PHARELoss, *, on_step: bool) -> None:
        self.log(
            f"{prefix}/loss",
            loss.total,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            sync_dist=prefix != "train",
        )
        component = loss.parent if self.stage == "parent" else loss.conditional
        self.log(
            f"{prefix}/{self.stage}_bce",
            component,
            on_step=on_step,
            on_epoch=True,
            sync_dist=prefix != "train",
        )
        if self.stage == "child" and str(getattr(self.model, "child_mode", "residual")) == "residual":
            self.log(
                f"{prefix}/residual_kl",
                loss.residual_kl,
                on_step=on_step,
                on_epoch=True,
                sync_dist=prefix != "train",
            )

    def _accumulate_predictions(
        self,
        accumulator: list[dict[str, Any]],
        output: Any,
        batch: Mapping[str, Any],
    ) -> None:
        labels = batch["labels"].detach().float().cpu()
        masks = derive_phare_masks(labels, batch["label_mask"].detach().cpu())
        batch_size = labels.shape[0]
        slide_ids = batch["slide_ids"]
        patient_ids = batch.get("patient_ids")
        if len(slide_ids) != batch_size:
            raise ValueError("slide_ids length does not match PHARE batch size")

        parent_logits = _cpu_output_dict(output.parent_logits, PARENT_TARGETS, batch_size)
        conditional_logits = _cpu_output_dict(
            output.conditional_logits, CHILD_TARGETS, batch_size
        )
        marginal_logits = _cpu_output_dict(output.marginal_logits, CHILD_TARGETS, batch_size)
        for row_index, raw_slide_id in enumerate(slide_ids):
            slide_id = str(raw_slide_id)
            if patient_ids is not None:
                patient_id = str(patient_ids[row_index])
            else:
                patient_id = self.patient_map.get(slide_id, slide_id)
            row: dict[str, Any] = {"slide_id": slide_id, "patient_id": patient_id}
            for target_index, name in enumerate(TARGET_NAMES):
                is_known = bool(masks.known[row_index, target_index])
                row[f"label_{name}"] = (
                    float(labels[row_index, target_index]) if is_known else float("nan")
                )
                row[f"label_mask_{name}"] = is_known
            for name in PARENT_TARGETS:
                logit = float(parent_logits[name][row_index])
                row[f"logit_{name}"] = logit
                row[f"prob_{name}"] = float(torch.sigmoid(parent_logits[name][row_index]))
            for child_index, name in enumerate(CHILD_TARGETS):
                conditional_logit = float(conditional_logits[name][row_index])
                marginal_logit = float(marginal_logits[name][row_index])
                row[f"conditional_mask_{name}"] = bool(
                    masks.conditional[row_index, child_index]
                )
                row[f"conditional_logit_{name}"] = conditional_logit
                row[f"conditional_prob_{name}"] = float(
                    torch.sigmoid(conditional_logits[name][row_index])
                )
                row[f"marginal_logit_{name}"] = marginal_logit
                row[f"marginal_prob_{name}"] = float(
                    torch.sigmoid(marginal_logits[name][row_index])
                )
            accumulator.append(row)

    def on_validation_epoch_start(self) -> None:
        self._val_predictions.clear()

    def on_test_epoch_start(self) -> None:
        self._test_predictions.clear()

    def on_validation_epoch_end(self) -> None:
        self._val_predictions = self._gather_objects_across_ranks(self._val_predictions)
        summary = compute_patient_level_phare_aurocs(
            self._val_predictions,
            self.patient_map,
        )
        self._last_val_patient_metrics = summary
        self._log_patient_aurocs("val", summary, progress_bar=True)

    def on_test_epoch_end(self) -> None:
        self._test_predictions = self._gather_objects_across_ranks(self._test_predictions)
        summary = compute_patient_level_phare_aurocs(
            self._test_predictions,
            self.patient_map,
        )
        self._last_test_patient_metrics = summary
        self._log_patient_aurocs("test", summary, progress_bar=False)

    def _log_patient_aurocs(
        self,
        prefix: str,
        summary: Mapping[str, Any],
        *,
        progress_bar: bool,
    ) -> None:
        for name in PARENT_TARGETS:
            self.log(
                f"{prefix}/patient_auroc/{name}",
                float(summary["parent"][name]["auroc"]),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        for name in CHILD_TARGETS:
            self.log(
                f"{prefix}/patient_auroc/{name}_conditional",
                float(summary["conditional"][name]["auroc"]),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                f"{prefix}/patient_auroc/{name}_marginal",
                float(summary["marginal"][name]["auroc"]),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        self.log(
            f"{prefix}/patient_auroc_macro_parent",
            float(summary["parent_macro"]),
            on_step=False,
            on_epoch=True,
            prog_bar=progress_bar and self.stage == "parent",
            sync_dist=False,
        )
        self.log(
            f"{prefix}/patient_auroc_macro_conditional",
            float(summary["conditional_macro"]),
            on_step=False,
            on_epoch=True,
            prog_bar=progress_bar and self.stage == "child",
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
        path = Path(output_dir) / f"preds_{prefix}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False, engine="pyarrow")
        return path

    def configure_optimizers(self) -> Any:
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError(f"PHARE {self.stage} stage has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
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
                    "monitor": self.monitor_metric,
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


def _validate_weight_sequence(weights: Sequence[float], names: Sequence[str]) -> None:
    if len(weights) != len(names):
        raise ValueError(f"weights for {tuple(names)} must have length {len(names)}")
    if any(not np.isfinite(value) or value <= 0 for value in weights):
        raise ValueError(f"weights for {tuple(names)} must be finite and positive")


def _cpu_output_dict(
    values: Mapping[str, torch.Tensor],
    names: Sequence[str],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    missing = sorted(set(names) - set(values))
    if missing:
        raise ValueError(f"PHARE output is missing heads: {missing}")
    result: dict[str, torch.Tensor] = {}
    for name in names:
        column = _column(values[name], name=name).detach().float().cpu()
        if column.shape[0] != batch_size:
            raise ValueError(f"PHARE output {name} has batch size {column.shape[0]}, expected {batch_size}")
        result[name] = column
    return result


__all__ = [
    "CHILD_TARGETS",
    "CHILD_TO_PARENT",
    "PARENT_TARGETS",
    "TARGET_NAMES",
    "PHARELoss",
    "PHAREMasks",
    "PHAREPosWeights",
    "PHARETrainModule",
    "compute_patient_level_phare_aurocs",
    "derive_phare_masks",
    "derive_phare_pos_weights",
    "masked_equal_head_bce",
]
