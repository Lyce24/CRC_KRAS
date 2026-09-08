"""
Lightning training module for MIL classification.

NaN handling under AMP (the correctness issue that caused the crash):
═══════════════════════════════════════════════════════════════════════

Under 16-mixed precision, extreme logits overflow float16 → NaN loss.
The GradScaler's optimizer step requires inf checks from backward():

  ❌ return torch.tensor(0.0, requires_grad=True)
     This is a LEAF TENSOR disconnected from the model. backward()
     produces no parameter gradients. scaler.unscale_() finds nothing
     to check → "No inf checks were recorded" assertion crash.

  ❌ return None / skip backward
     Same problem — scaler.step() still runs, finds no inf checks.

  ✅ return torch.nan_to_num(loss, nan=0.0, ...)
     nan_to_num STAYS IN THE COMPUTATION GRAPH. backward() produces
     zero gradients for all parameters (not "no gradients"). scaler
     finds these, records inf checks, calls optimizer.step() with
     zero updates → harmless no-op, training continues.

Prevention: we also clamp logits to [-100,100] and cast to float32
before loss computation, which eliminates >99% of NaN cases.
"""

import logging
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC

from oceanpath.models import MILOutput, build_classifier

logger = logging.getLogger(__name__)


class MILTrainModule(L.LightningModule):
    def __init__(
        self,
        arch: str = "abmil",
        in_dim: int = 1024,
        num_classes: int = 2,
        model_cfg: dict | None = None,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        warmup_epochs: int = 0,
        max_epochs: int = 30,
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_total_steps: int | None = None,
        loss_type: str = "ce",
        class_weights: list[float] | None = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        monitor_metric: str = "val/loss",
        monitor_mode: str = "min",
        canary_interval: int = 200,
        compile_model: bool = False,
        freeze_aggregator: bool = False,
        collect_embeddings: bool = True,
        aggregator_weights_path: str | None = None,
        aggregator_lr: float | None = None,
        head_lr: float | None = None,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        final_lr_fraction: float = 0.01,
        patient_map: dict[str, str] | None = None,
    ):
        super().__init__()
        # patient_map is data plumbing, not a hyperparameter: keeping it out of
        # the checkpoint payload means load_from_checkpoint reconstructs the
        # module without it (re-attach it when patient metrics are needed).
        self.save_hyperparameters(ignore=["patient_map"])
        self.patient_map = patient_map

        self.model = build_classifier(
            arch=arch,
            in_dim=in_dim,
            num_classes=num_classes,
            model_cfg=model_cfg or {},
            freeze_aggregator=freeze_aggregator,
            aggregator_weights_path=aggregator_weights_path,
        )

        if compile_model and hasattr(torch, "compile"):
            # Default mode, NOT "reduce-overhead": reduce-overhead captures CUDA
            # graphs, and full-bag evaluation feeds a different bag length per
            # slide — every distinct N would force a re-capture (recompile
            # storm) or trip cudagraphs' static-shape assumptions. Default mode
            # compiles the static fixed-bag train shape once and switches to a
            # dynamic-shape kernel on the second distinct N it sees.
            logger.info("Applying torch.compile to model")
            self.model = torch.compile(self.model)

        self.loss_fn = self._build_loss(
            loss_type, num_classes, class_weights, focal_gamma, label_smoothing
        )

        self.num_classes = num_classes
        self.canary_interval = canary_interval
        self.collect_embeddings = collect_embeddings
        self._setup_metrics()

        self._val_predictions: list[dict] = []
        self._test_predictions: list[dict] = []
        self._val_embeddings: list[dict] = []
        self._test_embeddings: list[dict] = []

        # Device-resident so counting a NaN step costs no host sync.
        self.register_buffer("_nan_steps", torch.zeros((), dtype=torch.long), persistent=False)

    def _forward_model(self, batch: dict, return_attention: bool = False) -> MILOutput:
        """
        Forward helper with an fp32 stability path for attention MIL models.

        ABMIL-style aggregators are prone to mixed-precision instabilities on
        very large bags (e.g. 4k instances): attention logits are a sum over N
        terms, and in float16 that sum overflows at 65504. Running the forward
        in fp32 avoids those NaN-producing steps while still letting the
        trainer manage the outer optimization loop.

        The escape hatch is scoped to float16 specifically. bfloat16 carries
        the same exponent range as float32, so it cannot overflow where fp32
        would not, and forcing fp32 there would throw away roughly half the
        achievable throughput for no stability benefit.

        Storage dtype is also reconciled here. A float16 packed feature store
        keeps batches half-width all the way through the DataLoader queue and
        the PCIe copy, which is the point — but under ``32-true`` there is no
        autocast to promote them, and a Half activation would meet a Float
        weight. Under an active autocast the layers do the casting themselves,
        so nothing is done.
        """
        arch = str(self.hparams.get("arch", "")).lower()
        autocast_dtype = self._autocast_dtype()

        if arch in {"abmil", "mhabmil"} and autocast_dtype is torch.float16:
            device_type = batch["features"].device.type
            with torch.amp.autocast(device_type=device_type, enabled=False):
                return self.model(
                    batch["features"].float(),
                    mask=batch["mask"].float() if batch.get("mask") is not None else None,
                    coords=batch.get("coords"),
                    return_attention=return_attention,
                )

        features = batch["features"]
        if autocast_dtype is None:
            features = features.to(self._param_dtype())

        return self.model(
            features,
            mask=batch["mask"],
            coords=batch.get("coords"),
            return_attention=return_attention,
        )

    def _param_dtype(self) -> torch.dtype:
        """Floating-point dtype the model's own weights are held in."""
        for parameter in self.model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return torch.float32

    def _autocast_dtype(self) -> torch.dtype | None:
        """Precision the trainer is actually running under, or None if fp32.

        Reads the trainer's precision plugin rather than a config string so
        this stays correct however the run was launched. Outside a trainer
        (unit tests, manual forwards) autocast is not active, so returns None.
        """
        trainer = self._trainer  # avoid the raising `.trainer` property
        if trainer is None:
            return None
        precision = str(getattr(trainer.precision_plugin, "precision", "32-true"))
        if precision.startswith("16"):
            return torch.float16
        if precision.startswith("bf16"):
            return torch.bfloat16
        return None

    # ── Loss factory ──────────────────────────────────────────────────────

    @staticmethod
    def _build_loss(
        loss_type: str,
        num_classes: int,
        class_weights: list[float] | None,
        focal_gamma: float,
        label_smoothing: float = 0.0,
    ) -> nn.Module:
        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)

        if label_smoothing and loss_type != "ce":
            raise ValueError(
                f"label_smoothing is only implemented for loss_type='ce', got '{loss_type}'"
            )
        if loss_type == "ce":
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        if loss_type == "bce":
            if num_classes != 1:
                raise ValueError(
                    "loss_type='bce' requires a single-logit classifier head "
                    f"(num_classes=1), but got num_classes={num_classes}. "
                    "Use loss_type='ce' for the current two-logit binary setup."
                )
            pos_weight = None
            if weight is not None and len(weight) == 2:
                pos_weight = weight[1:2] / weight[0:1]
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if loss_type == "focal":
            return FocalLoss(gamma=focal_gamma, weight=weight)
        raise ValueError(f"Unknown loss_type '{loss_type}'. Use 'ce', 'bce', or 'focal'.")

    # ── Metrics ───────────────────────────────────────────────────────────

    def _setup_metrics(self) -> None:
        task = "binary" if self.num_classes <= 2 else "multiclass"
        kwargs: dict[str, str | int] = {"task": task}
        if task == "multiclass":
            kwargs["num_classes"] = self.num_classes
        # Balanced accuracy is macro recall, not macro accuracy.
        balanced_kwargs = {"task": "multiclass", "num_classes": max(2, self.num_classes)}

        # Train
        self.train_acc = Accuracy(**kwargs)

        # Val — full suite
        self.val_acc = Accuracy(**kwargs)
        self.val_f1 = F1Score(**kwargs, average="macro")
        self.val_balanced_acc = Recall(**balanced_kwargs, average="macro")
        self.val_precision = Precision(**kwargs, average="macro")
        self.val_recall = Recall(**kwargs, average="macro")

        if task == "binary":
            self.val_auroc = BinaryAUROC()
        else:
            self.val_auroc = MulticlassAUROC(num_classes=self.num_classes)

        # Test — full suite (separate instances to avoid state contamination)
        self.test_acc = Accuracy(**kwargs)
        self.test_f1 = F1Score(**kwargs, average="macro")
        self.test_balanced_acc = Recall(**balanced_kwargs, average="macro")
        self.test_precision = Precision(**kwargs, average="macro")
        self.test_recall = Recall(**kwargs, average="macro")

        if task == "binary":
            self.test_auroc = BinaryAUROC()
        else:
            self.test_auroc = MulticlassAUROC(num_classes=self.num_classes)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, features, mask=None, coords=None, return_attention=False):
        return self.model(
            features,
            mask=mask,
            coords=coords,
            return_attention=return_attention,
        )

    # ── Training step ─────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Attention weights are only read by the canary, so only pay for them
        # on canary steps. (Previously they were never requested here, which
        # made the dead-attention check unreachable.)
        canary_step = self.canary_interval > 0 and self.global_step % self.canary_interval == 0
        output = self._forward_model(batch, return_attention=canary_step)

        loss = self._compute_loss(output.logits, batch["labels"], batch.get("weights"))
        B = batch["labels"].shape[0]

        # ── AMP-safe NaN recovery, without a per-step device sync ──────────
        #
        # nan_to_num STAYS IN THE GRAPH (see module docstring), so applying it
        # unconditionally is a no-op for finite losses and turns a poisoned
        # step into a zero-gradient step. Doing it unconditionally means we
        # never have to ask "is this finite?" on the host.
        #
        # The previous `torch.isfinite(loss).item()` forced the CPU to block on
        # the GPU on EVERY step, which serialises the whole pipeline: the H2D
        # copy of batch n+1 could not overlap the compute of batch n. At
        # batch_size=1 that sync dominated the step. The NaN tally now lives on
        # the device and is read once per epoch.
        self._nan_steps += (~torch.isfinite(loss.detach())).long()
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        preds = self._get_preds(output.logits)
        self.train_acc(preds, batch["labels"])

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, batch_size=B)

        if canary_step:
            self._canary_check(loss, output)

        return loss

    def on_train_epoch_start(self) -> None:
        self._nan_steps.zero_()

    def on_train_epoch_end(self) -> None:
        # Single host sync per epoch instead of one per step.
        n_nan = int(self._nan_steps.item())
        if n_nan:
            logger.warning(
                "Epoch %d had %d non-finite training step(s); each was zeroed via "
                "nan_to_num (no-op update). Check LR / precision if this persists.",
                self.current_epoch,
                n_nan,
            )

    # ── Validation step ───────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        # No consumer reads output.extras on the validation path; requesting
        # attention here only materialised a [B, N] tensor per batch to throw
        # away. Attention maps for interpretability come from eval/inference.
        output = self._forward_model(batch, return_attention=False)

        loss = self._compute_loss(output.logits, batch["labels"])
        B = batch["labels"].shape[0]

        preds = self._get_preds(output.logits)
        probs = self._get_probs(output.logits)

        self.val_acc(preds, batch["labels"])
        self.val_f1(preds, batch["labels"])
        self.val_balanced_acc(preds, batch["labels"])
        self.val_precision(preds, batch["labels"])
        self.val_recall(preds, batch["labels"])
        self.val_auroc(probs, batch["labels"])

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/acc", self.val_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=B
        )
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, sync_dist=True, batch_size=B)
        self.log(
            "val/balanced_acc",
            self.val_balanced_acc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/auroc",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=B,
        )

        self._accumulate_predictions(self._val_predictions, output, batch, probs)
        if self.collect_embeddings:
            self._accumulate_embeddings(self._val_embeddings, output, batch)

    # ── Test step ─────────────────────────────────────────────────────────

    def test_step(self, batch: dict, batch_idx: int) -> None:
        output = self._forward_model(batch, return_attention=False)

        loss = self._compute_loss(output.logits, batch["labels"])
        B = batch["labels"].shape[0]

        preds = self._get_preds(output.logits)
        probs = self._get_probs(output.logits)

        self.test_acc(preds, batch["labels"])
        self.test_f1(preds, batch["labels"])
        self.test_balanced_acc(preds, batch["labels"])
        self.test_precision(preds, batch["labels"])
        self.test_recall(preds, batch["labels"])
        self.test_auroc(probs, batch["labels"])

        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=B)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, batch_size=B)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, batch_size=B)
        self.log(
            "test/balanced_acc", self.test_balanced_acc, on_step=False, on_epoch=True, batch_size=B
        )
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, batch_size=B)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, batch_size=B)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True, batch_size=B)

        self._accumulate_predictions(self._test_predictions, output, batch, probs)
        if self.collect_embeddings:
            self._accumulate_embeddings(self._test_embeddings, output, batch)

    # ── Accumulation ──────────────────────────────────────────────────────

    def _accumulate_predictions(
        self,
        acc: list,
        output: MILOutput,
        batch: dict,
        probs: torch.Tensor,
    ) -> None:
        probs_np = probs.detach().cpu().numpy()
        # Persist the model score before sigmoid/softmax.  Reconstructing a
        # logit from a stored probability is lossy once float32 sigmoid
        # saturates, which matters when several slides are averaged for one
        # patient.  For a two-column binary head, the exact scalar log-odds is
        # logit(class 1) - logit(class 0).
        logits_np = output.logits.detach().float().cpu().numpy()
        labels_np = batch["labels"].detach().cpu().numpy()

        for i, sid in enumerate(batch["slide_ids"]):
            row = {"slide_id": sid, "label": int(labels_np[i])}
            if probs_np.ndim == 1:
                row["prob_1"] = float(probs_np[i])
                if logits_np.ndim == 1 or logits_np.shape[-1] == 1:
                    row["logit"] = float(logits_np.reshape(-1)[i])
                elif logits_np.shape[-1] == 2:
                    row["logit"] = float(logits_np[i, 1] - logits_np[i, 0])
                else:  # pragma: no cover - guarded by _get_probs
                    raise RuntimeError(
                        "One-dimensional probabilities require a one- or two-logit head"
                    )
            else:
                for c in range(probs_np.shape[1]):
                    row[f"prob_{c}"] = float(probs_np[i, c])
                    row[f"logit_{c}"] = float(logits_np[i, c])
            acc.append(row)

    def _accumulate_embeddings(
        self,
        acc: list,
        output: MILOutput,
        batch: dict,
    ) -> None:
        # .float() is required, not cosmetic: under bf16-mixed the aggregator
        # returns a BFloat16 embedding and NumPy has no bfloat16 dtype, so
        # .numpy() raises "Got unsupported ScalarType BFloat16". Embeddings are
        # archived as float32 regardless of the compute precision.
        emb = output.slide_embedding.detach().float().cpu().numpy()
        for i, sid in enumerate(batch["slide_ids"]):
            acc.append({"slide_id": sid, "embedding": emb[i]})

    # ── Epoch hooks ───────────────────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        # Reset rank-local accumulators at the start of every val epoch so
        # gather + dedupe at epoch_end yields exactly one full pass and the
        # post-fit ``trainer.validate(...)`` consumer sees only the latest
        # epoch's predictions.
        self._val_predictions.clear()
        self._val_embeddings.clear()

    def on_test_epoch_start(self) -> None:
        self._test_predictions.clear()
        self._test_embeddings.clear()

    def on_validation_epoch_end(self) -> None:
        """Gather rank-local predictions/embeddings, then log a brief report."""
        self._val_predictions = self._gather_objects_across_ranks(
            self._val_predictions, dedupe_key="slide_id"
        )
        if self.collect_embeddings:
            self._val_embeddings = self._gather_objects_across_ranks(
                self._val_embeddings, dedupe_key="slide_id"
            )
        # Patient-level AUROC over the gathered epoch predictions. Logged from
        # the module hook so EarlyStopping/ModelCheckpoint (which fire later in
        # on_validation_end) can monitor "val/patient_auroc". Identical on all
        # ranks after the gather, so no sync_dist is needed.
        self.log(
            "val/patient_auroc",
            self._patient_level_auroc(self._val_predictions),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # The stopping / checkpoint criterion. Logged from the module hook so
        # EarlyStopping and ModelCheckpoint (which fire in on_validation_end)
        # can monitor it; identical on all ranks after the gather.
        self.log(
            "val/patient_loss",
            self._patient_level_loss(self._val_predictions),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # Only log during training (not during standalone trainer.validate() calls)
        if not self.trainer.sanity_checking:
            self._log_epoch_report("val")

    def on_test_epoch_end(self) -> None:
        """Gather rank-local predictions/embeddings, then log a brief report."""
        self._test_predictions = self._gather_objects_across_ranks(
            self._test_predictions, dedupe_key="slide_id"
        )
        if self.collect_embeddings:
            self._test_embeddings = self._gather_objects_across_ranks(
                self._test_embeddings, dedupe_key="slide_id"
            )
        self.log(
            "test/patient_auroc",
            self._patient_level_auroc(self._test_predictions),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test/patient_loss",
            self._patient_level_loss(self._test_predictions),
            on_step=False,
            on_epoch=True,
        )
        self._log_epoch_report("test")

    def _patient_frame(self, rows: list[dict]) -> pd.DataFrame | None:
        """Slide rows aggregated to one row per patient by MEAN LOGIT.

        This is the single aggregation used for every patient-level quantity —
        validation loss, AUROC/AUPRC, outer-fold test, and external inference —
        so a patient contributing several slides counts once everywhere.
        Averaging happens on the logit scale and any sigmoid/softmax is applied
        afterwards; averaging probabilities instead would weight saturated
        slides differently.
        """
        if not rows:
            return None
        frame = pd.DataFrame(rows)
        patient_map = self.patient_map or {}
        frame["patient_id"] = frame["slide_id"].map(lambda sid: patient_map.get(sid, sid))
        prob_cols = sorted(
            (c for c in frame.columns if c.startswith("prob_")),
            key=lambda c: int(c.split("_")[1]),
        )
        if prob_cols == ["prob_1"]:
            if "logit" not in frame:
                clipped = frame["prob_1"].clip(1e-6, 1.0 - 1e-6)
                frame["logit"] = np.log(clipped / (1.0 - clipped))
            value_cols = ["logit"]
        else:
            value_cols = [f"logit_{int(c.split('_')[1])}" for c in prob_cols]
            if not all(column in frame for column in value_cols):
                clipped = frame[prob_cols].clip(1e-12, 1.0)
                for probability_column, logit_column in zip(prob_cols, value_cols, strict=True):
                    frame[logit_column] = np.log(clipped[probability_column])
        grouped = frame.groupby("patient_id").agg(
            {"label": "max", **{column: "mean" for column in value_cols}}
        )
        grouped.attrs["value_cols"] = value_cols
        return grouped

    def _patient_level_loss(self, rows: list[dict]) -> float:
        """Unweighted BCE / CE over PATIENT-averaged logits.

        The early-stopping and checkpoint-selection criterion. Deliberately
        unweighted — no class weights, no per-slide weights — so the number
        compares across folds and cohorts whose prevalence differs, and so a
        reweighting decision cannot silently move the stopping point.

        Returns NaN on a degenerate epoch (no predictions), which Lightning's
        monitors treat as "no improvement" rather than crashing.
        """
        grouped = self._patient_frame(rows)
        if grouped is None or grouped.empty:
            return float("nan")
        value_cols = grouped.attrs["value_cols"]
        labels = torch.from_numpy(grouped["label"].to_numpy()).long()
        scores = torch.from_numpy(grouped[value_cols].to_numpy()).float()
        with torch.no_grad():
            if value_cols == ["logit"]:
                loss = F.binary_cross_entropy_with_logits(
                    scores.reshape(-1), labels.float(), reduction="mean"
                )
            else:
                loss = F.cross_entropy(scores, labels, reduction="mean")
        return float(loss)

    def _patient_level_auroc(self, rows: list[dict]) -> float:
        """AUROC over patients: mean slide logit per patient (§44 aggregation).

        Slides map to patients via ``self.patient_map``; without a map each
        slide is its own patient (slide-level AUROC). Binary heads use
        P(class 1); multiclass heads use macro one-vs-rest over per-class
        mean logits. Degenerate epochs (one class present, e.g. the sanity
        check) return 0.5 rather than crash the monitor.
        """
        if not rows:
            return 0.5
        frame = pd.DataFrame(rows)
        patient_map = self.patient_map or {}
        frame["patient_id"] = frame["slide_id"].map(lambda sid: patient_map.get(sid, sid))
        prob_cols = sorted(
            (c for c in frame.columns if c.startswith("prob_")),
            key=lambda c: int(c.split("_")[1]),
        )
        from sklearn.metrics import roc_auc_score

        try:
            if prob_cols == ["prob_1"]:
                score_column = "logit"
                if score_column not in frame:
                    # Backward compatibility for legacy prediction rows.
                    clipped = frame["prob_1"].clip(1e-6, 1.0 - 1e-6)
                    frame[score_column] = np.log(clipped / (1.0 - clipped))
                grouped = frame.groupby("patient_id").agg({"label": "max", score_column: "mean"})
                y = grouped["label"].to_numpy()
                if len(np.unique(y)) < 2:
                    return 0.5
                return float(roc_auc_score(y, grouped[score_column].to_numpy()))

            logit_cols = [f"logit_{int(c.split('_')[1])}" for c in prob_cols]
            if not all(column in frame for column in logit_cols):
                # Legacy multiclass rows stored probabilities only.  Their
                # centered log probabilities preserve softmax rankings.
                clipped = frame[prob_cols].clip(1e-12, 1.0)
                for probability_column, logit_column in zip(prob_cols, logit_cols, strict=True):
                    frame[logit_column] = np.log(clipped[probability_column])
            grouped = frame.groupby("patient_id").agg(
                {"label": "max", **{column: "mean" for column in logit_cols}}
            )
            y = grouped["label"].to_numpy()
            scores = grouped[logit_cols].to_numpy()
            exp = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            if len(np.unique(y)) < len(prob_cols):
                return 0.5
            return float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
        except ValueError:
            return 0.5

    @staticmethod
    def _gather_objects_across_ranks(
        items: list[dict],
        dedupe_key: str | None = None,
    ) -> list[dict]:
        """All-gather a list of picklable objects across DDP ranks.

        Returns the same list unchanged when DDP is not initialized. Under
        DDP, returns the union across ranks, optionally deduplicated by
        ``dedupe_key`` (needed because Lightning's eval ``DistributedSampler``
        pads with repeats so each rank sees the same count).
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return items
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return items
        gathered: list[list[dict] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, items)
        flat = [row for shard in gathered if shard is not None for row in shard]
        if dedupe_key is None:
            return flat
        seen: set = set()
        deduped: list[dict] = []
        for row in flat:
            key = row.get(dedupe_key)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _log_epoch_report(self, prefix: str) -> None:
        """Print a one-line metrics summary for val or test."""
        metrics = self.trainer.callback_metrics
        keys = ["loss", "auroc", "acc", "precision", "recall", "f1"]
        parts = []
        for k in keys:
            full_key = f"{prefix}/{k}"
            val = metrics.get(full_key)
            if val is not None:
                v = val.item() if isinstance(val, torch.Tensor) else val
                parts.append(f"{k}={v:.4f}")
        if parts:
            epoch = self.current_epoch
            report = ", ".join(parts)
            logger.info(f"[{prefix.upper()} epoch={epoch}] {report}")

    # ── Save predictions/embeddings ───────────────────────────────────────

    def save_predictions(self, output_dir: str, prefix: str = "val") -> str | None:
        acc = self._val_predictions if prefix == "val" else self._test_predictions
        if not acc:
            return None
        path = Path(output_dir) / f"preds_{prefix}.parquet"
        df = pd.DataFrame(acc)
        df.to_parquet(str(path), index=False, engine="pyarrow")
        logger.info(f"Saved {len(df)} predictions → {path}")
        acc.clear()
        return str(path)

    def save_embeddings(self, output_dir: str, prefix: str = "val") -> str | None:
        acc = self._val_embeddings if prefix == "val" else self._test_embeddings
        if not acc:
            return None
        path = Path(output_dir) / f"embeddings_{prefix}.npz"
        slide_ids = [r["slide_id"] for r in acc]
        embeddings = np.stack([r["embedding"] for r in acc])
        np.savez(str(path), slide_ids=slide_ids, embeddings=embeddings)
        logger.info(f"Saved {len(slide_ids)} embeddings → {path}")
        acc.clear()
        return str(path)

    # ── Loss computation ──────────────────────────────────────────────────

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Loss with logit clamping + float32 upcast for AMP stability.

        Clamping to [-100, 100] prevents float16 overflow in softmax.
        Upcasting to float32 avoids precision loss in cross-entropy.

        ``weights`` gives a per-slide multiplier (e.g. 1/n_slides for the
        slide's patient). It is applied as a weighted mean — dividing by the
        weight sum rather than the batch size — so the loss scale, and with it
        the effective learning rate, does not depend on how many multi-slide
        patients happen to land in a batch.
        """
        logits_safe = logits.float().clamp(-100, 100)
        if weights is None:
            if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
                # A single-logit head emits [B, 1]; BCE targets are [B].
                # Flatten the logits — feeding [B, 1] against [B] would
                # silently broadcast to [B, B] and train on garbage.
                return self.loss_fn(logits_safe.reshape(-1), labels.float().reshape(-1))
            return self.loss_fn(logits_safe, labels)

        per_sample = self._per_sample_loss(logits_safe, labels)
        weights = weights.to(device=per_sample.device, dtype=per_sample.dtype)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)

    def _per_sample_loss(self, logits_safe: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Unreduced loss, reusing the configured loss's class weighting."""
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            return F.binary_cross_entropy_with_logits(
                logits_safe.reshape(-1),
                labels.float().reshape(-1),
                pos_weight=self.loss_fn.pos_weight,
                reduction="none",
            )
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            return F.cross_entropy(
                logits_safe,
                labels,
                weight=self.loss_fn.weight,
                label_smoothing=self.loss_fn.label_smoothing,
                reduction="none",
            )
        raise ValueError(
            f"Per-slide loss weighting is not implemented for {type(self.loss_fn).__name__}; "
            "use loss_type='bce' or 'ce', or drop training.sample_weight_column"
        )

    def _get_preds(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1 or logits.shape[-1] == 1:
            # reshape(-1), not squeeze(-1): a [1]-shaped batch of pre-squeezed
            # single-logit outputs must stay [B]=[1], not collapse to a scalar.
            return (logits.reshape(-1) > 0).long()
        return logits.argmax(dim=-1)

    def _get_probs(self, logits: torch.Tensor) -> torch.Tensor:
        device_type = logits.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            if logits.ndim == 1 or logits.shape[-1] == 1:
                return torch.sigmoid(logits.float()).reshape(-1)
            if self.num_classes <= 2:
                # Binary with 2-column logits: return P(class=1)
                return F.softmax(logits.float(), dim=-1)[:, 1]
            return F.softmax(logits.float(), dim=-1)

    # ── Optimizer + scheduler ─────────────────────────────────────────────

    def configure_optimizers(self):
        base_lr = self.hparams.lr
        agg_lr = self.hparams.get("aggregator_lr", None) or base_lr
        h_lr = self.hparams.get("head_lr", None) or base_lr

        if hasattr(self.model, "aggregator") and (agg_lr != h_lr):
            agg_params = [p for p in self.model.aggregator.parameters() if p.requires_grad]
            agg_ids = {id(p) for p in agg_params}
            other_params = [
                p for p in self.parameters() if p.requires_grad and id(p) not in agg_ids
            ]
            param_groups = [
                {"params": agg_params, "lr": agg_lr},
                {"params": other_params, "lr": h_lr},
            ]
            logger.info("Differential LR: aggregator=%.2e, head=%.2e", agg_lr, h_lr)
        else:
            param_groups = [
                {"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr}
            ]

        betas = tuple(self.hparams.get("adam_betas", (0.9, 0.999)))
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
            betas=betas,
            eps=float(self.hparams.get("adam_eps", 1e-8)),
        )

        scheduler_name = self.hparams.get("lr_scheduler", "cosine")

        if scheduler_name == "none" or scheduler_name is None:
            return optimizer

        warmup_epochs = self.hparams.warmup_epochs
        scheduler_interval = str(
            self.hparams.get("lr_scheduler_interval", "epoch")
        )
        if scheduler_interval not in {"epoch", "step"}:
            raise ValueError(
                "lr_scheduler_interval must be 'epoch' or 'step', got "
                f"{scheduler_interval!r}"
            )
        if scheduler_interval == "step":
            if warmup_epochs:
                raise ValueError(
                    "step-wise scheduling requires warmup_epochs=0; a warmup "
                    "must be specified in optimizer steps before enabling both"
                )
            total_steps = self.hparams.get("lr_scheduler_total_steps", None)
            if total_steps is None or int(total_steps) <= 0:
                raise ValueError(
                    "lr_scheduler_total_steps must be positive for step-wise scheduling"
                )
            effective_units = int(total_steps)
        else:
            effective_units = max(1, self.hparams.max_epochs - warmup_epochs)

        if scheduler_name == "cosine":
            # Anneal to a FIXED FRACTION of the peak lr (default peak/100)
            # rather than to a fixed absolute floor: a hard 1e-7 floor means
            # different configurations in an lr sweep end at different
            # fractions of their own peak, so the low-lr arms are effectively
            # annealed less and the sweep compares two things at once.
            final_fraction = float(self.hparams.get("final_lr_fraction", 0.01))
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=effective_units,
                eta_min=base_lr * final_fraction,
            )
        elif scheduler_name == "plateau":
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.hparams.monitor_mode,
                factor=0.5,
                patience=5,
                min_lr=1e-7,
            )
            if warmup_epochs > 0:
                logger.warning(
                    "warmup_epochs>0 with plateau scheduler is not supported. Using plateau only."
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": plateau,
                    "monitor": self.hparams.monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif scheduler_name == "step":
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / max(1, warmup_epochs),
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = main_scheduler

        if scheduler_interval == "step":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ── Canary checks ─────────────────────────────────────────────────────

    def _canary_check(self, loss: torch.Tensor, output: MILOutput) -> None:
        step = self.global_step

        if not torch.isfinite(loss):
            logger.error(f"[CANARY] Non-finite loss at step {step}")

        attn = output.extras.get("attention_weights")
        if attn is not None and attn.std() < 1e-7:
            logger.warning(f"[CANARY] Dead attention at step {step} (std={attn.std():.2e})")

        try:
            for pg in self.trainer.optimizers[0].param_groups:
                if pg["lr"] < 1e-10:
                    logger.warning(f"[CANARY] Near-zero LR at step {step}: {pg['lr']:.2e}")
        except (AttributeError, IndexError):
            pass


# ── Focal Loss ────────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., ICCV 2017)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()
