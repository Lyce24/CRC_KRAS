"""Phenotype-hierarchical attention residual MIL (PHARE).

PHARE learns three parent phenotypes (MSI, BRAF, and KRAS) with independent
attention heads.  Allele children are conditional predictions inside their
known molecular parent and are converted to marginal probabilities by exact
factorisation, for example::

    P(G12D | x) = P(KRAS | x) * P(G12D | KRAS, x)

The child branch is deliberately gradient-isolated from the shared projection
and parent heads.  This makes the two-stage training interpretation explicit:
parent morphology is learned first, then frozen while the allele adapter is
learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from oceanpath.models.components import GlobalAttention, GlobalGatedAttention, create_mlp

PARENT_TARGETS = ("msi", "braf", "kras")
CHILD_TARGETS = ("v600e", "g12d", "g12c")
TARGETS = PARENT_TARGETS + CHILD_TARGETS
CHILD_TO_PARENT = {"v600e": "braf", "g12d": "kras", "g12c": "kras"}
CHILD_MODES = ("parent_only", "independent", "residual")

ChildMode = Literal["parent_only", "independent", "residual"]
TrainingStage = Literal["parent", "child", "joint"]


@dataclass
class PHAREOutput:
    """Rich output from :class:`PHAREClassifier`.

    All named dictionaries use lowercase target names.  Logits have shape
    ``[B]``; embeddings have shape ``[B, E]``; attention tensors have shape
    ``[B, N]``; and each KL divergence has shape ``[B]``.  ``logits`` is a
    canonical ``[B, 6]`` tensor in :data:`TARGETS` order.  Its parent columns
    are parent logits and its child columns are *marginal* logits.
    """

    logits: torch.Tensor
    parent_logits: dict[str, torch.Tensor]
    conditional_logits: dict[str, torch.Tensor]
    marginal_logits: dict[str, torch.Tensor]
    parent_embeddings: dict[str, torch.Tensor]
    child_embeddings: dict[str, torch.Tensor]
    residual_embeddings: dict[str, torch.Tensor]
    parent_attention_logits: dict[str, torch.Tensor]
    parent_attention_weights: dict[str, torch.Tensor]
    child_attention_logits: dict[str, torch.Tensor]
    child_attention_weights: dict[str, torch.Tensor]
    centered_deltas: dict[str, torch.Tensor]
    child_kl: dict[str, torch.Tensor]

    @property
    def logits_by_target(self) -> dict[str, torch.Tensor]:
        """Return named views over the canonical marginal logits tensor."""
        return {name: self.logits[:, index] for index, name in enumerate(TARGETS)}

    @property
    def extras(self) -> dict[str, Any]:
        """Expose diagnostic tensors in a familiar model-output namespace."""
        return {
            "parent_logits": self.parent_logits,
            "conditional_logits": self.conditional_logits,
            "marginal_logits": self.marginal_logits,
            "parent_embeddings": self.parent_embeddings,
            "child_embeddings": self.child_embeddings,
            "residual_embeddings": self.residual_embeddings,
            "parent_attention_logits": self.parent_attention_logits,
            "parent_attention_weights": self.parent_attention_weights,
            "child_attention_logits": self.child_attention_logits,
            "child_attention_weights": self.child_attention_weights,
            "centered_deltas": self.centered_deltas,
            "child_kl": self.child_kl,
            "residual_kl": self.child_kl,
        }

    @property
    def residual_kl(self) -> dict[str, torch.Tensor]:
        """Backward-compatible descriptive alias for per-child attention KL."""
        return self.child_kl


def factorized_marginal_logit(
    parent_logit: torch.Tensor,
    conditional_logit: torch.Tensor,
) -> torch.Tensor:
    """Return the stable logit of ``sigmoid(parent) * sigmoid(conditional)``.

    Computing the probability product first loses precision when either logit
    is extreme.  In odds form, the exact denominator is
    ``exp(-p) + exp(-q) + exp(-p-q)``, which can be evaluated with log-sum-exp.
    """
    if parent_logit.shape != conditional_logit.shape:
        raise ValueError(
            "parent and conditional logits must have identical shapes, got "
            f"{tuple(parent_logit.shape)} and {tuple(conditional_logit.shape)}"
        )
    terms = torch.stack(
        (-parent_logit, -conditional_logit, -parent_logit - conditional_logit),
        dim=0,
    )
    return -torch.logsumexp(terms, dim=0)


class _AttentionScorer(nn.Module):
    """One target-specific scalar attention scorer."""

    def __init__(
        self,
        embed_dim: int,
        attn_dim: int,
        gate: bool,
        dropout: float,
        *,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        attention_cls = GlobalGatedAttention if gate else GlobalAttention
        self.network = attention_cls(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1,
        )
        if zero_output:
            self.zero_output_layer()

    def zero_output_layer(self) -> None:
        """Zero only the final scalar-score layer."""
        if isinstance(self.network, GlobalGatedAttention):
            output_layer = self.network.attention_c
        else:
            output_layer = cast(nn.Linear, self.network.module[-1])
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.network(tiles).squeeze(-1)


class _ParentHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attn_dim: int,
        gate: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = _AttentionScorer(embed_dim, attn_dim, gate, dropout)
        self.classifier = nn.Linear(embed_dim, 1)


class _ParentOnlyChildHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(embed_dim, 1)


class _IndependentChildHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attn_dim: int,
        gate: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = _AttentionScorer(embed_dim, attn_dim, gate, dropout)
        self.classifier = nn.Linear(embed_dim, 1)


class _ResidualChildHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attn_dim: int,
        gate: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        # Only the residual score output starts at zero.  The base and residual
        # classifiers retain ordinary initialisation, avoiding a dead adapter.
        self.delta_attention = _AttentionScorer(
            embed_dim,
            attn_dim,
            gate,
            dropout,
            zero_output=True,
        )
        self.base_classifier = nn.Linear(embed_dim, 1)
        self.residual_classifier = nn.Linear(embed_dim, 1, bias=False)


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def _masked_attention(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask logits and run attention softmax in float32."""
    masked_logits = logits.float().masked_fill(~mask, float("-inf"))
    weights = F.softmax(masked_logits, dim=-1)
    return masked_logits, weights


def _pool(weights: torch.Tensor, tiles: torch.Tensor) -> torch.Tensor:
    return torch.bmm(weights.unsqueeze(1).to(tiles.dtype), tiles).squeeze(1)


def _center_over_valid(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    delta = delta.float()
    valid = mask.to(delta.dtype)
    mean = (delta * valid).sum(dim=-1, keepdim=True) / valid.sum(
        dim=-1, keepdim=True
    )
    return (delta - mean) * valid


def _attention_kl(
    child_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    child_weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute ``KL(child attention || parent attention)`` per bag."""
    child_log_weights = F.log_softmax(child_logits, dim=-1)
    parent_log_weights = F.log_softmax(parent_logits, dim=-1)
    terms = child_weights * (child_log_weights - parent_log_weights)
    terms = torch.where(mask, terms, torch.zeros_like(terms))
    return terms.sum(dim=-1).clamp_min(0.0)


class PHAREClassifier(nn.Module):
    """Parent-specific ABMIL heads with hierarchy-safe allele child heads."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        attn_dim: int = 384,
        gate: bool = True,
        dropout: float = 0.25,
        child_mode: ChildMode = "residual",
    ) -> None:
        super().__init__()
        if in_dim <= 0 or embed_dim <= 0 or attn_dim <= 0:
            raise ValueError("in_dim, embed_dim, and attn_dim must be positive")
        if num_fc_layers < 1:
            raise ValueError("num_fc_layers must be at least 1")
        if child_mode not in CHILD_MODES:
            raise ValueError(
                f"child_mode must be one of {CHILD_MODES}, got {child_mode!r}"
            )

        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.child_mode = child_mode
        self.projection = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False,
        )
        self.parent_heads = nn.ModuleDict(
            {
                name: _ParentHead(embed_dim, attn_dim, gate, dropout)
                for name in PARENT_TARGETS
            }
        )

        if child_mode == "parent_only":
            self.child_heads = nn.ModuleDict(
                {name: _ParentOnlyChildHead(embed_dim) for name in CHILD_TARGETS}
            )
        elif child_mode == "independent":
            self.child_heads = nn.ModuleDict(
                {
                    name: _IndependentChildHead(embed_dim, attn_dim, gate, dropout)
                    for name in CHILD_TARGETS
                }
            )
        else:
            self.child_heads = nn.ModuleDict(
                {
                    name: _ResidualChildHead(embed_dim, attn_dim, gate, dropout)
                    for name in CHILD_TARGETS
                }
            )

        self._training_stage: TrainingStage = "joint"
        self._parents_frozen = False

    @property
    def target_names(self) -> tuple[str, ...]:
        return TARGETS

    @property
    def training_stage(self) -> TrainingStage:
        return self._training_stage

    def set_training_stage(self, stage: TrainingStage) -> PHAREClassifier:
        """Select trainable parameters for parent, child, or joint training."""
        if stage not in ("parent", "child", "joint"):
            raise ValueError(
                "stage must be one of ('parent', 'child', 'joint'), "
                f"got {stage!r}"
            )
        self._training_stage = stage
        parent_trainable = stage in ("parent", "joint")
        child_trainable = stage in ("child", "joint")
        _set_requires_grad(self.projection, parent_trainable)
        _set_requires_grad(self.parent_heads, parent_trainable)
        _set_requires_grad(self.child_heads, child_trainable)
        self._parents_frozen = not parent_trainable
        # Reapply module modes, because ``train()`` is recursive by default.
        self.train(self.training)
        return self

    def freeze_parents(self) -> PHAREClassifier:
        """Freeze the projection and parent heads for child-stage training."""
        return self.set_training_stage("child")

    def assert_child_stage_frozen(self) -> None:
        """Fail closed if a child-stage optimizer could update parent semantics."""
        parent_parameters = tuple(self.projection.parameters()) + tuple(
            self.parent_heads.parameters()
        )
        if not self._parents_frozen or any(
            parameter.requires_grad for parameter in parent_parameters
        ):
            raise RuntimeError(
                "PHARE parent projection/heads are not frozen; call freeze_parents()"
            )

    def train(self, mode: bool = True) -> PHAREClassifier:
        super().train(mode)
        if mode and self._training_stage == "child":
            self.projection.eval()
            self.parent_heads.eval()
        elif mode and self._training_stage == "parent":
            self.child_heads.eval()
        return self

    def _validate_inputs(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"features must have shape [B, N, D], got {tuple(features.shape)}"
            )
        batch_size, bag_size, feature_dim = features.shape
        if feature_dim != self.in_dim:
            raise ValueError(
                f"expected feature dimension {self.in_dim}, got {feature_dim}"
            )
        if batch_size < 1 or bag_size < 1:
            raise ValueError("features must contain at least one non-empty bag")
        if mask is None:
            return torch.ones(
                (batch_size, bag_size), dtype=torch.bool, device=features.device
            )
        if mask.shape != (batch_size, bag_size):
            raise ValueError(
                f"mask must have shape {(batch_size, bag_size)}, got {tuple(mask.shape)}"
            )
        mask = mask.to(device=features.device, dtype=torch.bool)
        all_masked = ~mask.any(dim=-1)
        if all_masked.any():
            indices = all_masked.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"all-masked bags are invalid; batch indices={indices}")
        return mask

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = True,
    ) -> PHAREOutput:
        mask = self._validate_inputs(features, mask)
        projected = self.projection(features)

        parent_logits: dict[str, torch.Tensor] = {}
        parent_embeddings: dict[str, torch.Tensor] = {}
        parent_attention_logits: dict[str, torch.Tensor] = {}
        parent_attention_weights: dict[str, torch.Tensor] = {}

        for name in PARENT_TARGETS:
            head = self.parent_heads[name]
            raw_attention = head.attention(projected)
            attention_logits, attention_weights = _masked_attention(raw_attention, mask)
            embedding = _pool(attention_weights, projected)
            parent_embeddings[name] = embedding
            parent_logits[name] = head.classifier(embedding).squeeze(-1)
            parent_attention_logits[name] = attention_logits
            parent_attention_weights[name] = attention_weights

        # The child branch only sees detached parent/projection tensors.  This
        # remains true even if callers optimize marginal rather than conditional
        # logits, because factorisation below also uses a detached parent logit.
        child_tiles = projected.detach()
        conditional_logits: dict[str, torch.Tensor] = {}
        marginal_logits: dict[str, torch.Tensor] = {}
        child_embeddings: dict[str, torch.Tensor] = {}
        residual_embeddings: dict[str, torch.Tensor] = {}
        child_attention_logits: dict[str, torch.Tensor] = {}
        child_attention_weights: dict[str, torch.Tensor] = {}
        centered_deltas: dict[str, torch.Tensor] = {}
        child_kl: dict[str, torch.Tensor] = {}

        for child_name in CHILD_TARGETS:
            parent_name = CHILD_TO_PARENT[child_name]
            parent_embedding = parent_embeddings[parent_name].detach()
            parent_attn_logits = parent_attention_logits[parent_name].detach()
            parent_attn_weights = parent_attention_weights[parent_name].detach()
            child_head = self.child_heads[child_name]

            if self.child_mode == "parent_only":
                conditional = child_head.classifier(parent_embedding).squeeze(-1)
                child_embedding = parent_embedding
                residual_embedding = torch.zeros_like(parent_embedding)
                child_attn_logits = parent_attn_logits
                child_attn_weights = parent_attn_weights
                kl = torch.zeros(
                    parent_embedding.shape[0],
                    dtype=parent_attn_weights.dtype,
                    device=parent_embedding.device,
                )
            elif self.child_mode == "independent":
                raw_child_attention = child_head.attention(child_tiles)
                child_attn_logits, child_attn_weights = _masked_attention(
                    raw_child_attention, mask
                )
                child_embedding = _pool(child_attn_weights, child_tiles)
                residual_embedding = child_embedding - parent_embedding
                conditional = child_head.classifier(child_embedding).squeeze(-1)
                kl = _attention_kl(
                    child_attn_logits,
                    parent_attn_logits,
                    child_attn_weights,
                    mask,
                )
            else:
                raw_delta = child_head.delta_attention(child_tiles)
                centered_delta = _center_over_valid(raw_delta, mask)
                child_attn_logits, child_attn_weights = _masked_attention(
                    parent_attn_logits + centered_delta, mask
                )
                child_embedding = _pool(child_attn_weights, child_tiles)
                residual_embedding = child_embedding - parent_embedding
                conditional = (
                    child_head.base_classifier(parent_embedding)
                    + child_head.residual_classifier(residual_embedding)
                ).squeeze(-1)
                centered_deltas[child_name] = centered_delta
                kl = _attention_kl(
                    child_attn_logits,
                    parent_attn_logits,
                    child_attn_weights,
                    mask,
                )

            conditional_logits[child_name] = conditional
            marginal_logits[child_name] = factorized_marginal_logit(
                parent_logits[parent_name].detach(), conditional
            )
            child_embeddings[child_name] = child_embedding
            residual_embeddings[child_name] = residual_embedding
            child_attention_logits[child_name] = child_attn_logits
            child_attention_weights[child_name] = child_attn_weights
            child_kl[child_name] = kl

        logits = torch.stack(
            [
                *(parent_logits[name] for name in PARENT_TARGETS),
                *(marginal_logits[name] for name in CHILD_TARGETS),
            ],
            dim=-1,
        )
        if not return_attention:
            parent_attention_logits = {}
            parent_attention_weights = {}
            child_attention_logits = {}
            child_attention_weights = {}
            centered_deltas = {}

        return PHAREOutput(
            logits=logits,
            parent_logits=parent_logits,
            conditional_logits=conditional_logits,
            marginal_logits=marginal_logits,
            parent_embeddings=parent_embeddings,
            child_embeddings=child_embeddings,
            residual_embeddings=residual_embeddings,
            parent_attention_logits=parent_attention_logits,
            parent_attention_weights=parent_attention_weights,
            child_attention_logits=child_attention_logits,
            child_attention_weights=child_attention_weights,
            centered_deltas=centered_deltas,
            child_kl=child_kl,
        )


PHARE = PHAREClassifier

__all__ = [
    "CHILD_MODES",
    "CHILD_TARGETS",
    "CHILD_TO_PARENT",
    "PARENT_TARGETS",
    "PHARE",
    "PHAREClassifier",
    "PHAREOutput",
    "TARGETS",
    "factorized_marginal_logit",
]
