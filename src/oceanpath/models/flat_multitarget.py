"""Shared-trunk ABMIL with independent one-logit molecular target heads.

This is deliberately different from :class:`MultiheadABMIL`: that model has
multiple *attention* heads for one classification task.  The model here has
one shared ABMIL attention trunk and one binary output head per target.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
import torch.nn as nn

from oceanpath.models.abmil import ABMIL
from oceanpath.models.base import MILOutput

_TARGET_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _validate_target_names(target_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in target_names)
    if not names:
        raise ValueError("target_names must contain at least one target")
    if len(set(names)) != len(names):
        raise ValueError(f"target_names must be unique, got {names}")
    invalid = [name for name in names if not _TARGET_NAME.fullmatch(name)]
    if invalid:
        raise ValueError(
            "target names must start with a letter and contain only letters, "
            f"numbers, and underscores; invalid={invalid}"
        )
    return names


class FlatMultiTargetClassifier(nn.Module):
    """One ABMIL representation shared by independent binary target heads.

    The returned ``MILOutput.logits`` tensor is ``[B, T]`` in the exact order
    of :attr:`target_names`.  Each column is a raw, uncalibrated binary logit;
    no softmax is applied across targets.
    """

    def __init__(
        self,
        in_dim: int,
        target_names: Sequence[str],
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        attn_dim: int = 384,
        gate: bool = True,
        dropout: float = 0.25,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.target_names = _validate_target_names(target_names)
        self.aggregator = ABMIL(
            in_dim=in_dim,
            embed_dim=embed_dim,
            num_fc_layers=num_fc_layers,
            attn_dim=attn_dim,
            gate=gate,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(embed_dim, 1) for name in self.target_names}
        )

    @property
    def num_targets(self) -> int:
        return len(self.target_names)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> MILOutput:
        trunk = self.aggregator(
            features,
            mask=mask,
            coords=coords,
            return_attention=return_attention,
        )
        logits = torch.cat(
            [self.heads[name](trunk.slide_embedding) for name in self.target_names],
            dim=-1,
        )
        return MILOutput(
            slide_embedding=trunk.slide_embedding,
            logits=logits,
            extras=trunk.extras,
        )

    def logits_by_target(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return named views over a ``[B, T]`` logits tensor."""
        if logits.ndim != 2 or logits.shape[1] != self.num_targets:
            raise ValueError(
                f"expected logits [B, {self.num_targets}], got {tuple(logits.shape)}"
            )
        return {name: logits[:, index] for index, name in enumerate(self.target_names)}


# Backward-compatible descriptive alias: the classifier's shared trunk is ABMIL.
FlatMultiTargetABMIL = FlatMultiTargetClassifier


__all__ = ["FlatMultiTargetABMIL", "FlatMultiTargetClassifier"]
