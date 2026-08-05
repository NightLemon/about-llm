"""A transparent LoRA wrapper for PyTorch linear layers."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Freeze a base linear layer and learn a low-rank update.

    The effective weight is W + (alpha / rank) * B @ A. B is initialized to
    zero, so wrapping a layer does not change its initial function.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.base = base
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)

        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return cast(Tensor, self.base(x) + update * self.scaling)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Return an independent ordinary Linear with the adapter merged."""
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.weight.copy_(self.base.weight + self.scaling * (self.lora_b @ self.lora_a))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged

    def adapter_state_dict(self) -> dict[str, Tensor | float | int]:
        """Return only portable adapter values and shape-critical metadata."""
        return {
            "lora_a": self.lora_a.detach().clone(),
            "lora_b": self.lora_b.detach().clone(),
            "rank": self.rank,
            "alpha": self.alpha,
            "in_features": self.base.in_features,
            "out_features": self.base.out_features,
        }
