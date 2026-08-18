"""A compact decoder-only Transformer implemented directly in PyTorch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 256
    context_length: int = 128
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self) -> None:
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        positive = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"all size fields must be positive: {positive}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    causal_mask: Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.qkv = nn.Linear(config.model_dim, 3 * config.model_dim, bias=config.bias)
        self.output = nn.Linear(config.model_dim, config.model_dim, bias=config.bias)
        self.attention_dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask", mask.view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, model_dim = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        visible = self.causal_mask[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1)
        probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
        attended = probabilities @ value
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, model_dim)
        )
        return cast(Tensor, self.residual_dropout(self.output(attended)))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden_dim = config.mlp_ratio * config.model_dim
        self.layers = nn.Sequential(
            nn.Linear(config.model_dim, hidden_dim, bias=config.bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, config.model_dim, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.layers(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.model_dim)
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return cast(Tensor, x + self.mlp(self.mlp_norm(x)))


class MiniGPT(nn.Module):
    """A teaching model with learned positions and tied token embeddings."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, time], got {tuple(input_ids.shape)}"
            )
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context {self.config.context_length}"
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            if targets.shape != (batch_size, sequence_length):
                raise ValueError("targets must have the same [batch, time] shape as input_ids")
            supervised_targets = targets[targets != -100]
            if supervised_targets.numel() == 0:
                raise ValueError("targets must contain at least one supervised token")
            if torch.any(supervised_targets < 0) or torch.any(
                supervised_targets >= self.config.vocab_size
            ):
                raise ValueError("supervised target ids must be in the vocabulary")
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        output = input_ids
        for _ in range(max_new_tokens):
            window = output[:, -self.config.context_length :]
            logits, _ = self(window)
            next_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    k = min(top_k, next_logits.shape[-1])
                    threshold = torch.topk(next_logits, k).values[:, [-1]]
                    next_logits = next_logits.masked_fill(next_logits < threshold, -torch.inf)
                if top_p is not None and top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True, dim=-1
                    )
                    sorted_probabilities = F.softmax(sorted_logits, dim=-1)
                    cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
                    sorted_remove = cumulative_probabilities > top_p
                    # Keep the first token whose inclusion reaches the threshold.
                    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
                    sorted_remove[:, 0] = False
                    remove = torch.zeros_like(sorted_remove).scatter(
                        dim=-1, index=sorted_indices, src=sorted_remove
                    )
                    next_logits = next_logits.masked_fill(remove, -torch.inf)
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            output = torch.cat((output, next_token), dim=1)
        return output
