"""A functional decoder-only Transformer implemented with core JAX."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import-untyped]
from jax import Array

PyTree = dict[str, Any]
TrainStep = Callable[
    [PyTree, optax.OptState, Array, Array],
    tuple[PyTree, optax.OptState, Array, Array],
]


@dataclass(frozen=True)
class JAXGPTConfig:
    vocab_size: int = 256
    context_length: int = 128
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if (
            min(
                self.vocab_size,
                self.context_length,
                self.model_dim,
                self.num_heads,
                self.num_layers,
                self.mlp_ratio,
            )
            <= 0
        ):
            raise ValueError("all configuration sizes must be positive")


def _normal(key: Array, shape: tuple[int, ...]) -> Array:
    return jax.random.normal(key, shape) * 0.02


def init_params(key: Array, config: JAXGPTConfig) -> PyTree:
    """Initialize a parameter PyTree without Flax abstractions."""
    keys = iter(jax.random.split(key, 2 + 4 * config.num_layers))
    params: PyTree = {
        "token_embedding": _normal(next(keys), (config.vocab_size, config.model_dim)),
        "position_embedding": _normal(next(keys), (config.context_length, config.model_dim)),
        "blocks": [],
    }
    hidden_dim = config.mlp_ratio * config.model_dim
    for _ in range(config.num_layers):
        params["blocks"].append(
            {
                "qkv": _normal(next(keys), (config.model_dim, 3 * config.model_dim)),
                "output": _normal(next(keys), (config.model_dim, config.model_dim)),
                "up": _normal(next(keys), (config.model_dim, hidden_dim)),
                "down": _normal(next(keys), (hidden_dim, config.model_dim)),
                "attention_norm": jnp.ones((config.model_dim,)),
                "mlp_norm": jnp.ones((config.model_dim,)),
            }
        )
    params["blocks"] = tuple(params["blocks"])
    params["final_norm"] = jnp.ones((config.model_dim,))
    return params


def rms_norm(x: Array, scale: Array, epsilon: float = 1e-6) -> Array:
    normalized = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + epsilon)
    return normalized * scale


def causal_self_attention(x: Array, block: PyTree, config: JAXGPTConfig) -> Array:
    """Compute multi-head causal self-attention for x shaped [batch, time, dim]."""
    batch_size, sequence_length, _ = x.shape
    head_dim = config.model_dim // config.num_heads
    qkv = x @ block["qkv"]
    query, key, value = jnp.split(qkv, 3, axis=-1)

    def split_heads(tensor: Array) -> Array:
        return tensor.reshape(batch_size, sequence_length, config.num_heads, head_dim).transpose(
            0, 2, 1, 3
        )

    query, key, value = map(split_heads, (query, key, value))
    scores = query @ key.swapaxes(-2, -1) / math.sqrt(head_dim)
    mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=jnp.bool_))
    scores = jnp.where(mask, scores, jnp.asarray(-1e30, dtype=scores.dtype))
    probabilities = jax.nn.softmax(scores, axis=-1)
    attended = probabilities @ value
    attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, config.model_dim)
    return cast(Array, attended @ block["output"])


def forward(params: PyTree, input_ids: Array, config: JAXGPTConfig) -> Array:
    """Return token logits with shape [batch, time, vocab]."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have two dimensions, got {input_ids.shape}")
    sequence_length = input_ids.shape[1]
    if sequence_length > config.context_length:
        raise ValueError(
            f"sequence length {sequence_length} exceeds context {config.context_length}"
        )
    x = params["token_embedding"][input_ids]
    x = x + params["position_embedding"][jnp.arange(sequence_length)]
    for block in params["blocks"]:
        attention_input = rms_norm(x, block["attention_norm"])
        x = x + causal_self_attention(attention_input, block, config)
        mlp_input = rms_norm(x, block["mlp_norm"])
        hidden = jax.nn.gelu(mlp_input @ block["up"], approximate=True)
        x = x + hidden @ block["down"]
    x = rms_norm(x, params["final_norm"])
    return cast(Array, x @ params["token_embedding"].T)


def cross_entropy_loss(logits: Array, targets: Array, ignore_index: int = -100) -> Array:
    """Mean next-token loss with a JIT-compatible invalid-target sentinel.

    Empty supervision or visible targets outside the vocabulary produce a
    non-finite scalar instead of a plausible zero. ``make_train_step`` performs
    a host-side check and raises before the compiled update is entered.
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match the batch and time dimensions of logits")
    visible = targets != ignore_index
    target_in_range = (targets >= 0) & (targets < logits.shape[-1])
    valid_targets = ~visible | target_in_range
    safe_targets = jnp.where(visible & target_in_range, targets, 0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_loss = -jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1).squeeze(-1)
    visible_count = visible.sum()
    denominator = jnp.maximum(visible_count, 1)
    mean_loss = (token_loss * visible).sum() / denominator
    return jnp.where((visible_count > 0) & jnp.all(valid_targets), mean_loss, jnp.nan)


def adamw_optimizer(
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
) -> optax.GradientTransformation:
    """Create the explicit optimizer chain used by the tiny JAX experiment.

    The reported gradient norm in ``make_train_step`` is measured before
    clipping. This simple global weight decay is pedagogical; production LLM
    training commonly excludes norm and bias-like parameters with a mask.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )


def make_train_step(
    config: JAXGPTConfig,
    optimizer: optax.GradientTransformation,
) -> TrainStep:
    """Return a checked wrapper around one JIT-compiled next-token update."""

    def step(
        params: PyTree,
        optimizer_state: optax.OptState,
        input_ids: Array,
        targets: Array,
    ) -> tuple[PyTree, optax.OptState, Array, Array]:
        def loss_function(current_params: PyTree) -> Array:
            return cross_entropy_loss(forward(current_params, input_ids, config), targets)

        loss, gradients = jax.value_and_grad(loss_function)(params)
        gradient_norm = optax.tree.norm(gradients)
        updates, new_optimizer_state = optimizer.update(
            gradients, optimizer_state, params
        )
        new_params = optax.apply_updates(params, updates)
        return cast(PyTree, new_params), new_optimizer_state, loss, gradient_norm

    compiled_step = cast(TrainStep, jax.jit(step))

    def checked_step(
        params: PyTree,
        optimizer_state: optax.OptState,
        input_ids: Array,
        targets: Array,
    ) -> tuple[PyTree, optax.OptState, Array, Array]:
        target_values = np.asarray(jax.device_get(targets))
        if target_values.dtype.kind not in "iu":
            raise ValueError("targets must contain integer token ids")
        visible = target_values != -100
        if not np.any(visible):
            raise ValueError("targets must contain at least one supervised token")
        visible_targets = target_values[visible]
        if np.any(visible_targets < 0) or np.any(
            visible_targets >= config.vocab_size
        ):
            raise ValueError("supervised target ids must be in the vocabulary")
        return compiled_step(params, optimizer_state, input_ids, targets)

    return cast(TrainStep, checked_step)
