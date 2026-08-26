"""Capacity calculations whose assumptions are explicit and unit-testable."""

from __future__ import annotations


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def estimate_causal_generation_forward_positions(
    *,
    prompt_tokens: int,
    output_tokens: int,
    cached_prompt_tokens: int = 0,
) -> int:
    """Return positions evaluated by a standard cached decoder-only generation.

    The cache represents a strict prompt prefix. At least one prompt position is
    still evaluated so its logits can produce the first output token. Each later
    output token needs one decode position. Speculation, beams, recomputation,
    padding, and auxiliary model work are outside this ledger.
    """

    prompt_tokens = _positive_integer(prompt_tokens, "prompt_tokens")
    output_tokens = _positive_integer(output_tokens, "output_tokens")
    cached_prompt_tokens = _non_negative_integer(
        cached_prompt_tokens, "cached_prompt_tokens"
    )
    if cached_prompt_tokens >= prompt_tokens:
        raise ValueError("cached_prompt_tokens must leave one prompt position to evaluate")
    return prompt_tokens - cached_prompt_tokens + output_tokens - 1


def estimate_kv_cache_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    tokens: int,
    batch_size: int = 1,
    bytes_per_element: int = 2,
) -> int:
    """Return idealized dense K/V storage, excluding allocator and block metadata.

    The factor of two represents key and value. Architectures with latent or
    otherwise compressed caches need their own layout-specific calculation.
    """
    num_layers = _positive_integer(num_layers, "num_layers")
    num_kv_heads = _positive_integer(num_kv_heads, "num_kv_heads")
    head_dim = _positive_integer(head_dim, "head_dim")
    tokens = _positive_integer(tokens, "tokens")
    batch_size = _positive_integer(batch_size, "batch_size")
    bytes_per_element = _positive_integer(bytes_per_element, "bytes_per_element")
    return (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * tokens
        * batch_size
        * bytes_per_element
    )
