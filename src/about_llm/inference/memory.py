"""Capacity calculations whose assumptions are explicit and unit-testable."""

from __future__ import annotations


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
    values = (
        num_layers,
        num_kv_heads,
        head_dim,
        tokens,
        batch_size,
        bytes_per_element,
    )
    if any(value <= 0 for value in values):
        raise ValueError("all KV-cache dimensions must be positive")
    return (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * tokens
        * batch_size
        * bytes_per_element
    )
