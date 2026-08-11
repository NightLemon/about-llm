"""NumPy attention used as a framework-independent correctness oracle."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def rms_norm(
    x: FloatArray,
    weight: FloatArray | None = None,
    *,
    epsilon: float = 1e-6,
) -> FloatArray:
    """Apply RMSNorm over the last dimension with float64 accumulation.

    The result is cast to the NumPy result dtype of ``x`` and ``weight``. This
    is a numerical reference, not a fused-kernel performance implementation.
    """

    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("x must have a non-empty feature dimension")
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("x must have a floating dtype")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    output_dtype = x.dtype
    weight64: FloatArray | None = None
    if weight is not None:
        if weight.shape != (x.shape[-1],):
            raise ValueError(
                f"weight must have shape ({x.shape[-1]},), got {weight.shape}"
            )
        if not np.issubdtype(weight.dtype, np.floating):
            raise ValueError("weight must have a floating dtype")
        output_dtype = np.result_type(x.dtype, weight.dtype)
        weight64 = weight.astype(np.float64)
    x64 = x.astype(np.float64)
    inverse_rms = np.reciprocal(
        np.sqrt(np.mean(np.square(x64), axis=-1, keepdims=True) + epsilon)
    )
    normalized = x64 * inverse_rms
    if weight64 is not None:
        normalized = normalized * weight64
    return cast(FloatArray, normalized.astype(output_dtype))


def apply_rope(
    query: FloatArray,
    key: FloatArray,
    positions: NDArray[np.integer],
    *,
    base: float = 10_000.0,
) -> tuple[FloatArray, FloatArray]:
    """Apply interleaved-pair rotary position embedding to query and key.

    Query and key may have different leading/head dimensions, but both must use
    shape ``(..., sequence_length, head_dim)`` with the same sequence and even
    head dimensions. ``positions`` supplies one absolute position per token.
    """

    if query.ndim < 2 or key.ndim < 2:
        raise ValueError("query and key must have at least two dimensions")
    if not np.issubdtype(query.dtype, np.floating) or not np.issubdtype(
        key.dtype, np.floating
    ):
        raise ValueError("query and key must have floating dtypes")
    sequence_length = query.shape[-2]
    head_dim = query.shape[-1]
    if key.shape[-2:] != (sequence_length, head_dim):
        raise ValueError("query and key must share sequence_length and head_dim")
    if head_dim == 0 or head_dim % 2:
        raise ValueError("RoPE head_dim must be a positive even number")
    if (
        positions.ndim != 1
        or positions.shape[0] != sequence_length
        or not np.issubdtype(positions.dtype, np.integer)
        or np.any(positions < 0)
    ):
        raise ValueError(
            "positions must be a non-negative integer vector matching sequence_length"
        )
    if not np.isfinite(base) or base <= 0:
        raise ValueError("base must be finite and positive")

    pair_indices = np.arange(0, head_dim, 2, dtype=np.float64)
    inverse_frequencies = np.power(base, -pair_indices / head_dim)
    angles = positions.astype(np.float64)[:, None] * inverse_frequencies[None, :]
    cosine = np.cos(angles)
    sine = np.sin(angles)
    query_broadcast = (1,) * (query.ndim - 2) + angles.shape
    key_broadcast = (1,) * (key.ndim - 2) + angles.shape
    return (
        _rotate_interleaved(
            query,
            cosine.reshape(query_broadcast),
            sine.reshape(query_broadcast),
        ),
        _rotate_interleaved(
            key,
            cosine.reshape(key_broadcast),
            sine.reshape(key_broadcast),
        ),
    )


def grouped_query_attention(
    query: FloatArray,
    key: FloatArray,
    value: FloatArray,
    *,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Reference grouped-query attention for ``[B, H, T, D]`` tensors.

    K/V heads are physically repeated only to make the mathematical equivalence
    explicit. Production GQA kernels should avoid this materialization.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("GQA query, key and value must have shape [B, H, T, D]")
    batch_size, query_heads, _, head_dim = query.shape
    key_batch, key_value_heads, key_length, key_dim = key.shape
    value_batch, value_heads, value_length, _ = value.shape
    if batch_size != key_batch or batch_size != value_batch:
        raise ValueError("GQA query, key and value batch sizes must match")
    if key_value_heads <= 0 or query_heads % key_value_heads:
        raise ValueError("query head count must be divisible by the K/V head count")
    if value_heads != key_value_heads:
        raise ValueError("key and value head counts must match")
    if key_length != value_length:
        raise ValueError("key and value sequence lengths must match")
    if head_dim != key_dim:
        raise ValueError("query and key head dimensions must match")
    repeats = query_heads // key_value_heads
    expanded_key = np.repeat(key, repeats, axis=1)
    expanded_value = np.repeat(value, repeats, axis=1)
    return scaled_dot_product_attention(
        query,
        expanded_key,
        expanded_value,
        mask=mask,
    )


def softmax(x: FloatArray, axis: int = -1) -> FloatArray:
    """Numerically stable softmax."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return cast(FloatArray, exp / exp.sum(axis=axis, keepdims=True))


def causal_mask(query_length: int, key_length: int | None = None) -> NDArray[np.bool_]:
    """Return a mask where True means the key is visible to the query."""
    if query_length <= 0:
        raise ValueError("query_length must be positive")
    key_length = query_length if key_length is None else key_length
    if key_length <= 0:
        raise ValueError("key_length must be positive")
    past_length = key_length - query_length
    if past_length < 0:
        raise ValueError("key_length cannot be smaller than query_length")
    query_positions = np.arange(query_length)[:, None] + past_length
    key_positions = np.arange(key_length)[None, :]
    return key_positions <= query_positions


def scaled_dot_product_attention(
    query: FloatArray,
    key: FloatArray,
    value: FloatArray,
    *,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute attention and return (output, probabilities).

    Expected shapes are (..., query_length, head_dim) for query and
    (..., key_length, head_dim) for key/value. Leading dimensions broadcast.
    """
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key and value must have at least two dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")

    scale = float(query.shape[-1]) ** -0.5
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale
    if mask is not None:
        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as error:
            raise ValueError(
                f"mask shape {mask.shape} cannot broadcast to scores shape {scores.shape}"
            ) from error
        scores = np.where(mask, scores, -np.inf)
        if np.any(np.all(~mask, axis=-1)):
            raise ValueError("every query row must have at least one visible key")

    probabilities = softmax(scores, axis=-1)
    return np.matmul(probabilities, value), probabilities


def _rotate_interleaved(
    tensor: FloatArray, cosine: FloatArray, sine: FloatArray
) -> FloatArray:
    even = tensor[..., 0::2].astype(np.float64)
    odd = tensor[..., 1::2].astype(np.float64)
    rotated = np.empty(tensor.shape, dtype=np.float64)
    rotated[..., 0::2] = even * cosine - odd * sine
    rotated[..., 1::2] = even * sine + odd * cosine
    return cast(FloatArray, rotated.astype(tensor.dtype))
