"""NumPy attention used as a framework-independent correctness oracle."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


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
