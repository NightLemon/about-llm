"""CPU reference for per-token symmetric INT8 KV-cache quantization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from about_llm.from_scratch.attention_numpy import causal_mask, grouped_query_attention

_INT8_QMAX = 127


@dataclass(frozen=True)
class QuantizedKVCache:
    """INT8 K/V codes with one FP32 scale per batch/head/token vector.

    Shapes are ``[batch, kv_heads, cached_tokens, head_dim]``. Keys and values
    may use different final dimensions, but share batch/head/token axes. This
    is an unpacked NumPy payload and not a runtime-specific paged KV layout.
    """

    key_codes: NDArray[np.int8]
    key_scales: NDArray[np.float32]
    value_codes: NDArray[np.int8]
    value_scales: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.key_codes.ndim != 4 or self.value_codes.ndim != 4:
            raise ValueError("K/V codes must have shape [B, Hkv, T, D]")
        if self.key_codes.shape[:3] != self.value_codes.shape[:3]:
            raise ValueError("K/V codes must share batch, head, and token dimensions")
        if any(dimension <= 0 for dimension in self.key_codes.shape):
            raise ValueError("K code dimensions must be positive")
        if self.value_codes.shape[-1] <= 0:
            raise ValueError("V head dimension must be positive")
        if self.key_codes.dtype != np.int8 or self.value_codes.dtype != np.int8:
            raise ValueError("K/V codes must use int8")
        expected_scale_shape = self.key_codes.shape[:3]
        if self.key_scales.shape != expected_scale_shape:
            raise ValueError("K scales must have shape [B, Hkv, T]")
        if self.value_scales.shape != expected_scale_shape:
            raise ValueError("V scales must have shape [B, Hkv, T]")
        if self.key_scales.dtype != np.float32 or self.value_scales.dtype != np.float32:
            raise ValueError("K/V scales must use float32")
        for name, codes in (("K", self.key_codes), ("V", self.value_codes)):
            if np.any(codes == -128):
                raise ValueError(f"{name} codes cannot use the reserved -128 value")
        for name, scales in (("K", self.key_scales), ("V", self.value_scales)):
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError(f"{name} scales must be finite and positive")

        for field_name in ("key_codes", "key_scales", "value_codes", "value_scales"):
            copied = getattr(self, field_name).copy()
            copied.setflags(write=False)
            object.__setattr__(self, field_name, copied)

    @property
    def batch_size(self) -> int:
        return int(self.key_codes.shape[0])

    @property
    def key_value_heads(self) -> int:
        return int(self.key_codes.shape[1])

    @property
    def cached_tokens(self) -> int:
        return int(self.key_codes.shape[2])

    @property
    def key_head_dim(self) -> int:
        return int(self.key_codes.shape[3])

    @property
    def value_head_dim(self) -> int:
        return int(self.value_codes.shape[3])

    @property
    def reference_fp32_bytes(self) -> int:
        return int(
            (self.key_codes.size + self.value_codes.size)
            * np.dtype(np.float32).itemsize
        )

    @property
    def int8_code_bytes(self) -> int:
        return int(self.key_codes.nbytes + self.value_codes.nbytes)

    @property
    def scale_metadata_bytes(self) -> int:
        return int(self.key_scales.nbytes + self.value_scales.nbytes)

    @property
    def payload_bytes(self) -> int:
        return self.int8_code_bytes + self.scale_metadata_bytes

    @property
    def payload_compression_ratio(self) -> float:
        return self.reference_fp32_bytes / self.payload_bytes

    def dequantize(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        key = self.key_codes.astype(np.float32) * self.key_scales[..., None]
        value = self.value_codes.astype(np.float32) * self.value_scales[..., None]
        return key, value


def quantize_kv_cache_int8(key: ArrayLike, value: ArrayLike) -> QuantizedKVCache:
    """Quantize each K/V token vector independently with symmetric absmax."""

    key_array = _finite_kv_array(key, "key")
    value_array = _finite_kv_array(value, "value")
    if key_array.shape[:3] != value_array.shape[:3]:
        raise ValueError("key and value must share batch, head, and token dimensions")
    key_codes, key_scales = _quantize_token_vectors(key_array)
    value_codes, value_scales = _quantize_token_vectors(value_array)
    return QuantizedKVCache(
        key_codes=key_codes,
        key_scales=key_scales,
        value_codes=value_codes,
        value_scales=value_scales,
    )


def quantized_kv_grouped_query_attention(
    query: ArrayLike,
    cache: QuantizedKVCache,
    *,
    causal: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Dequantize K/V and execute the existing GQA numerical oracle.

    This validates cache quantization semantics and error propagation. It does
    not execute attention directly on INT8 codes or model a fused GPU kernel.
    """

    query_array = np.asarray(query, dtype=np.float32)
    if query_array.ndim != 4 or any(dimension <= 0 for dimension in query_array.shape):
        raise ValueError("query must have non-empty shape [B, Hq, Tq, D]")
    if not np.all(np.isfinite(query_array)):
        raise ValueError("query must contain only finite values")
    if query_array.shape[0] != cache.batch_size:
        raise ValueError("query and cache batch sizes must match")
    if query_array.shape[-1] != cache.key_head_dim:
        raise ValueError("query and key head dimensions must match")
    if query_array.shape[-2] > cache.cached_tokens:
        raise ValueError("query length cannot exceed cached token count")
    key, value = cache.dequantize()
    mask = (
        causal_mask(query_array.shape[-2], cache.cached_tokens)
        if causal
        else None
    )
    output, probabilities = grouped_query_attention(
        query_array,
        key,
        value,
        mask=mask,
    )
    return (
        np.asarray(output, dtype=np.float32),
        np.asarray(probabilities, dtype=np.float32),
    )


def _finite_kv_array(value: ArrayLike, name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 4 or any(dimension <= 0 for dimension in array.shape):
        raise ValueError(f"{name} must have non-empty shape [B, Hkv, T, D]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _quantize_token_vectors(
    value: NDArray[np.float32],
) -> tuple[NDArray[np.int8], NDArray[np.float32]]:
    maximum = np.max(np.abs(value), axis=-1)
    scales = np.where(maximum == 0, 1.0, maximum / _INT8_QMAX).astype(np.float32)
    codes = np.clip(
        np.rint(value / scales[..., None]),
        -_INT8_QMAX,
        _INT8_QMAX,
    ).astype(np.int8)
    return codes, scales
