from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from about_llm.from_scratch.attention_numpy import causal_mask, grouped_query_attention
from about_llm.inference import (
    QuantizedKVCache,
    quantize_kv_cache_int8,
    quantized_kv_grouped_query_attention,
)

pytestmark = pytest.mark.formula

ROOT = Path(__file__).resolve().parents[1]


def test_per_token_int8_codes_scales_and_storage_ledger() -> None:
    key = np.array([[[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.5]]]])
    value = np.array([[[[0.0, 0.0], [2.0, -1.0]]]])

    cache = quantize_kv_cache_int8(key, value)

    np.testing.assert_array_equal(
        cache.key_codes,
        [[[[0, 0, 0, 0], [0, 127, -127, 64]]]],
    )
    np.testing.assert_array_equal(cache.value_codes, [[[[0, 0], [127, -64]]]])
    np.testing.assert_allclose(cache.key_scales, [[[1.0, 1 / 127]]])
    np.testing.assert_allclose(cache.value_scales, [[[1.0, 2 / 127]]])
    assert cache.reference_fp32_bytes == 48
    assert cache.int8_code_bytes == 12
    assert cache.scale_metadata_bytes == 16
    assert cache.payload_bytes == 28
    assert cache.payload_compression_ratio == pytest.approx(48 / 28)
    assert not cache.key_codes.flags.writeable
    assert not cache.value_scales.flags.writeable


def test_per_vector_absmax_quantization_error_is_bounded_by_half_scale() -> None:
    rng = np.random.default_rng(11)
    key = rng.normal(size=(2, 3, 5, 7)).astype(np.float32)
    value = rng.normal(size=(2, 3, 5, 9)).astype(np.float32)

    cache = quantize_kv_cache_int8(key, value)
    restored_key, restored_value = cache.dequantize()

    assert np.all(np.abs(restored_key - key) <= cache.key_scales[..., None] / 2 + 1e-6)
    assert np.all(
        np.abs(restored_value - value) <= cache.value_scales[..., None] / 2 + 1e-6
    )
    assert np.all(cache.key_codes != -128)
    assert np.all(cache.value_codes != -128)


def test_quantized_gqa_matches_explicit_dequantization_and_causal_mask() -> None:
    rng = np.random.default_rng(17)
    query = rng.normal(size=(1, 4, 3, 6)).astype(np.float32)
    key = rng.normal(size=(1, 2, 5, 6)).astype(np.float32)
    value = rng.normal(size=(1, 2, 5, 8)).astype(np.float32)
    cache = quantize_kv_cache_int8(key, value)

    actual_output, actual_probabilities = quantized_kv_grouped_query_attention(
        query, cache
    )
    restored_key, restored_value = cache.dequantize()
    expected_output, expected_probabilities = grouped_query_attention(
        query,
        restored_key,
        restored_value,
        mask=causal_mask(3, 5),
    )

    np.testing.assert_allclose(actual_output, expected_output)
    np.testing.assert_allclose(actual_probabilities, expected_probabilities)
    np.testing.assert_allclose(actual_probabilities.sum(axis=-1), 1.0, atol=2e-7)
    assert np.all(actual_probabilities[..., 0, 3:] == 0)
    assert np.all(actual_probabilities[..., 1, 4:] == 0)


def test_incremental_quantized_cache_matches_full_quantized_causal_attention() -> None:
    rng = np.random.default_rng(23)
    query = rng.normal(size=(1, 4, 5, 4)).astype(np.float32)
    key = rng.normal(size=(1, 2, 5, 4)).astype(np.float32)
    value = rng.normal(size=(1, 2, 5, 6)).astype(np.float32)
    full_cache = quantize_kv_cache_int8(key, value)
    full_output, _ = quantized_kv_grouped_query_attention(query, full_cache)

    incremental = []
    for position in range(5):
        prefix_cache = quantize_kv_cache_int8(
            key[:, :, : position + 1], value[:, :, : position + 1]
        )
        output, _ = quantized_kv_grouped_query_attention(
            query[:, :, position : position + 1], prefix_cache
        )
        incremental.append(output)

    np.testing.assert_allclose(
        np.concatenate(incremental, axis=2), full_output, rtol=1e-5, atol=1e-7
    )


def test_quantized_cache_constructor_rejects_reserved_code_and_bad_scale() -> None:
    valid_codes = np.zeros((1, 1, 1, 2), dtype=np.int8)
    valid_scales = np.ones((1, 1, 1), dtype=np.float32)
    reserved = valid_codes.copy()
    reserved[0, 0, 0, 0] = -128
    with pytest.raises(ValueError, match="reserved -128"):
        QuantizedKVCache(reserved, valid_scales, valid_codes, valid_scales)
    with pytest.raises(ValueError, match="finite and positive"):
        QuantizedKVCache(
            valid_codes,
            np.zeros_like(valid_scales),
            valid_codes,
            valid_scales,
        )


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: quantize_kv_cache_int8(np.ones((2, 3)), np.ones((2, 3))), "shape"),
        (
            lambda: quantize_kv_cache_int8(
                np.full((1, 1, 1, 2), np.nan), np.ones((1, 1, 1, 2))
            ),
            "finite",
        ),
        (
            lambda: quantize_kv_cache_int8(
                np.ones((1, 1, 2, 2)), np.ones((1, 1, 3, 2))
            ),
            "share batch",
        ),
        (
            lambda: quantized_kv_grouped_query_attention(
                np.ones((1, 2, 3, 2)),
                quantize_kv_cache_int8(
                    np.ones((1, 1, 2, 2)), np.ones((1, 1, 2, 2))
                ),
            ),
            "cannot exceed",
        ),
        (
            lambda: quantized_kv_grouped_query_attention(
                np.ones((1, 2, 1, 3)),
                quantize_kv_cache_int8(
                    np.ones((1, 1, 2, 2)), np.ones((1, 1, 2, 2))
                ),
            ),
            "head dimensions",
        ),
    ],
)
def test_kv_quantization_rejects_invalid_contracts(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()

