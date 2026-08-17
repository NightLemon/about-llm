from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from about_llm.from_scratch.attention_numpy import (
    apply_rope,
    blockwise_online_attention,
    causal_mask,
    grouped_query_attention,
    rms_norm,
    scaled_dot_product_attention,
)


def test_rms_norm_matches_definition_and_preserves_dtype() -> None:
    x = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    weight = np.array([2.0, 0.5], dtype=np.float32)

    actual = rms_norm(x, weight, epsilon=1e-6)
    expected = x * np.reciprocal(
        np.sqrt(np.mean(np.square(x.astype(np.float64)), axis=-1, keepdims=True) + 1e-6)
    ) * weight

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_rope_preserves_vector_norm_and_common_position_shift_dot_product() -> None:
    rng = np.random.default_rng(19)
    query = rng.normal(size=(1, 4, 3, 6)).astype(np.float64)
    key = rng.normal(size=(1, 2, 3, 6)).astype(np.float64)
    rotated_query, rotated_key = apply_rope(
        query, key, np.array([4, 5, 6], dtype=np.int64)
    )

    np.testing.assert_allclose(
        np.linalg.norm(rotated_query, axis=-1), np.linalg.norm(query, axis=-1)
    )
    np.testing.assert_allclose(
        np.linalg.norm(rotated_key, axis=-1), np.linalg.norm(key, axis=-1)
    )

    one_query = query[:, :1, :1]
    one_key = key[:, :1, :1]
    query_at_2, _ = apply_rope(one_query, one_query, np.array([2]))
    _, key_at_7 = apply_rope(one_key, one_key, np.array([7]))
    query_at_5, _ = apply_rope(one_query, one_query, np.array([5]))
    _, key_at_10 = apply_rope(one_key, one_key, np.array([10]))
    np.testing.assert_allclose(
        np.sum(query_at_2 * key_at_7, axis=-1),
        np.sum(query_at_5 * key_at_10, axis=-1),
    )


def test_grouped_query_attention_matches_explicit_kv_head_repetition() -> None:
    rng = np.random.default_rng(23)
    query = rng.normal(size=(2, 4, 3, 5))
    key = rng.normal(size=(2, 2, 3, 5))
    value = rng.normal(size=(2, 2, 3, 7))
    mask = causal_mask(3)

    actual_output, actual_probabilities = grouped_query_attention(
        query, key, value, mask=mask
    )
    expected_output, expected_probabilities = scaled_dot_product_attention(
        query,
        np.repeat(key, 2, axis=1),
        np.repeat(value, 2, axis=1),
        mask=mask,
    )

    np.testing.assert_allclose(actual_output, expected_output)
    np.testing.assert_allclose(actual_probabilities, expected_probabilities)


def test_incremental_cached_attention_matches_full_causal_attention() -> None:
    rng = np.random.default_rng(29)
    query = rng.normal(size=(1, 2, 5, 4))
    key = rng.normal(size=(1, 2, 5, 4))
    value = rng.normal(size=(1, 2, 5, 6))
    full_output, _ = scaled_dot_product_attention(
        query, key, value, mask=causal_mask(5)
    )

    incremental_outputs = []
    for position in range(5):
        output, _ = scaled_dot_product_attention(
            query[:, :, position : position + 1],
            key[:, :, : position + 1],
            value[:, :, : position + 1],
            mask=causal_mask(1, position + 1),
        )
        incremental_outputs.append(output)
    cached_output = np.concatenate(incremental_outputs, axis=-2)

    np.testing.assert_allclose(cached_output, full_output)


def test_causal_mask_with_past_keys() -> None:
    expected = np.array([[True, True, True, False], [True, True, True, True]])
    np.testing.assert_array_equal(causal_mask(query_length=2, key_length=4), expected)


def test_attention_probabilities_respect_causal_mask() -> None:
    rng = np.random.default_rng(7)
    query = rng.normal(size=(2, 3, 4))
    key = rng.normal(size=(2, 3, 4))
    value = rng.normal(size=(2, 3, 5))
    output, probabilities = scaled_dot_product_attention(query, key, value, mask=causal_mask(3))

    assert output.shape == (2, 3, 5)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    assert np.all(probabilities[..., 0, 1:] == 0)
    assert np.all(probabilities[..., 1, 2:] == 0)


def test_attention_rejects_fully_masked_query() -> None:
    tensor = np.ones((1, 2, 3))
    with pytest.raises(ValueError, match="at least one visible key"):
        scaled_dot_product_attention(tensor, tensor, tensor, mask=np.zeros((2, 2), dtype=np.bool_))


@pytest.mark.parametrize("block_size", [1, 2, 3, 7, 16])
def test_blockwise_online_attention_matches_dense_broadcast_reference(
    block_size: int,
) -> None:
    rng = np.random.default_rng(31)
    query = rng.normal(size=(2, 1, 5, 4))
    key = rng.normal(size=(1, 3, 7, 4))
    value = rng.normal(size=(2, 1, 7, 6))

    expected, _ = scaled_dot_product_attention(query, key, value)
    result = blockwise_online_attention(
        query, key, value, block_size=block_size
    )

    np.testing.assert_allclose(result.output, expected, rtol=1e-12, atol=1e-12)
    assert result.output.dtype == expected.dtype
    assert result.key_block_count == (7 + block_size - 1) // block_size
    assert result.logical_peak_score_elements == 2 * 3 * 5 * min(block_size, 7)
    assert result.full_score_elements == 2 * 3 * 5 * 7
    assert result.running_row_max.shape == (2, 3, 5)
    assert result.row_normalizer.shape == (2, 3, 5)
    assert not hasattr(result, "probabilities")


def test_blockwise_online_attention_matches_causal_prefill_and_decode() -> None:
    rng = np.random.default_rng(37)
    query = rng.normal(size=(1, 2, 6, 5))
    key = rng.normal(size=(1, 2, 6, 5))
    value = rng.normal(size=(1, 2, 6, 3))
    prefill_mask = causal_mask(6)

    expected_prefill, _ = scaled_dot_product_attention(
        query, key, value, mask=prefill_mask
    )
    actual_prefill = blockwise_online_attention(
        query, key, value, block_size=2, mask=prefill_mask
    ).output
    np.testing.assert_allclose(actual_prefill, expected_prefill, rtol=1e-12, atol=1e-12)

    decode_mask = causal_mask(query_length=1, key_length=6)
    expected_decode, _ = scaled_dot_product_attention(
        query[..., -1:, :], key, value, mask=decode_mask
    )
    actual_decode = blockwise_online_attention(
        query[..., -1:, :],
        key,
        value,
        block_size=4,
        mask=decode_mask,
    ).output
    np.testing.assert_allclose(actual_decode, expected_decode, rtol=1e-12, atol=1e-12)


def test_blockwise_online_attention_handles_sparse_mask_and_large_logits() -> None:
    query = np.array(
        [[1000.0, -1000.0], [-1000.0, 1000.0], [1000.0, 1000.0]]
    )
    key = np.array(
        [[1000.0, -1000.0], [-1000.0, 1000.0], [1000.0, 1000.0], [0.0, 0.0]]
    )
    value = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0], [13.0, 17.0]])
    mask = np.array(
        [
            [False, False, True, False],
            [False, True, True, False],
            [True, True, False, True],
        ],
        dtype=np.bool_,
    )

    expected, _ = scaled_dot_product_attention(query, key, value, mask=mask)
    with np.errstate(all="raise"):
        result = blockwise_online_attention(
            query, key, value, block_size=2, mask=mask
        )

    assert np.all(np.isfinite(result.output))
    assert np.all(np.isfinite(result.row_normalizer))
    np.testing.assert_allclose(result.output, expected, rtol=1e-12, atol=1e-12)


def test_blockwise_online_attention_preserves_result_dtype() -> None:
    tensor = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    result = blockwise_online_attention(tensor, tensor, tensor, block_size=2)

    assert result.output.dtype == np.float32
    assert result.running_row_max.dtype == np.float64
    assert result.row_normalizer.dtype == np.float64


def test_blockwise_online_attention_rejects_fully_masked_row() -> None:
    tensor = np.ones((2, 3), dtype=np.float64)
    mask = np.array([[True, True], [False, False]], dtype=np.bool_)

    with pytest.raises(ValueError, match="at least one visible key"):
        blockwise_online_attention(tensor, tensor, tensor, block_size=1, mask=mask)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: blockwise_online_attention(
                np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 3)), block_size=0
            ),
            "block_size",
        ),
        (
            lambda: blockwise_online_attention(
                np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 3)), block_size=True
            ),
            "block_size",
        ),
        (
            lambda: blockwise_online_attention(
                np.ones((2, 3), dtype=np.int64),
                np.ones((2, 3)),
                np.ones((2, 3)),
                block_size=1,
            ),
            "floating dtype",
        ),
        (
            lambda: blockwise_online_attention(
                np.array([[np.nan, 0.0]]),
                np.ones((1, 2)),
                np.ones((1, 2)),
                block_size=1,
            ),
            "finite values",
        ),
        (
            lambda: blockwise_online_attention(
                np.ones((2, 3)),
                np.ones((2, 4)),
                np.ones((2, 3)),
                block_size=1,
            ),
            "head dimensions",
        ),
        (
            lambda: blockwise_online_attention(
                np.ones((2, 3)),
                np.ones((2, 3)),
                np.ones((3, 3)),
                block_size=1,
            ),
            "sequence lengths",
        ),
        (
            lambda: blockwise_online_attention(
                np.ones((2, 3)),
                np.ones((2, 3)),
                np.ones((2, 3)),
                block_size=1,
                mask=np.ones((2, 2), dtype=np.int64),
            ),
            "boolean",
        ),
        (
            lambda: blockwise_online_attention(
                np.ones((2, 3)),
                np.ones((2, 3)),
                np.ones((2, 3)),
                block_size=1,
                mask=np.ones((3, 3), dtype=np.bool_),
            ),
            "cannot broadcast",
        ),
        (
            lambda: blockwise_online_attention(
                np.ones((2, 1, 2, 3)),
                np.ones((3, 1, 2, 3)),
                np.ones((2, 1, 2, 3)),
                block_size=1,
            ),
            "leading dimensions",
        ),
    ],
)
def test_blockwise_online_attention_rejects_invalid_contracts(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: rms_norm(np.ones((2, 3)), epsilon=0), "epsilon"),
        (
            lambda: apply_rope(
                np.ones((1, 2, 3)),
                np.ones((1, 2, 3)),
                np.array([0, 1]),
            ),
            "even",
        ),
        (
            lambda: grouped_query_attention(
                np.ones((1, 3, 2, 4)),
                np.ones((1, 2, 2, 4)),
                np.ones((1, 2, 2, 4)),
            ),
            "divisible",
        ),
    ],
)
def test_modern_attention_primitives_reject_invalid_contracts(
    operation: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()
