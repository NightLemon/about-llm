from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from about_llm.from_scratch.attention_numpy import (
    apply_rope,
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
