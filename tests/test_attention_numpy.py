from __future__ import annotations

import numpy as np
import pytest

from about_llm.from_scratch.attention_numpy import causal_mask, scaled_dot_product_attention


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
