from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
gpt_jax = pytest.importorskip("about_llm.from_scratch.gpt_jax")
JAXGPTConfig = gpt_jax.JAXGPTConfig
cross_entropy_loss = gpt_jax.cross_entropy_loss
forward = gpt_jax.forward
init_params = gpt_jax.init_params


def test_jax_forward_loss_and_causality() -> None:
    config = JAXGPTConfig(
        vocab_size=32,
        context_length=8,
        model_dim=16,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2,
    )
    params = init_params(jax.random.key(3), config)
    first = jnp.array([[1, 2, 3, 4]])
    second = jnp.array([[1, 2, 7, 8]])
    first_logits = forward(params, first, config)
    second_logits = forward(params, second, config)
    loss = cross_entropy_loss(first_logits, jnp.array([[2, 3, 4, 5]]))

    assert first_logits.shape == (1, 4, 32)
    assert np.isfinite(np.asarray(loss))
    np.testing.assert_allclose(
        np.asarray(first_logits[:, :2]),
        np.asarray(second_logits[:, :2]),
        rtol=1e-5,
        atol=1e-6,
    )
