from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
gpt_jax = pytest.importorskip("about_llm.from_scratch.gpt_jax")
JAXGPTConfig = gpt_jax.JAXGPTConfig
adamw_optimizer = gpt_jax.adamw_optimizer
cross_entropy_loss = gpt_jax.cross_entropy_loss
forward = gpt_jax.forward
init_params = gpt_jax.init_params
make_train_step = gpt_jax.make_train_step


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


def test_config_rejects_invalid_sizes_and_head_partition() -> None:
    with pytest.raises(ValueError, match="divisible"):
        JAXGPTConfig(model_dim=10, num_heads=4)
    with pytest.raises(ValueError, match="positive"):
        JAXGPTConfig(vocab_size=0)


def test_forward_rejects_rank_and_context_overflow() -> None:
    config = JAXGPTConfig(
        vocab_size=8,
        context_length=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2,
    )
    params = init_params(jax.random.key(5), config)
    with pytest.raises(ValueError, match="two dimensions"):
        forward(params, jnp.array([0, 1, 2, 3]), config)
    with pytest.raises(ValueError, match="exceeds context"):
        forward(params, jnp.array([[0, 1, 2, 3, 4]]), config)


def test_masked_loss_rejects_shape_drift_and_handles_empty_supervision() -> None:
    logits = jnp.zeros((1, 3, 5), dtype=jnp.float32)
    with pytest.raises(ValueError, match="targets must match"):
        cross_entropy_loss(logits, jnp.zeros((1, 2), dtype=jnp.int32))
    empty = cross_entropy_loss(
        logits,
        jnp.full((1, 3), -100, dtype=jnp.int32),
    )
    assert float(empty) == 0.0


def test_jitted_optax_step_updates_params_and_overfits_tiny_batch() -> None:
    config = JAXGPTConfig(
        vocab_size=8,
        context_length=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2,
    )
    input_ids = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    targets = jnp.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    params = init_params(jax.random.key(11), config)
    optimizer = adamw_optimizer(learning_rate=0.02, weight_decay=0.0)
    optimizer_state = optimizer.init(params)
    train_step = make_train_step(config, optimizer)
    initial_embedding = np.asarray(params["token_embedding"]).copy()
    initial_loss = float(cross_entropy_loss(forward(params, input_ids, config), targets))
    gradient_norms: list[float] = []

    for _ in range(60):
        params, optimizer_state, loss, gradient_norm = train_step(
            params, optimizer_state, input_ids, targets
        )
        gradient_norms.append(float(gradient_norm))

    final_loss = float(cross_entropy_loss(forward(params, input_ids, config), targets))

    assert np.isfinite(float(loss))
    assert np.all(np.isfinite(gradient_norms))
    assert final_loss < initial_loss * 0.25
    assert not np.array_equal(initial_embedding, np.asarray(params["token_embedding"]))


def test_optimizer_configuration_rejects_invalid_hyperparameters() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        adamw_optimizer(learning_rate=0)
    with pytest.raises(ValueError, match="weight_decay"):
        adamw_optimizer(learning_rate=0.01, weight_decay=-1)
    with pytest.raises(ValueError, match="max_grad_norm"):
        adamw_optimizer(learning_rate=0.01, max_grad_norm=0)
