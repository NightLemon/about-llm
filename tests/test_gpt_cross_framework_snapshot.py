from __future__ import annotations

import numpy as np
import pytest
import torch

from about_llm.from_scratch.gpt_cross_framework import (
    torch_model_to_layernorm_jax_params,
)
from about_llm.from_scratch.gpt_torch import GPTConfig, MiniGPT

pytestmark = [pytest.mark.contract, pytest.mark.integration]


def test_torch_to_jax_mapping_is_an_owned_parameter_snapshot() -> None:
    """A later Torch in-place update cannot alter the mapped JAX values."""

    model = MiniGPT(
        GPTConfig(
            vocab_size=11,
            context_length=5,
            model_dim=8,
            num_heads=2,
            num_layers=2,
            mlp_ratio=2,
            dropout=0.0,
            bias=False,
        )
    )
    params = torch_model_to_layernorm_jax_params(model)
    mapped_before_update = np.asarray(params["token_embedding"]).copy()

    with torch.no_grad():
        model.token_embedding.weight.add_(1.0)

    np.testing.assert_array_equal(
        np.asarray(params["token_embedding"]),
        mapped_before_update,
    )
