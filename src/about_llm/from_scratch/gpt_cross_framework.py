"""PyTorch/JAX MiniGPT numerical parity control on one explicit contract."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, cast

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax  # type: ignore[import-untyped]
import torch
from jax import Array
from torch import Tensor

from about_llm.from_scratch.gpt_jax import (
    JAXGPTConfig,
    cross_entropy_loss,
)
from about_llm.from_scratch.gpt_jax import forward as native_rmsnorm_forward
from about_llm.from_scratch.gpt_torch import GPTConfig, MiniGPT

GPT_CROSS_FRAMEWORK_PARITY_VERSION = (
    "about-llm.gpt-cross-framework-parity.v1"
)
LEARNING_RATE = 0.025
MAX_ABS_TOLERANCE = 2e-6

PyTree = dict[str, Any]


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _fixture_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=11,
        context_length=5,
        model_dim=8,
        num_heads=2,
        num_layers=2,
        mlp_ratio=2,
        dropout=0.0,
        bias=False,
    )


def _jax_config(config: GPTConfig) -> JAXGPTConfig:
    return JAXGPTConfig(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        mlp_ratio=config.mlp_ratio,
    )


def _fill_deterministic_parameters(model: MiniGPT) -> None:
    """Fill every unique parameter without depending on a framework RNG."""

    with torch.no_grad():
        for parameter_index, (name, parameter) in enumerate(
            model.named_parameters()
        ):
            indices = torch.arange(
                parameter.numel(),
                dtype=torch.float64,
            ).reshape(parameter.shape)
            phase = (indices + 1.0) * (parameter_index + 1.0) * 0.137
            if name.endswith("norm.weight"):
                values = 1.0 + 0.03 * torch.cos(phase)
            elif name.endswith("norm.bias"):
                values = 0.02 * torch.sin(phase)
            else:
                values = 0.05 * torch.sin(phase)
            parameter.copy_(values.to(dtype=parameter.dtype))


def _jax_array(tensor: Tensor, *, transpose: bool = False) -> Array:
    values = tensor.detach().cpu().numpy()
    if transpose:
        values = values.T
    return jnp.asarray(values)


def torch_model_to_layernorm_jax_params(model: MiniGPT) -> PyTree:
    """Map the authored PyTorch model to the parity JAX parameter tree."""

    if model.config.bias:
        raise ValueError("the parity fixture requires bias-free linear layers")
    state = model.state_dict()

    def parameter(name: str) -> Tensor:
        return cast(Tensor, state[name])

    blocks: list[PyTree] = []
    for index in range(model.config.num_layers):
        prefix = f"blocks.{index}"
        blocks.append(
            {
                "qkv": _jax_array(
                    parameter(f"{prefix}.attention.qkv.weight"),
                    transpose=True,
                ),
                "output": _jax_array(
                    parameter(f"{prefix}.attention.output.weight"),
                    transpose=True,
                ),
                "up": _jax_array(
                    parameter(f"{prefix}.mlp.layers.0.weight"),
                    transpose=True,
                ),
                "down": _jax_array(
                    parameter(f"{prefix}.mlp.layers.2.weight"),
                    transpose=True,
                ),
                "attention_norm_weight": _jax_array(
                    parameter(f"{prefix}.attention_norm.weight")
                ),
                "attention_norm_bias": _jax_array(
                    parameter(f"{prefix}.attention_norm.bias")
                ),
                "mlp_norm_weight": _jax_array(
                    parameter(f"{prefix}.mlp_norm.weight")
                ),
                "mlp_norm_bias": _jax_array(
                    parameter(f"{prefix}.mlp_norm.bias")
                ),
            }
        )
    return {
        "token_embedding": _jax_array(parameter("token_embedding.weight")),
        "position_embedding": _jax_array(
            parameter("position_embedding.weight")
        ),
        "blocks": tuple(blocks),
        "final_norm_weight": _jax_array(parameter("final_norm.weight")),
        "final_norm_bias": _jax_array(parameter("final_norm.bias")),
    }


def layer_norm(
    hidden_states: Array,
    weight: Array,
    bias: Array,
    *,
    epsilon: float = 1e-5,
) -> Array:
    """Match torch.nn.LayerNorm over the last dimension."""

    mean = jnp.mean(hidden_states, axis=-1, keepdims=True)
    centered = hidden_states - mean
    variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    normalized = centered * jax.lax.rsqrt(variance + epsilon)
    return normalized * weight + bias


def _layernorm_attention(
    hidden_states: Array,
    block: PyTree,
    config: JAXGPTConfig,
) -> Array:
    batch_size, sequence_length, _ = hidden_states.shape
    head_dim = config.model_dim // config.num_heads
    qkv = hidden_states @ block["qkv"]
    query, key, value = jnp.split(qkv, 3, axis=-1)

    def split_heads(tensor: Array) -> Array:
        return tensor.reshape(
            batch_size,
            sequence_length,
            config.num_heads,
            head_dim,
        ).transpose(0, 2, 1, 3)

    query, key, value = map(split_heads, (query, key, value))
    scores = query @ key.swapaxes(-2, -1) / math.sqrt(head_dim)
    visible = jnp.tril(
        jnp.ones((sequence_length, sequence_length), dtype=jnp.bool_)
    )
    masked_value = jnp.asarray(np.finfo(np.float32).min, dtype=scores.dtype)
    scores = jnp.where(visible, scores, masked_value)
    probabilities = jax.nn.softmax(scores, axis=-1)
    attended = probabilities @ value
    attended = attended.transpose(0, 2, 1, 3).reshape(
        batch_size,
        sequence_length,
        config.model_dim,
    )
    return cast(Array, attended @ block["output"])


def layernorm_jax_forward(
    params: PyTree,
    input_ids: Array,
    config: JAXGPTConfig,
) -> Array:
    """Execute the exact architecture contract used by PyTorch MiniGPT."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have two dimensions")
    sequence_length = input_ids.shape[1]
    if sequence_length > config.context_length:
        raise ValueError("input sequence exceeds the configured context")
    hidden_states = params["token_embedding"][input_ids]
    hidden_states = hidden_states + params["position_embedding"][
        jnp.arange(sequence_length)
    ]
    for block in params["blocks"]:
        attention_input = layer_norm(
            hidden_states,
            block["attention_norm_weight"],
            block["attention_norm_bias"],
        )
        hidden_states = hidden_states + _layernorm_attention(
            attention_input,
            block,
            config,
        )
        mlp_input = layer_norm(
            hidden_states,
            block["mlp_norm_weight"],
            block["mlp_norm_bias"],
        )
        activated = jax.nn.gelu(
            mlp_input @ block["up"],
            approximate=True,
        )
        hidden_states = hidden_states + activated @ block["down"]
    hidden_states = layer_norm(
        hidden_states,
        params["final_norm_weight"],
        params["final_norm_bias"],
    )
    return cast(Array, hidden_states @ params["token_embedding"].T)


def _native_rmsnorm_params(params: PyTree) -> PyTree:
    return {
        "token_embedding": params["token_embedding"],
        "position_embedding": params["position_embedding"],
        "blocks": tuple(
            {
                "qkv": block["qkv"],
                "output": block["output"],
                "up": block["up"],
                "down": block["down"],
                "attention_norm": block["attention_norm_weight"],
                "mlp_norm": block["mlp_norm_weight"],
            }
            for block in params["blocks"]
        ),
        "final_norm": params["final_norm_weight"],
    }


def _arrays_by_torch_name(params: PyTree) -> dict[str, Array]:
    arrays: dict[str, Array] = {
        "token_embedding.weight": params["token_embedding"],
        "position_embedding.weight": params["position_embedding"],
        "final_norm.weight": params["final_norm_weight"],
        "final_norm.bias": params["final_norm_bias"],
    }
    for index, block in enumerate(params["blocks"]):
        prefix = f"blocks.{index}"
        arrays[f"{prefix}.attention.qkv.weight"] = block["qkv"].T
        arrays[f"{prefix}.attention.output.weight"] = block["output"].T
        arrays[f"{prefix}.attention_norm.weight"] = block[
            "attention_norm_weight"
        ]
        arrays[f"{prefix}.attention_norm.bias"] = block[
            "attention_norm_bias"
        ]
        arrays[f"{prefix}.mlp.layers.0.weight"] = block["up"].T
        arrays[f"{prefix}.mlp.layers.2.weight"] = block["down"].T
        arrays[f"{prefix}.mlp_norm.weight"] = block["mlp_norm_weight"]
        arrays[f"{prefix}.mlp_norm.bias"] = block["mlp_norm_bias"]
    return arrays


def _max_abs_difference(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise AssertionError(
            f"shape mismatch: {left_array.shape} != {right_array.shape}"
        )
    if left_array.size == 0:
        return 0.0
    return float(np.max(np.abs(left_array - right_array)))


def _torch_arrays(model: MiniGPT, *, gradients: bool) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, parameter in model.named_parameters():
        tensor = parameter.grad if gradients else parameter
        if tensor is None:
            raise AssertionError(f"parameter {name} has no gradient")
        arrays[name] = tensor.detach().cpu().numpy().copy()
    return arrays


def _tree_differences(
    torch_arrays: dict[str, np.ndarray],
    jax_params: PyTree,
) -> dict[str, float]:
    jax_arrays = _arrays_by_torch_name(jax_params)
    if set(torch_arrays) != set(jax_arrays):
        raise AssertionError("PyTorch and JAX parameter names differ")
    return {
        name: _max_abs_difference(torch_arrays[name], jax_arrays[name])
        for name in sorted(torch_arrays)
    }


def _run_gpt_cross_framework_parity_control() -> dict[str, object]:
    """Compare one mapped PyTorch/JAX forward, backward, and SGD update."""

    config = _fixture_config()
    jax_config = _jax_config(config)
    model = MiniGPT(config).eval()
    _fill_deterministic_parameters(model)
    input_ids = torch.tensor(
        [[0, 1, 2, 3], [3, 2, 1, 0]],
        dtype=torch.int64,
    )
    targets = torch.tensor(
        [[1, 2, 3, 4], [2, -100, 0, 5]],
        dtype=torch.int64,
    )
    jax_input_ids = jnp.asarray(input_ids.numpy())
    jax_targets = jnp.asarray(targets.numpy())
    initial_params = torch_model_to_layernorm_jax_params(model)

    model.zero_grad(set_to_none=True)
    torch_logits, torch_loss = model(input_ids, targets)
    if torch_loss is None:
        raise AssertionError("PyTorch loss unexpectedly missing")
    torch_loss.backward()

    def jax_loss_function(params: PyTree) -> Array:
        return cross_entropy_loss(
            layernorm_jax_forward(params, jax_input_ids, jax_config),
            jax_targets,
        )

    jax_loss, jax_gradients = jax.value_and_grad(jax_loss_function)(
        initial_params
    )
    jax_logits = layernorm_jax_forward(
        initial_params,
        jax_input_ids,
        jax_config,
    )
    native_rmsnorm_logits = native_rmsnorm_forward(
        _native_rmsnorm_params(initial_params),
        jax_input_ids,
        jax_config,
    )

    gradient_differences = _tree_differences(
        _torch_arrays(model, gradients=True),
        cast(PyTree, jax_gradients),
    )
    torch_optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    torch_optimizer.step()
    jax_optimizer = optax.sgd(learning_rate=LEARNING_RATE)
    jax_optimizer_state = jax_optimizer.init(initial_params)
    jax_updates, _ = jax_optimizer.update(
        jax_gradients,
        jax_optimizer_state,
        initial_params,
    )
    updated_jax_params = cast(
        PyTree,
        optax.apply_updates(initial_params, jax_updates),
    )
    parameter_differences = _tree_differences(
        _torch_arrays(model, gradients=False),
        updated_jax_params,
    )

    with torch.no_grad():
        torch_logits_after, torch_loss_after = model(input_ids, targets)
    if torch_loss_after is None:
        raise AssertionError("post-update PyTorch loss unexpectedly missing")
    jax_logits_after = layernorm_jax_forward(
        updated_jax_params,
        jax_input_ids,
        jax_config,
    )
    jax_loss_after = cross_entropy_loss(jax_logits_after, jax_targets)

    initial_logits_difference = _max_abs_difference(
        torch_logits.detach().numpy(),
        np.asarray(jax_logits),
    )
    initial_loss_difference = abs(
        float(torch_loss.detach().item()) - float(jax_loss)
    )
    gradient_global_difference = max(gradient_differences.values())
    parameter_global_difference = max(parameter_differences.values())
    post_update_logits_difference = _max_abs_difference(
        torch_logits_after.detach().numpy(),
        np.asarray(jax_logits_after),
    )
    post_update_loss_difference = abs(
        float(torch_loss_after.item()) - float(jax_loss_after)
    )
    rmsnorm_counterfactual_difference = _max_abs_difference(
        torch_logits.detach().numpy(),
        np.asarray(native_rmsnorm_logits),
    )
    comparisons: dict[str, object] = {
        "initial_logits_max_abs_difference": initial_logits_difference,
        "initial_loss_abs_difference": initial_loss_difference,
        "gradient_max_abs_difference_by_parameter": gradient_differences,
        "gradient_global_max_abs_difference": gradient_global_difference,
        "post_update_parameter_max_abs_difference_by_parameter": (
            parameter_differences
        ),
        "post_update_parameter_global_max_abs_difference": (
            parameter_global_difference
        ),
        "post_update_logits_max_abs_difference": (
            post_update_logits_difference
        ),
        "post_update_loss_abs_difference": post_update_loss_difference,
        "native_rmsnorm_counterfactual_logits_max_abs_difference": (
            rmsnorm_counterfactual_difference
        ),
    }
    assertions = {
        "mapped_initial_logits_match": (
            initial_logits_difference <= MAX_ABS_TOLERANCE
        ),
        "mapped_initial_loss_matches": (
            initial_loss_difference <= MAX_ABS_TOLERANCE
        ),
        "all_parameter_gradients_match": (
            gradient_global_difference <= MAX_ABS_TOLERANCE
        ),
        "plain_sgd_updated_parameters_match": (
            parameter_global_difference <= MAX_ABS_TOLERANCE
        ),
        "post_update_logits_and_loss_match": (
            post_update_logits_difference <= MAX_ABS_TOLERANCE
            and post_update_loss_difference <= MAX_ABS_TOLERANCE
        ),
        "native_rmsnorm_is_not_misreported_as_layernorm_parity": (
            rmsnorm_counterfactual_difference > 1e-3
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(f"cross-framework parity failed: {assertions}")

    execution_device = next(iter(jax_logits.devices()))
    report: dict[str, object] = {
        "schema_version": GPT_CROSS_FRAMEWORK_PARITY_VERSION,
        "runtime": {
            "torch_version": torch.__version__,
            "jax_version": jax.__version__,
            "jaxlib_version": getattr(jaxlib, "__version__", "unknown"),
            "optax_version": optax.__version__,
            "jax_backend": getattr(execution_device, "platform", "unknown"),
            "jax_device": str(execution_device),
            "torch_device": "cpu",
            "dtype": "float32",
        },
        "fixture": {
            "config": {
                "vocab_size": config.vocab_size,
                "context_length": config.context_length,
                "model_dim": config.model_dim,
                "num_heads": config.num_heads,
                "num_layers": config.num_layers,
                "mlp_ratio": config.mlp_ratio,
                "dropout": config.dropout,
                "linear_bias": config.bias,
                "normalization": "LayerNorm with affine scale/bias",
                "normalization_epsilon": 1e-5,
                "gelu": "tanh approximation",
                "token_embedding_lm_head_tied": True,
            },
            "input_ids": input_ids.tolist(),
            "targets": targets.tolist(),
            "ignored_target_count": int((targets == -100).sum().item()),
            "parameter_initializer": (
                "name-ordered analytic sin/cos; no framework RNG"
            ),
            "optimizer": "plain SGD without momentum or weight decay",
            "learning_rate": LEARNING_RATE,
            "comparison_tolerance": MAX_ABS_TOLERANCE,
        },
        "observation": {
            "torch_loss_before_step": float(torch_loss.detach().item()),
            "jax_loss_before_step": float(jax_loss),
            "torch_loss_after_step": float(torch_loss_after.item()),
            "jax_loss_after_step": float(jax_loss_after),
            "torch_logits_before_step": torch_logits.detach().tolist(),
            "jax_logits_before_step": np.asarray(jax_logits).tolist(),
        },
        "comparison": comparisons,
        "assertions": assertions,
        "scope": {
            "same_initial_parameter_values_compared": True,
            "layernorm_bias_epsilon_gelu_mask_and_tying_aligned": True,
            "masked_cross_entropy_forward_compared": True,
            "every_unique_parameter_gradient_compared": True,
            "plain_sgd_one_step_compared": True,
            "post_update_forward_compared": True,
            "native_rmsnorm_architecture_counterfactual_executed": True,
            "jax_cpu_execution_forced": True,
            "framework_rng_equivalence_claimed": False,
            "adamw_optimizer_state_or_schedule_compared": False,
            "dropout_rng_or_stochastic_sampling_compared": False,
            "jit_compile_or_async_timing_compared": False,
            "cuda_tpu_multi_device_or_sharding_executed": False,
            "large_model_training_convergence_or_performance_proved": False,
        },
    }
    report["report_fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    return report


def run_gpt_cross_framework_parity_control() -> dict[str, object]:
    """Run the parity fixture on CPU even when JAX accelerators are present."""

    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("a JAX CPU device is required for the parity control")
    with jax.default_device(cpu_devices[0]):
        return _run_gpt_cross_framework_parity_control()


__all__ = [
    "GPT_CROSS_FRAMEWORK_PARITY_VERSION",
    "layer_norm",
    "layernorm_jax_forward",
    "run_gpt_cross_framework_parity_control",
    "torch_model_to_layernorm_jax_params",
]
