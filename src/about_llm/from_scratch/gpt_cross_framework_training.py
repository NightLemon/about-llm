"""PyTorch/JAX AdamW trajectory parity on one explicit stochastic contract."""

from __future__ import annotations

import hashlib
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
from torch.nn import functional as torch_functional

from about_llm.from_scratch.gpt_cross_framework import (
    PyTree,
    _arrays_by_torch_name,
    _canonical_bytes,
    _fill_deterministic_parameters,
    _fixture_config,
    _jax_config,
    _layernorm_attention,
    _max_abs_difference,
    _torch_arrays,
    _tree_differences,
    layer_norm,
    torch_model_to_layernorm_jax_params,
)
from about_llm.from_scratch.gpt_jax import cross_entropy_loss
from about_llm.from_scratch.gpt_torch import MiniGPT

GPT_CROSS_FRAMEWORK_TRAINING_PARITY_VERSION = (
    "about-llm.gpt-cross-framework-training-parity.v1"
)
TRAINING_TOLERANCE = 5e-6
LEARNING_RATES = (0.02, 0.01, 0.005)
WEIGHT_DECAY = 0.03
BETA1 = 0.9
BETA2 = 0.95
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 0.08
DROPOUT_RATE = 0.25
DROPOUT_SEED = 20260814

_INPUT_IDS = (
    ((0, 1, 2, 3), (3, 2, 1, 0)),
    ((1, 2, 3, 4), (4, 3, 2, 1)),
    ((2, 3, 4, 5), (5, 4, 3, 2)),
)
_TARGETS = (
    ((1, 2, 3, 4), (2, -100, 0, 5)),
    ((2, 3, 4, 5), (3, 2, -100, 0)),
    ((3, 4, 5, 6), (4, -100, 2, 1)),
)


def _mask_sha256(mask: np.ndarray[Any, np.dtype[np.float32]]) -> str:
    payload = np.asarray(mask, dtype="<f4", order="C").tobytes(order="C")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _materialized_dropout_masks() -> tuple[np.ndarray[Any, np.dtype[np.float32]], ...]:
    generator = np.random.Generator(np.random.PCG64(DROPOUT_SEED))
    masks: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    shape = (2, 4, 8)
    for _ in LEARNING_RATES:
        keep = generator.random(shape) >= DROPOUT_RATE
        mask = np.asarray(
            np.asarray(keep, dtype=np.float32)
            / np.float32(1.0 - DROPOUT_RATE),
            dtype=np.float32,
        )
        masks.append(cast(np.ndarray[Any, np.dtype[np.float32]], mask))
    return tuple(masks)


def layernorm_jax_forward_with_embedding_mask(
    params: PyTree,
    input_ids: Array,
    embedding_mask: Array,
) -> Array:
    """Run the LayerNorm parity model with one supplied inverted-dropout mask."""

    config = _jax_config(_fixture_config())
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have two dimensions")
    sequence_length = input_ids.shape[1]
    if sequence_length > config.context_length:
        raise ValueError("input sequence exceeds the configured context")
    expected_shape = (input_ids.shape[0], sequence_length, config.model_dim)
    if embedding_mask.shape != expected_shape:
        raise ValueError(
            f"embedding_mask must have shape {expected_shape}, got "
            f"{embedding_mask.shape}"
        )
    if not jnp.issubdtype(embedding_mask.dtype, jnp.floating):
        raise TypeError("embedding_mask must have a floating dtype")

    hidden_states = params["token_embedding"][input_ids]
    hidden_states = hidden_states + params["position_embedding"][
        jnp.arange(sequence_length)
    ]
    hidden_states = hidden_states * embedding_mask
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


def _torch_forward_with_embedding_mask(
    model: MiniGPT,
    input_ids: Tensor,
    targets: Tensor,
    embedding_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    if model.config.dropout != 0.0:
        raise ValueError("native module dropout must be disabled")
    batch_size, sequence_length = input_ids.shape
    expected_shape = (batch_size, sequence_length, model.config.model_dim)
    if tuple(embedding_mask.shape) != expected_shape:
        raise ValueError("embedding_mask shape does not match model activations")
    positions = torch.arange(sequence_length, device=input_ids.device)
    hidden_states = (
        model.token_embedding(input_ids) + model.position_embedding(positions)
    ) * embedding_mask
    for block in model.blocks:
        hidden_states = block(hidden_states)
    logits = model.lm_head(model.final_norm(hidden_states))
    loss = torch_functional.cross_entropy(
        logits.reshape(-1, model.config.vocab_size),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return logits, loss


def _torch_optimizer_arrays(
    model: MiniGPT,
    optimizer: torch.optim.AdamW,
    field: str,
) -> dict[str, np.ndarray[Any, Any]]:
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for name, parameter in model.named_parameters():
        value = optimizer.state[parameter].get(field)
        if not isinstance(value, Tensor):
            raise AssertionError(f"optimizer state {field} missing for {name}")
        arrays[name] = value.detach().cpu().numpy().copy()
    return arrays


def _torch_optimizer_step(model: MiniGPT, optimizer: torch.optim.AdamW) -> int:
    steps: set[int] = set()
    for name, parameter in model.named_parameters():
        value = optimizer.state[parameter].get("step")
        if not isinstance(value, Tensor) or value.numel() != 1:
            raise AssertionError(f"optimizer step missing for {name}")
        scalar = float(value.detach().cpu().item())
        if not math.isfinite(scalar) or scalar < 0 or not scalar.is_integer():
            raise AssertionError(f"invalid optimizer step for {name}")
        steps.add(int(scalar))
    if len(steps) != 1:
        raise AssertionError("per-parameter AdamW steps diverged")
    return steps.pop()


def _jax_optimizer() -> optax.GradientTransformation:
    schedule_values = jnp.asarray(LEARNING_RATES, dtype=jnp.float32)

    def schedule(count: Array) -> Array:
        index = jnp.minimum(count, len(LEARNING_RATES) - 1)
        return schedule_values[index]

    return optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adamw(
            learning_rate=schedule,
            b1=BETA1,
            b2=BETA2,
            eps=ADAM_EPSILON,
            weight_decay=WEIGHT_DECAY,
        ),
    )


def _jax_state_parts(
    optimizer_state: optax.OptState,
) -> tuple[int, int, PyTree, PyTree]:
    state = cast(Any, optimizer_state)
    adam_state = state[1][0]
    schedule_state = state[1][2]
    return (
        int(np.asarray(adam_state.count)),
        int(np.asarray(schedule_state.count)),
        cast(PyTree, adam_state.mu),
        cast(PyTree, adam_state.nu),
    )


def _jax_loss_and_gradients(
    params: PyTree,
    input_ids: Array,
    targets: Array,
    mask: Array,
) -> tuple[Array, PyTree]:
    def loss_function(current_params: PyTree) -> Array:
        logits = layernorm_jax_forward_with_embedding_mask(
            current_params,
            input_ids,
            mask,
        )
        return cross_entropy_loss(logits, targets)

    loss, gradients = jax.value_and_grad(loss_function)(params)
    return loss, cast(PyTree, gradients)


def _run_wrong_mask_counterfactual(
    initial_params: PyTree,
    masks: tuple[np.ndarray[Any, np.dtype[np.float32]], ...],
) -> PyTree:
    optimizer = _jax_optimizer()
    state = optimizer.init(initial_params)
    params = initial_params
    for input_rows, target_rows, mask in zip(
        _INPUT_IDS,
        _TARGETS,
        masks,
        strict=True,
    ):
        wrong_mask = np.roll(mask, shift=1, axis=-1).copy()
        _, gradients = _jax_loss_and_gradients(
            params,
            jnp.asarray(input_rows, dtype=jnp.int32),
            jnp.asarray(target_rows, dtype=jnp.int32),
            jnp.asarray(wrong_mask),
        )
        updates, state = optimizer.update(gradients, state, params)
        params = cast(PyTree, optax.apply_updates(params, updates))
    return params


def _run_gpt_cross_framework_training_parity_control() -> dict[str, object]:
    config = _fixture_config()
    model = MiniGPT(config).train()
    _fill_deterministic_parameters(model)
    initial_params = torch_model_to_layernorm_jax_params(model)
    jax_params = initial_params
    masks = _materialized_dropout_masks()

    torch_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATES[0],
        betas=(BETA1, BETA2),
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )
    jax_optimizer = _jax_optimizer()
    jax_optimizer_state = jax_optimizer.init(jax_params)
    clip_transform = optax.clip_by_global_norm(MAX_GRAD_NORM)

    step_observations: list[dict[str, object]] = []
    for step_index, (input_rows, target_rows, mask, learning_rate) in enumerate(
        zip(_INPUT_IDS, _TARGETS, masks, LEARNING_RATES, strict=True),
        start=1,
    ):
        input_ids = torch.tensor(input_rows, dtype=torch.int64)
        targets = torch.tensor(target_rows, dtype=torch.int64)
        torch_mask = torch.from_numpy(mask.copy())
        jax_input_ids = jnp.asarray(input_rows, dtype=jnp.int32)
        jax_targets = jnp.asarray(target_rows, dtype=jnp.int32)
        jax_mask = jnp.asarray(mask)

        for group in torch_optimizer.param_groups:
            group["lr"] = learning_rate
        torch_optimizer.zero_grad(set_to_none=True)
        _, torch_loss = _torch_forward_with_embedding_mask(
            model,
            input_ids,
            targets,
            torch_mask,
        )
        torch_loss.backward()  # type: ignore[no-untyped-call]
        jax_loss, jax_gradients = _jax_loss_and_gradients(
            jax_params,
            jax_input_ids,
            jax_targets,
            jax_mask,
        )

        raw_gradient_differences = _tree_differences(
            _torch_arrays(model, gradients=True),
            jax_gradients,
        )
        torch_preclip_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
            error_if_nonfinite=True,
            foreach=False,
        )
        jax_preclip_norm = optax.tree.norm(jax_gradients)
        clipped_jax_gradients, _ = clip_transform.update(
            jax_gradients,
            clip_transform.init(jax_params),
        )
        clipped_gradient_differences = _tree_differences(
            _torch_arrays(model, gradients=True),
            cast(PyTree, clipped_jax_gradients),
        )

        torch_optimizer.step()
        updates, jax_optimizer_state = jax_optimizer.update(
            jax_gradients,
            jax_optimizer_state,
            jax_params,
        )
        jax_params = cast(PyTree, optax.apply_updates(jax_params, updates))

        parameter_differences = _tree_differences(
            _torch_arrays(model, gradients=False),
            jax_params,
        )
        jax_adam_count, jax_schedule_count, jax_mu, jax_nu = (
            _jax_state_parts(jax_optimizer_state)
        )
        first_moment_differences = _tree_differences(
            _torch_optimizer_arrays(model, torch_optimizer, "exp_avg"),
            jax_mu,
        )
        second_moment_differences = _tree_differences(
            _torch_optimizer_arrays(model, torch_optimizer, "exp_avg_sq"),
            jax_nu,
        )
        torch_step = _torch_optimizer_step(model, torch_optimizer)

        with torch.no_grad():
            torch_logits_after, torch_loss_after = (
                _torch_forward_with_embedding_mask(
                    model,
                    input_ids,
                    targets,
                    torch_mask,
                )
            )
        jax_logits_after = layernorm_jax_forward_with_embedding_mask(
            jax_params,
            jax_input_ids,
            jax_mask,
        )
        jax_loss_after = cross_entropy_loss(jax_logits_after, jax_targets)

        step_observations.append(
            {
                "step": step_index,
                "learning_rate": learning_rate,
                "mask_sha256": _mask_sha256(mask),
                "kept_elements": int(np.count_nonzero(mask)),
                "torch_loss_before_step": float(torch_loss.detach().item()),
                "jax_loss_before_step": float(jax_loss),
                "loss_before_step_abs_difference": abs(
                    float(torch_loss.detach().item()) - float(jax_loss)
                ),
                "torch_preclip_gradient_norm": float(torch_preclip_norm.item()),
                "jax_preclip_gradient_norm": float(jax_preclip_norm),
                "preclip_gradient_norm_abs_difference": abs(
                    float(torch_preclip_norm.item()) - float(jax_preclip_norm)
                ),
                "raw_gradient_global_max_abs_difference": max(
                    raw_gradient_differences.values()
                ),
                "clipped_gradient_global_max_abs_difference": max(
                    clipped_gradient_differences.values()
                ),
                "parameter_global_max_abs_difference": max(
                    parameter_differences.values()
                ),
                "first_moment_global_max_abs_difference": max(
                    first_moment_differences.values()
                ),
                "second_moment_global_max_abs_difference": max(
                    second_moment_differences.values()
                ),
                "torch_adam_step": torch_step,
                "jax_adam_count": jax_adam_count,
                "jax_schedule_count": jax_schedule_count,
                "post_step_logits_max_abs_difference": _max_abs_difference(
                    torch_logits_after.detach().numpy(),
                    np.asarray(jax_logits_after),
                ),
                "post_step_loss_abs_difference": abs(
                    float(torch_loss_after.item()) - float(jax_loss_after)
                ),
            }
        )

    wrong_mask_params = _run_wrong_mask_counterfactual(initial_params, masks)
    final_jax_arrays = {
        name: np.asarray(value)
        for name, value in _arrays_by_torch_name(jax_params).items()
    }
    wrong_mask_difference = max(
        _tree_differences(final_jax_arrays, wrong_mask_params).values()
    )

    scalar_fields = (
        "loss_before_step_abs_difference",
        "preclip_gradient_norm_abs_difference",
        "raw_gradient_global_max_abs_difference",
        "clipped_gradient_global_max_abs_difference",
        "parameter_global_max_abs_difference",
        "first_moment_global_max_abs_difference",
        "second_moment_global_max_abs_difference",
        "post_step_logits_max_abs_difference",
        "post_step_loss_abs_difference",
    )
    maximum_differences = {
        field: max(cast(float, step[field]) for step in step_observations)
        for field in scalar_fields
    }
    assertions = {
        "all_three_steps_use_authored_schedule": [
            step["learning_rate"] for step in step_observations
        ]
        == list(LEARNING_RATES),
        "clipping_is_active_on_every_step": all(
            cast(float, step["torch_preclip_gradient_norm"])
            > MAX_GRAD_NORM
            and cast(float, step["jax_preclip_gradient_norm"])
            > MAX_GRAD_NORM
            for step in step_observations
        ),
        "raw_and_clipped_gradients_match": (
            maximum_differences["raw_gradient_global_max_abs_difference"]
            <= TRAINING_TOLERANCE
            and maximum_differences[
                "clipped_gradient_global_max_abs_difference"
            ]
            <= TRAINING_TOLERANCE
        ),
        "gradient_norms_match": (
            maximum_differences["preclip_gradient_norm_abs_difference"]
            <= TRAINING_TOLERANCE
        ),
        "adamw_moments_and_counts_match": (
            maximum_differences[
                "first_moment_global_max_abs_difference"
            ]
            <= TRAINING_TOLERANCE
            and maximum_differences[
                "second_moment_global_max_abs_difference"
            ]
            <= TRAINING_TOLERANCE
            and all(
                step["torch_adam_step"]
                == step["jax_adam_count"]
                == step["jax_schedule_count"]
                == step["step"]
                for step in step_observations
            )
        ),
        "updated_parameters_and_forward_match": (
            maximum_differences["parameter_global_max_abs_difference"]
            <= TRAINING_TOLERANCE
            and maximum_differences["post_step_logits_max_abs_difference"]
            <= TRAINING_TOLERANCE
            and maximum_differences["post_step_loss_abs_difference"]
            <= TRAINING_TOLERANCE
        ),
        "wrong_materialized_mask_diverges": wrong_mask_difference > 1e-4,
    }
    if not all(assertions.values()):
        raise AssertionError(
            f"cross-framework training parity failed: {assertions}; "
            f"maxima={maximum_differences}"
        )

    execution_device = next(iter(jax_params["token_embedding"].devices()))
    report: dict[str, object] = {
        "schema_version": GPT_CROSS_FRAMEWORK_TRAINING_PARITY_VERSION,
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
            "steps": len(LEARNING_RATES),
            "input_ids": _INPUT_IDS,
            "targets": _TARGETS,
            "parameter_initializer": (
                "name-ordered analytic sin/cos; no framework RNG"
            ),
            "dropout": {
                "site": "embedding sum only",
                "kind": "externally materialized inverted dropout",
                "rate": DROPOUT_RATE,
                "generator": "NumPy PCG64",
                "seed": DROPOUT_SEED,
                "mask_shape": [2, 4, 8],
                "mask_sha256": [_mask_sha256(mask) for mask in masks],
            },
            "optimizer": {
                "kind": "AdamW",
                "learning_rates": LEARNING_RATES,
                "beta1": BETA1,
                "beta2": BETA2,
                "epsilon": ADAM_EPSILON,
                "weight_decay": WEIGHT_DECAY,
                "weight_decay_mask": "all parameters",
                "max_grad_norm": MAX_GRAD_NORM,
            },
            "comparison_tolerance": TRAINING_TOLERANCE,
        },
        "steps": step_observations,
        "comparison": {
            "maximum_difference_across_steps": maximum_differences,
            "wrong_mask_final_parameter_max_abs_difference": (
                wrong_mask_difference
            ),
        },
        "assertions": assertions,
        "scope": {
            "same_initial_parameter_values_compared": True,
            "shared_materialized_embedding_dropout_masks_compared": True,
            "framework_native_rng_equivalence_claimed": False,
            "dropout_prng_state_advance_compared": False,
            "raw_and_global_norm_clipped_gradients_compared": True,
            "adamw_first_second_moments_and_count_compared": True,
            "learning_rate_schedule_compared": True,
            "all_parameter_weight_decay_compared": True,
            "norm_or_bias_weight_decay_mask_compared": False,
            "three_post_update_forwards_compared": True,
            "wrong_materialized_mask_counterfactual_executed": True,
            "checkpoint_resume_or_artifact_serialization_compared": False,
            "jit_compile_or_async_timing_compared": False,
            "cuda_tpu_multi_device_or_sharding_executed": False,
            "large_model_training_convergence_or_performance_proved": False,
        },
    }
    report["report_fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    return report


def run_gpt_cross_framework_training_parity_control() -> dict[str, object]:
    """Run the authored training trajectory on a JAX CPU device."""

    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("a JAX CPU device is required for the parity control")
    with jax.default_device(cpu_devices[0]):
        return _run_gpt_cross_framework_training_parity_control()


__all__ = [
    "GPT_CROSS_FRAMEWORK_TRAINING_PARITY_VERSION",
    "layernorm_jax_forward_with_embedding_mask",
    "run_gpt_cross_framework_training_parity_control",
]
