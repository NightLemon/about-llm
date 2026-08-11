"""Strict residual-stream activation patching for the teaching MiniGPT."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from about_llm.from_scratch.gpt_torch import MiniGPT


@dataclass(frozen=True)
class ResidualPatchResult:
    """Raw logit-difference metrics for one aligned clean/corrupt intervention."""

    layer_index: int
    patched_positions: tuple[int, ...]
    metric_position: int
    positive_token_id: int
    negative_token_id: int
    clean_metric: float
    corrupted_metric: float
    patched_metric: float
    normalized_recovery: float

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "patched_positions": list(self.patched_positions),
            "metric_position": self.metric_position,
            "positive_token_id": self.positive_token_id,
            "negative_token_id": self.negative_token_id,
            "clean_metric": self.clean_metric,
            "corrupted_metric": self.corrupted_metric,
            "patched_metric": self.patched_metric,
            "normalized_recovery": self.normalized_recovery,
        }


def capture_block_residual(
    model: MiniGPT, input_ids: Tensor, *, layer_index: int
) -> tuple[Tensor, Tensor]:
    """Return logits and a detached clone of one block's post-residual output."""

    _validate_model_and_input(model, input_ids)
    block = _block(model, layer_index)
    captures: list[Tensor] = []

    def capture(_: Any, __: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, Tensor):
            raise TypeError("MiniGPT block hook must receive one Tensor output")
        captures.append(output.detach().clone())

    handle = block.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            logits, _ = model(input_ids)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise RuntimeError(f"expected one block activation, captured {len(captures)}")
    return cast(Tensor, logits.detach().clone()), captures[0]


def patch_block_residual(
    model: MiniGPT,
    corrupted_input_ids: Tensor,
    *,
    layer_index: int,
    clean_activation: Tensor,
    positions: Sequence[int],
) -> Tensor:
    """Patch aligned batch/position residual vectors during a corrupted forward."""

    _validate_model_and_input(model, corrupted_input_ids)
    block = _block(model, layer_index)
    patch_positions = _positions(positions, corrupted_input_ids.shape[1])
    expected_shape = (
        corrupted_input_ids.shape[0],
        corrupted_input_ids.shape[1],
        model.config.model_dim,
    )
    if not isinstance(clean_activation, Tensor) or tuple(clean_activation.shape) != expected_shape:
        raise ValueError(
            f"clean_activation must have shape {expected_shape}, "
            f"got {getattr(clean_activation, 'shape', None)}"
        )
    if clean_activation.device != corrupted_input_ids.device:
        raise ValueError("clean_activation and corrupted_input_ids must share a device")
    clean = clean_activation.detach().clone()

    def patch(_: Any, __: tuple[Any, ...], output: Any) -> Tensor:
        if not isinstance(output, Tensor):
            raise TypeError("MiniGPT block hook must receive one Tensor output")
        if tuple(output.shape) != expected_shape:
            raise ValueError("hook output shape changed from the validated MiniGPT contract")
        if output.dtype != clean.dtype or output.device != clean.device:
            raise ValueError("clean activation dtype/device does not match hook output")
        patched = output.clone()
        patched[:, patch_positions, :] = clean[:, patch_positions, :]
        return patched

    handle = block.register_forward_hook(patch)
    try:
        with torch.inference_mode():
            logits, _ = model(corrupted_input_ids)
    finally:
        handle.remove()
    return cast(Tensor, logits.detach().clone())


def mean_logit_difference(
    logits: Tensor,
    *,
    position: int,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    """Mean batch logit difference at one sequence position."""

    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [batch, time, vocabulary]")
    batch, sequence_length, vocabulary_size = logits.shape
    if batch <= 0 or sequence_length <= 0 or vocabulary_size <= 1:
        raise ValueError("logits dimensions must be non-empty and vocabulary > 1")
    _index(position, sequence_length, "metric position")
    _index(positive_token_id, vocabulary_size, "positive token id")
    _index(negative_token_id, vocabulary_size, "negative token id")
    if positive_token_id == negative_token_id:
        raise ValueError("positive and negative token ids must differ")
    value = (
        logits[:, position, positive_token_id]
        - logits[:, position, negative_token_id]
    ).mean()
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("logit difference must be finite")
    return result


def normalized_patch_recovery(
    *,
    clean_metric: float,
    corrupted_metric: float,
    patched_metric: float,
    minimum_absolute_denominator: float = 1e-8,
) -> float:
    """Compute an unclipped recovery score and reject an unstable denominator."""

    values = (clean_metric, corrupted_metric, patched_metric)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("patch metrics must be finite numbers")
    if (
        isinstance(minimum_absolute_denominator, bool)
        or not isinstance(minimum_absolute_denominator, (int, float))
        or not math.isfinite(minimum_absolute_denominator)
        or minimum_absolute_denominator <= 0
    ):
        raise ValueError("minimum_absolute_denominator must be finite and positive")
    denominator = float(clean_metric - corrupted_metric)
    if abs(denominator) < minimum_absolute_denominator:
        raise ValueError("clean-corrupted metric denominator is too small")
    recovery = float((patched_metric - corrupted_metric) / denominator)
    if not math.isfinite(recovery):
        raise ValueError("normalized recovery must be finite")
    return recovery


def run_residual_patch_experiment(
    model: MiniGPT,
    clean_input_ids: Tensor,
    corrupted_input_ids: Tensor,
    *,
    layer_index: int,
    positions: Sequence[int],
    metric_position: int,
    positive_token_id: int,
    negative_token_id: int,
    minimum_absolute_denominator: float = 1e-8,
) -> ResidualPatchResult:
    """Run aligned clean, corrupt, and patched forwards with one fixed metric."""

    _validate_model_and_input(model, clean_input_ids)
    _validate_model_and_input(model, corrupted_input_ids)
    if tuple(clean_input_ids.shape) != tuple(corrupted_input_ids.shape):
        raise ValueError("clean and corrupted input ids must have identical shape")
    if clean_input_ids.device != corrupted_input_ids.device:
        raise ValueError("clean and corrupted input ids must share a device")
    patch_positions = _positions(positions, clean_input_ids.shape[1])
    clean_logits, clean_activation = capture_block_residual(
        model, clean_input_ids, layer_index=layer_index
    )
    with torch.inference_mode():
        corrupted_logits, _ = model(corrupted_input_ids)
    patched_logits = patch_block_residual(
        model,
        corrupted_input_ids,
        layer_index=layer_index,
        clean_activation=clean_activation,
        positions=patch_positions,
    )

    def metric(logits: Tensor) -> float:
        return mean_logit_difference(
            logits,
            position=metric_position,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
        )

    clean_metric = metric(clean_logits)
    corrupted_metric = metric(corrupted_logits)
    patched_metric = metric(patched_logits)
    return ResidualPatchResult(
        layer_index=layer_index,
        patched_positions=patch_positions,
        metric_position=metric_position,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        clean_metric=clean_metric,
        corrupted_metric=corrupted_metric,
        patched_metric=patched_metric,
        normalized_recovery=normalized_patch_recovery(
            clean_metric=clean_metric,
            corrupted_metric=corrupted_metric,
            patched_metric=patched_metric,
            minimum_absolute_denominator=minimum_absolute_denominator,
        ),
    )


def _validate_model_and_input(model: MiniGPT, input_ids: Tensor) -> None:
    if not isinstance(model, MiniGPT):
        raise TypeError("model must be the teaching MiniGPT")
    if model.training:
        raise ValueError("activation patching requires model.eval() for deterministic hooks")
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, time]")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must use torch.long token ids")
    if input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
        raise ValueError("input_ids dimensions must be non-empty")
    if input_ids.shape[1] > model.config.context_length:
        raise ValueError("input sequence exceeds the model context length")
    if input_ids.device != next(model.parameters()).device:
        raise ValueError("input_ids and model parameters must share a device")
    if bool(torch.any(input_ids < 0)) or bool(
        torch.any(input_ids >= model.config.vocab_size)
    ):
        raise ValueError("input_ids contain a token outside the model vocabulary")


def _block(model: MiniGPT, layer_index: int) -> Any:
    _index(layer_index, len(model.blocks), "layer index")
    return model.blocks[layer_index]


def _positions(values: Sequence[int], sequence_length: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("positions must be an integer sequence")
    positions = tuple(values)
    if not positions:
        raise ValueError("positions must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in positions):
        raise ValueError("positions must contain integers")
    if len(positions) != len(set(positions)):
        raise ValueError("positions must not contain duplicates")
    for position in positions:
        _index(position, sequence_length, "patch position")
    return positions


def _index(value: int, upper_bound: int, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= upper_bound
    ):
        raise ValueError(f"{label} must be in [0, {upper_bound})")
