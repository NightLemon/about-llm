"""Reviewed activation-patching control for one immutable Qwen checkpoint.

The import surface deliberately has no torch or Transformers dependency.  The
heavy dependencies are imported only by the execution entry point so a core
wheel remains importable without optional extras.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    download_checkpoint_snapshot,
    verify_checkpoint_snapshot,
)
from about_llm.integrations.transformers_tools import parameter_report
from about_llm.llmops import artifact_fingerprint

TARGET_ACTIVATION_PATCHING_CONTROL_VERSION = (
    "about-llm.target-activation-patching-control.v1"
)
TARGET_ACTIVATION_PATCHING_REPORT_VERSION = (
    "about-llm.target-activation-patching-control-report.v1"
)
TARGET_ACTIVATION_PATCHING_EVIDENCE_BOUNDARY = (
    "This authored fixed-protocol control verifies selected files from one immutable "
    "Qwen2.5-0.5B-Instruct revision, loads those verified local paths with "
    "trust_remote_code disabled, executes CPU FP32 eager forward hooks, and checks "
    "two factual next-token contrasts plus constructive and causal-structure controls. "
    "It is not an externally timestamped preregistration, a discovery of a unique or "
    "natural circuit, an unbiased effect estimate, an SAE experiment, or evidence about "
    "other prompts, facts, languages, checkpoints, hook sites, dtypes, devices, kernels, "
    "runtimes, model quality, factual reliability, training data, safety, or production "
    "behavior. File hashes do not authenticate the publisher, eliminate the "
    "verification-to-loader-reopen TOCTOU window, or prove licensing compatibility."
)


@dataclass(frozen=True)
class ActivationPatchingProtocol:
    """JSON-projectable intervention contract fixed independently of model outputs."""

    model_id: str
    revision: str
    expected_model_class: str
    expected_model_type: str
    expected_hidden_size: int
    expected_layer_count: int
    system_message: str
    clean_user_message: str
    corrupt_user_message: str
    clean_input_ids: tuple[int, ...]
    corrupt_input_ids: tuple[int, ...]
    source_position: int
    metric_position: int
    positive_token_text: str
    positive_token_id: int
    negative_token_text: str
    negative_token_id: int
    source_layer_indices: tuple[int, ...]
    future_token_text: str
    future_token_id: int
    expected_clean_top_token_id: int | None
    expected_corrupt_top_token_id: int | None
    minimum_clean_corrupt_gap: float
    control_tolerance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "control_version": TARGET_ACTIVATION_PATCHING_CONTROL_VERSION,
            "model_id": self.model_id,
            "revision": self.revision,
            "expected_model_class": self.expected_model_class,
            "expected_model_type": self.expected_model_type,
            "expected_hidden_size": self.expected_hidden_size,
            "expected_layer_count": self.expected_layer_count,
            "messages": {
                "system": self.system_message,
                "clean_user": self.clean_user_message,
                "corrupt_user": self.corrupt_user_message,
            },
            "clean_input_ids": list(self.clean_input_ids),
            "corrupt_input_ids": list(self.corrupt_input_ids),
            "source_position": self.source_position,
            "metric_position": self.metric_position,
            "metric": {
                "name": "positive-minus-negative-next-token-logit",
                "positive_token_text": self.positive_token_text,
                "positive_token_id": self.positive_token_id,
                "negative_token_text": self.negative_token_text,
                "negative_token_id": self.negative_token_id,
                "normalized_recovery_clipped": False,
            },
            "source_layer_indices": list(self.source_layer_indices),
            "source_layer_selection_rule": "first, lower-middle, and final decoder layer",
            "constructive_controls": {
                "full_prefix_layer_index": self.source_layer_indices[0],
                "final_readout_layer_index": self.source_layer_indices[-1],
            },
            "causal_controls": {
                "final_layer_source_position": self.source_position,
                "future_position": len(self.clean_input_ids),
                "future_token_text": self.future_token_text,
                "future_token_id": self.future_token_id,
            },
            "expected_clean_top_token_id": self.expected_clean_top_token_id,
            "expected_corrupt_top_token_id": self.expected_corrupt_top_token_id,
            "minimum_clean_corrupt_gap": self.minimum_clean_corrupt_gap,
            "control_tolerance": self.control_tolerance,
        }

    @property
    def fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self.to_dict())


QWEN2_5_0_5B_ACTIVATION_PATCHING_PROTOCOL = ActivationPatchingProtocol(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    expected_model_class="Qwen2ForCausalLM",
    expected_model_type="qwen2",
    expected_hidden_size=896,
    expected_layer_count=24,
    system_message="Answer the factual prompt concisely.",
    clean_user_message="The capital of France is",
    corrupt_user_message="The capital of Germany is",
    clean_input_ids=(
        151644,
        8948,
        198,
        16141,
        279,
        59901,
        9934,
        3529,
        285,
        974,
        13,
        151645,
        198,
        151644,
        872,
        198,
        785,
        6722,
        315,
        9625,
        374,
        151645,
        198,
        151644,
        77091,
        198,
    ),
    corrupt_input_ids=(
        151644,
        8948,
        198,
        16141,
        279,
        59901,
        9934,
        3529,
        285,
        974,
        13,
        151645,
        198,
        151644,
        872,
        198,
        785,
        6722,
        315,
        9856,
        374,
        151645,
        198,
        151644,
        77091,
        198,
    ),
    source_position=19,
    metric_position=25,
    positive_token_text="Paris",
    positive_token_id=59604,
    negative_token_text="Berlin",
    negative_token_id=94409,
    source_layer_indices=(0, 11, 23),
    future_token_text=" unrelated",
    future_token_id=45205,
    expected_clean_top_token_id=59604,
    expected_corrupt_top_token_id=94409,
    minimum_clean_corrupt_gap=1.0,
    control_tolerance=1e-5,
)


def normalized_patch_recovery(
    *, clean_metric: float, corrupt_metric: float, patched_metric: float
) -> float:
    """Return unclipped normalized recovery and reject an unstable denominator."""

    values = (clean_metric, corrupt_metric, patched_metric)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("patch metrics must be finite numbers")
    denominator = clean_metric - corrupt_metric
    if abs(denominator) < 1e-8:
        raise ValueError("clean-corrupt metric denominator is too small")
    return (patched_metric - corrupt_metric) / denominator


def _validate_protocol(protocol: ActivationPatchingProtocol) -> None:
    if len(protocol.clean_input_ids) != len(protocol.corrupt_input_ids):
        raise ValueError("clean and corrupt input ids must have identical length")
    sequence_length = len(protocol.clean_input_ids)
    if sequence_length < 2:
        raise ValueError("protocol inputs must contain at least two tokens")
    differences = tuple(
        index
        for index, (clean, corrupt) in enumerate(
            zip(protocol.clean_input_ids, protocol.corrupt_input_ids, strict=True)
        )
        if clean != corrupt
    )
    if differences != (protocol.source_position,):
        raise ValueError("clean/corrupt inputs must differ only at source_position")
    if protocol.metric_position != sequence_length - 1:
        raise ValueError("the reviewed next-token metric must be at the final prompt token")
    if protocol.positive_token_id == protocol.negative_token_id:
        raise ValueError("positive and negative token ids must differ")
    token_ids = (*protocol.clean_input_ids, *protocol.corrupt_input_ids)
    if min(token_ids) < 0:
        raise ValueError("input token ids must be non-negative")
    if protocol.future_token_id < 0:
        raise ValueError("future token id must be non-negative")
    if not protocol.source_layer_indices:
        raise ValueError("source layer indices must not be empty")
    if tuple(sorted(set(protocol.source_layer_indices))) != protocol.source_layer_indices:
        raise ValueError("source layer indices must be sorted and unique")
    if protocol.source_layer_indices[0] != 0:
        raise ValueError("the constructive prefix control must use the first layer")
    if protocol.source_layer_indices[-1] != protocol.expected_layer_count - 1:
        raise ValueError("the causal source control must include the final layer")
    if protocol.minimum_clean_corrupt_gap <= 0 or not math.isfinite(
        protocol.minimum_clean_corrupt_gap
    ):
        raise ValueError("minimum_clean_corrupt_gap must be finite and positive")
    if protocol.control_tolerance <= 0 or not math.isfinite(protocol.control_tolerance):
        raise ValueError("control_tolerance must be finite and positive")


def _decoder_layers(model: Any, protocol: ActivationPatchingProtocol) -> Any:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or not hasattr(layers, "__len__") or not hasattr(
        layers, "__getitem__"
    ):
        raise ValueError("model.model.layers is not an indexable decoder-layer sequence")
    if len(layers) != protocol.expected_layer_count:
        raise ValueError("loaded decoder layer count does not match protocol")
    return layers


def _hidden_from_layer_output(output: Any) -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("torch is required for activation patching") from error
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("decoder layer hook must expose a Tensor or tuple[Tensor, ...]")


def _replace_hidden_in_layer_output(output: Any, hidden: Any) -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("torch is required for activation patching") from error
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (hidden, *output[1:])
    raise TypeError("decoder layer hook must expose a Tensor or tuple[Tensor, ...]")


def _forward_logits(model: Any, input_ids: Any) -> Any:
    output = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = getattr(output, "logits", None)
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("torch is required for activation patching") from error
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("model forward must return rank-3 logits")
    if tuple(logits.shape[:2]) != tuple(input_ids.shape):
        raise ValueError("model logits batch/time axes do not match input ids")
    return logits.detach().clone()


def _capture_layers(
    model: Any, input_ids: Any, *, layer_indices: Sequence[int]
) -> tuple[Any, dict[int, Any]]:
    protocol_count = max(layer_indices) + 1
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if (
        layers is None
        or not hasattr(layers, "__len__")
        or not hasattr(layers, "__getitem__")
        or len(layers) < protocol_count
    ):
        raise ValueError("requested decoder layer does not exist")
    captures: dict[int, list[Any]] = {index: [] for index in layer_indices}
    handles: list[Any] = []
    for index in layer_indices:
        def capture(_: Any, __: tuple[Any, ...], output: Any, *, site: int = index) -> None:
            hidden = _hidden_from_layer_output(output)
            captures[site].append(hidden.detach().clone())

        handles.append(layers[index].register_forward_hook(capture))
    try:
        logits = _forward_logits(model, input_ids)
    finally:
        for handle in handles:
            handle.remove()
    result: dict[int, Any] = {}
    for index in layer_indices:
        if len(captures[index]) != 1:
            raise RuntimeError(f"expected one capture at decoder layer {index}")
        result[index] = captures[index][0]
    return logits, result


def _patch_layer(
    model: Any,
    input_ids: Any,
    *,
    layer_index: int,
    clean_activation: Any,
    positions: Sequence[int],
) -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("torch is required for activation patching") from error
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if (
        layers is None
        or not hasattr(layers, "__len__")
        or not hasattr(layers, "__getitem__")
        or layer_index < 0
        or layer_index >= len(layers)
    ):
        raise ValueError("patch layer index is outside the decoder")
    patch_positions = tuple(positions)
    if not patch_positions or len(patch_positions) != len(set(patch_positions)):
        raise ValueError("patch positions must be non-empty and unique")
    sequence_length = int(input_ids.shape[1])
    if any(
        isinstance(position, bool)
        or not isinstance(position, int)
        or position < 0
        or position >= sequence_length
        for position in patch_positions
    ):
        raise ValueError("patch position is outside the input sequence")
    if not isinstance(clean_activation, torch.Tensor):
        raise TypeError("clean activation must be a Tensor")
    expected_shape = (int(input_ids.shape[0]), sequence_length)
    if tuple(clean_activation.shape[:2]) != expected_shape or clean_activation.ndim != 3:
        raise ValueError("clean activation batch/time shape does not match patched input")
    clean = clean_activation.detach().clone()

    def patch(_: Any, __: tuple[Any, ...], output: Any) -> Any:
        hidden = _hidden_from_layer_output(output)
        if (
            hidden.shape != clean.shape
            or hidden.dtype != clean.dtype
            or hidden.device != clean.device
        ):
            raise ValueError("clean activation shape/dtype/device does not match hook output")
        patched = hidden.clone()
        patched[:, patch_positions, :] = clean[:, patch_positions, :]
        return _replace_hidden_in_layer_output(output, patched)

    handle = layers[layer_index].register_forward_hook(patch)
    try:
        return _forward_logits(model, input_ids)
    finally:
        handle.remove()


def _metric(logits: Any, protocol: ActivationPatchingProtocol, *, position: int) -> float:
    value = (
        logits[:, position, protocol.positive_token_id]
        - logits[:, position, protocol.negative_token_id]
    ).mean()
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("logit-difference metric is not finite")
    return result


def _tensor_sha256(tensor: Any) -> str:
    data = tensor.detach().to(dtype=__import__("torch").float32).cpu().contiguous()
    prefix = json.dumps(
        {"dtype": "float32", "shape": list(data.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(prefix + data.numpy().tobytes()).hexdigest()


def _decode_one(tokenizer: Any, token_id: int) -> str:
    value = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(value, str):
        raise TypeError("tokenizer.decode must return a string")
    return value


def _condition(
    *,
    name: str,
    role: str,
    layer_index: int,
    positions: Sequence[int],
    metric_position: int,
    clean_metric: float,
    corrupt_metric: float,
    patched_logits: Any,
    protocol: ActivationPatchingProtocol,
) -> dict[str, object]:
    patched_metric = _metric(patched_logits, protocol, position=metric_position)
    return {
        "name": name,
        "role": role,
        "layer_index": layer_index,
        "patched_positions": list(positions),
        "metric_position": metric_position,
        "clean_metric": clean_metric,
        "corrupt_metric": corrupt_metric,
        "patched_metric": patched_metric,
        "normalized_recovery": normalized_patch_recovery(
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
            patched_metric=patched_metric,
        ),
        "patched_logits_sha256": _tensor_sha256(
            patched_logits[:, metric_position, :]
        ),
    }


def execute_loaded_activation_patching_control(
    protocol: ActivationPatchingProtocol,
    *,
    model: Any,
    tokenizer: Any,
) -> dict[str, object]:
    """Execute the fixed hook protocol against an already loaded causal LM."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("torch is required for activation patching") from error
    _validate_protocol(protocol)
    if type(model).__name__ != protocol.expected_model_class:
        raise ValueError("loaded model class does not match protocol")
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != protocol.expected_model_type:
        raise ValueError("loaded model type does not match protocol")
    if getattr(config, "hidden_size", None) != protocol.expected_hidden_size:
        raise ValueError("loaded hidden size does not match protocol")
    layers = _decoder_layers(model, protocol)
    vocabulary_size = int(getattr(config, "vocab_size", 0))
    all_ids = (
        *protocol.clean_input_ids,
        *protocol.corrupt_input_ids,
        protocol.positive_token_id,
        protocol.negative_token_id,
        protocol.future_token_id,
    )
    if vocabulary_size <= max(all_ids):
        raise ValueError("protocol token id is outside loaded model vocabulary")

    def messages(content: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": protocol.system_message},
            {"role": "user", "content": content},
        ]
    rendered_clean = tokenizer.apply_chat_template(
        messages(protocol.clean_user_message),
        tokenize=True,
        add_generation_prompt=True,
    )
    rendered_corrupt = tokenizer.apply_chat_template(
        messages(protocol.corrupt_user_message),
        tokenize=True,
        add_generation_prompt=True,
    )
    if tuple(rendered_clean) != protocol.clean_input_ids:
        raise ValueError("clean chat-template token ids drifted from reviewed protocol")
    if tuple(rendered_corrupt) != protocol.corrupt_input_ids:
        raise ValueError("corrupt chat-template token ids drifted from reviewed protocol")
    if _decode_one(tokenizer, protocol.positive_token_id) != protocol.positive_token_text:
        raise ValueError("positive token decoding drifted from reviewed protocol")
    if _decode_one(tokenizer, protocol.negative_token_id) != protocol.negative_token_text:
        raise ValueError("negative token decoding drifted from reviewed protocol")
    if _decode_one(tokenizer, protocol.future_token_id) != protocol.future_token_text:
        raise ValueError("future-control token decoding drifted from reviewed protocol")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = next(model.parameters()).device
    clean_ids = torch.tensor(
        [protocol.clean_input_ids], dtype=torch.long, device=device
    )
    corrupt_ids = torch.tensor(
        [protocol.corrupt_input_ids], dtype=torch.long, device=device
    )
    with torch.inference_mode():
        clean_logits, clean_activations = _capture_layers(
            model, clean_ids, layer_indices=protocol.source_layer_indices
        )
        corrupt_logits = _forward_logits(model, corrupt_ids)
        clean_metric = _metric(
            clean_logits, protocol, position=protocol.metric_position
        )
        corrupt_metric = _metric(
            corrupt_logits, protocol, position=protocol.metric_position
        )
        gap = clean_metric - corrupt_metric
        if gap < protocol.minimum_clean_corrupt_gap:
            raise RuntimeError("reviewed clean/corrupt behavior gap is absent")
        clean_top = int(
            torch.argmax(clean_logits[:, protocol.metric_position, :], dim=-1).item()
        )
        corrupt_top = int(
            torch.argmax(corrupt_logits[:, protocol.metric_position, :], dim=-1).item()
        )
        if (
            protocol.expected_clean_top_token_id is not None
            and clean_top != protocol.expected_clean_top_token_id
        ):
            raise RuntimeError("clean prompt top token drifted from reviewed behavior")
        if (
            protocol.expected_corrupt_top_token_id is not None
            and corrupt_top != protocol.expected_corrupt_top_token_id
        ):
            raise RuntimeError("corrupt prompt top token drifted from reviewed behavior")

        conditions: list[dict[str, object]] = []
        for layer_index in protocol.source_layer_indices:
            patched = _patch_layer(
                model,
                corrupt_ids,
                layer_index=layer_index,
                clean_activation=clean_activations[layer_index],
                positions=(protocol.source_position,),
            )
            conditions.append(
                _condition(
                    name=f"source_position_layer_{layer_index}",
                    role="preselected_source-site_intervention",
                    layer_index=layer_index,
                    positions=(protocol.source_position,),
                    metric_position=protocol.metric_position,
                    clean_metric=clean_metric,
                    corrupt_metric=corrupt_metric,
                    patched_logits=patched,
                    protocol=protocol,
                )
            )

        first_layer = protocol.source_layer_indices[0]
        full_prefix = tuple(range(len(protocol.clean_input_ids)))
        prefix_patched = _patch_layer(
            model,
            corrupt_ids,
            layer_index=first_layer,
            clean_activation=clean_activations[first_layer],
            positions=full_prefix,
        )
        conditions.append(
            _condition(
                name="full_prefix_first_layer_positive_control",
                role="constructive-positive-control",
                layer_index=first_layer,
                positions=full_prefix,
                metric_position=protocol.metric_position,
                clean_metric=clean_metric,
                corrupt_metric=corrupt_metric,
                patched_logits=prefix_patched,
                protocol=protocol,
            )
        )

        final_layer = protocol.source_layer_indices[-1]
        readout_patched = _patch_layer(
            model,
            corrupt_ids,
            layer_index=final_layer,
            clean_activation=clean_activations[final_layer],
            positions=(protocol.metric_position,),
        )
        conditions.append(
            _condition(
                name="readout_position_final_layer_positive_control",
                role="constructive-positive-control",
                layer_index=final_layer,
                positions=(protocol.metric_position,),
                metric_position=protocol.metric_position,
                clean_metric=clean_metric,
                corrupt_metric=corrupt_metric,
                patched_logits=readout_patched,
                protocol=protocol,
            )
        )

        clean_future_ids = torch.cat(
            (
                clean_ids,
                torch.tensor([[protocol.future_token_id]], dtype=torch.long, device=device),
            ),
            dim=1,
        )
        corrupt_future_ids = torch.cat(
            (
                corrupt_ids,
                torch.tensor([[protocol.future_token_id]], dtype=torch.long, device=device),
            ),
            dim=1,
        )
        clean_future_logits, clean_future_activations = _capture_layers(
            model, clean_future_ids, layer_indices=(first_layer,)
        )
        corrupt_future_logits = _forward_logits(model, corrupt_future_ids)
        clean_future_metric = _metric(
            clean_future_logits, protocol, position=protocol.metric_position
        )
        corrupt_future_metric = _metric(
            corrupt_future_logits, protocol, position=protocol.metric_position
        )
        future_patched = _patch_layer(
            model,
            corrupt_future_ids,
            layer_index=first_layer,
            clean_activation=clean_future_activations[first_layer],
            positions=(len(protocol.clean_input_ids),),
        )
        conditions.append(
            _condition(
                name="future_position_first_layer_negative_control",
                role="causal-future-negative-control",
                layer_index=first_layer,
                positions=(len(protocol.clean_input_ids),),
                metric_position=protocol.metric_position,
                clean_metric=clean_future_metric,
                corrupt_metric=corrupt_future_metric,
                patched_logits=future_patched,
                protocol=protocol,
            )
        )

    by_name = {cast(str, item["name"]): item for item in conditions}
    prefix_recovery = cast(
        float,
        by_name["full_prefix_first_layer_positive_control"]["normalized_recovery"],
    )
    readout_recovery = cast(
        float,
        by_name["readout_position_final_layer_positive_control"][
            "normalized_recovery"
        ],
    )
    future_recovery = cast(
        float,
        by_name["future_position_first_layer_negative_control"][
            "normalized_recovery"
        ],
    )
    final_source_recovery = cast(
        float,
        by_name[f"source_position_layer_{final_layer}"]["normalized_recovery"],
    )
    appended_clean_delta = clean_future_metric - clean_metric
    appended_corrupt_delta = corrupt_future_metric - corrupt_metric
    tolerance = protocol.control_tolerance
    checks = {
        "full_prefix_first_layer_recovery_is_one": abs(prefix_recovery - 1.0)
        <= tolerance,
        "readout_final_layer_recovery_is_one": abs(readout_recovery - 1.0)
        <= tolerance,
        "future_position_cannot_change_past_metric": abs(future_recovery) <= tolerance,
        "final_layer_source_cannot_change_other_position": abs(final_source_recovery)
        <= tolerance,
        "appended_future_token_preserves_clean_past_metric": abs(appended_clean_delta)
        <= tolerance,
        "appended_future_token_preserves_corrupt_past_metric": abs(appended_corrupt_delta)
        <= tolerance,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"activation-patching structural control failed: {failed}")

    return {
        "input_contract": {
            "batch_size": 1,
            "prompt_token_count": len(protocol.clean_input_ids),
            "clean_input_ids": list(protocol.clean_input_ids),
            "corrupt_input_ids": list(protocol.corrupt_input_ids),
            "only_changed_position": protocol.source_position,
            "clean_changed_token": _decode_one(
                tokenizer, protocol.clean_input_ids[protocol.source_position]
            ),
            "corrupt_changed_token": _decode_one(
                tokenizer, protocol.corrupt_input_ids[protocol.source_position]
            ),
            "metric_position": protocol.metric_position,
            "future_control_position": len(protocol.clean_input_ids),
        },
        "baseline": {
            "clean_metric": clean_metric,
            "corrupt_metric": corrupt_metric,
            "clean_minus_corrupt_gap": gap,
            "clean_top_token_id": clean_top,
            "clean_top_token_text": _decode_one(tokenizer, clean_top),
            "corrupt_top_token_id": corrupt_top,
            "corrupt_top_token_text": _decode_one(tokenizer, corrupt_top),
            "clean_metric_logits_sha256": _tensor_sha256(
                clean_logits[:, protocol.metric_position, :]
            ),
            "corrupt_metric_logits_sha256": _tensor_sha256(
                corrupt_logits[:, protocol.metric_position, :]
            ),
        },
        "conditions": conditions,
        "structural_controls": {
            "checks": checks,
            "control_tolerance": tolerance,
            "appended_future_clean_metric_delta": appended_clean_delta,
            "appended_future_corrupt_metric_delta": appended_corrupt_delta,
            "all_passed": True,
        },
        "execution": {
            "real_forward_hooks_executed": True,
            "normal_forward_count": 7,
            "future_augmented_forward_count": 3,
            "total_forward_count": 10,
            "parameters_frozen_for_control": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
            "model_eval_mode": model.training is False,
            "gradient_or_backward_executed": False,
            "hook_count_after_control": sum(
                len(getattr(layer, "_forward_hooks", {})) for layer in layers
            ),
        },
    }


def run_target_activation_patching_control(
    spec: CheckpointControlSpec,
    *,
    protocol: ActivationPatchingProtocol = QWEN2_5_0_5B_ACTIVATION_PATCHING_PROTOCOL,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Verify, load, and execute the reviewed Qwen activation-patching control."""

    try:
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "torch, transformers, huggingface_hub, and packaging are required"
        ) from error
    _validate_protocol(protocol)
    if (spec.model_id, spec.revision) != (protocol.model_id, protocol.revision):
        raise ValueError("checkpoint manifest model/revision does not match protocol")
    if (spec.device, spec.dtype, spec.attention_implementation) != (
        "cpu",
        "float32",
        "eager",
    ):
        raise ValueError("checkpoint manifest runtime does not match CPU FP32 eager protocol")
    snapshot_directory = download_checkpoint_snapshot(
        spec, local_files_only=local_files_only
    )
    snapshot = verify_checkpoint_snapshot(spec, snapshot_directory)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        snapshot.directory,
        trust_remote_code=False,
        local_files_only=True,
        use_fast=True,
    )
    dtype_argument = (
        {"dtype": torch.float32}
        if Version(transformers.__version__) >= Version("4.56")
        else {"torch_dtype": torch.float32}
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot.directory,
        trust_remote_code=False,
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        **dtype_argument,
    )
    execution = execute_loaded_activation_patching_control(
        protocol, model=model, tokenizer=tokenizer
    )
    projection: dict[str, object] = {
        "report_version": TARGET_ACTIVATION_PATCHING_REPORT_VERSION,
        "protocol": protocol.to_dict(),
        "protocol_fingerprint": protocol.fingerprint,
        "checkpoint_manifest_fingerprint": spec.manifest_fingerprint,
        "checked_at": spec.checked_at,
        "source": {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "source_base_url": spec.source_base_url,
            "loader_input": "verified_local_snapshot_directory",
            "all_selected_file_bytes_verified_before_load": True,
            "selected_files": [dict(item) for item in snapshot.files],
            "selected_total_bytes": sum(
                cast(int, item["size_bytes"]) for item in snapshot.files
            ),
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": "eager",
            "torch_num_threads": torch.get_num_threads(),
            "cuda_executed": False,
        },
        "model": {
            "class": type(model).__name__,
            "model_type": getattr(model.config, "model_type", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "decoder_layer_count": len(model.model.layers),
            "parameter_report": parameter_report(model),
            "parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in model.parameters()}
            ),
        },
        "result": execution,
        "scope": {
            "target_checkpoint_weights_loaded": True,
            "real_transformers_forward_hooks_executed": True,
            "fixed_behavior_contrast_observed": True,
            "constructive_controls_passed": True,
            "causal_structure_controls_passed": True,
            "external_timestamped_preregistration": False,
            "unique_natural_circuit_proven": False,
            "unbiased_effect_estimate_proven": False,
            "sae_experiment_executed": False,
            "model_quality_or_factual_reliability_proven": False,
            "cuda_gpu_or_vllm_executed": False,
            "performance_benchmark_performed": False,
            "publisher_authenticated_by_signature": False,
            "verification_to_loader_reopen_toctou_eliminated": False,
            "production_safety_proven": False,
        },
        "evidence_boundary": TARGET_ACTIVATION_PATCHING_EVIDENCE_BOUNDARY,
    }
    projection["report_fingerprint"] = "sha256:" + artifact_fingerprint(projection)
    return projection


def verify_recorded_activation_patching_report(
    path: Path,
    *,
    expected_checkpoint_manifest_fingerprint: str,
    expected_protocol_fingerprint: str = (
        QWEN2_5_0_5B_ACTIVATION_PATCHING_PROTOCOL.fingerprint
    ),
) -> Mapping[str, Any]:
    """Verify the canonical identity and fixed evidence boundaries of a report."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("recorded activation-patching report must be a JSON object")
    report = cast(dict[str, Any], raw)
    fingerprint = report.get("report_fingerprint")
    projection = {key: value for key, value in report.items() if key != "report_fingerprint"}
    expected = "sha256:" + artifact_fingerprint(projection)
    if fingerprint != expected:
        raise ValueError("recorded activation-patching report fingerprint mismatch")
    expected_top_level = {
        "checked_at",
        "checkpoint_manifest_fingerprint",
        "evidence_boundary",
        "model",
        "protocol",
        "protocol_fingerprint",
        "report_fingerprint",
        "report_version",
        "result",
        "runtime",
        "scope",
        "source",
    }
    if set(report) != expected_top_level:
        raise ValueError("recorded activation-patching top-level schema drifted")
    if report["report_version"] != TARGET_ACTIVATION_PATCHING_REPORT_VERSION:
        raise ValueError("recorded activation-patching report version mismatch")
    if report["protocol"] != QWEN2_5_0_5B_ACTIVATION_PATCHING_PROTOCOL.to_dict():
        raise ValueError("recorded activation-patching protocol body drifted")
    if report["protocol_fingerprint"] != expected_protocol_fingerprint:
        raise ValueError("recorded activation-patching protocol fingerprint mismatch")
    if report["checkpoint_manifest_fingerprint"] != (
        expected_checkpoint_manifest_fingerprint
    ):
        raise ValueError("recorded checkpoint manifest fingerprint mismatch")
    if report["checked_at"] != "2026-08-13":
        raise ValueError("recorded activation-patching checked_at drifted")
    if report["evidence_boundary"] != TARGET_ACTIVATION_PATCHING_EVIDENCE_BOUNDARY:
        raise ValueError("recorded activation-patching evidence boundary mismatch")

    expected_files = [
        {
            "filename": "config.json",
            "sha256": "sha256:18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
            "size_bytes": 659,
            "verified": True,
        },
        {
            "filename": "generation_config.json",
            "sha256": "sha256:e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
            "size_bytes": 242,
            "verified": True,
        },
        {
            "filename": "merges.txt",
            "sha256": "sha256:599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
            "size_bytes": 1_671_839,
            "verified": True,
        },
        {
            "filename": "model.safetensors",
            "sha256": "sha256:fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
            "size_bytes": 988_097_824,
            "verified": True,
        },
        {
            "filename": "tokenizer.json",
            "sha256": "sha256:c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
            "size_bytes": 7_031_645,
            "verified": True,
        },
        {
            "filename": "tokenizer_config.json",
            "sha256": "sha256:5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
            "size_bytes": 7_305,
            "verified": True,
        },
        {
            "filename": "vocab.json",
            "sha256": "sha256:ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
            "size_bytes": 2_776_833,
            "verified": True,
        },
    ]
    expected_source = {
        "all_selected_file_bytes_verified_before_load": True,
        "loader_input": "verified_local_snapshot_directory",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "selected_files": expected_files,
        "selected_total_bytes": 999_586_347,
        "source_base_url": (
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/"
            "7ae557604adf67be50417f59c2c2f167def9a775/"
        ),
    }
    if report["source"] != expected_source:
        raise ValueError("recorded activation-patching source evidence drifted")
    expected_runtime = {
        "attention_implementation": "eager",
        "cuda_executed": False,
        "device": "cpu",
        "dtype": "float32",
        "platform": "Windows-11-10.0.26200-SP0",
        "python_implementation": "CPython",
        "python_version": "3.12.10",
        "torch_num_threads": 8,
        "torch_version": "2.13.0+cpu",
        "transformers_version": "4.57.6",
    }
    if report["runtime"] != expected_runtime:
        raise ValueError("recorded activation-patching runtime drifted")
    expected_model = {
        "class": "Qwen2ForCausalLM",
        "decoder_layer_count": 24,
        "hidden_size": 896,
        "model_type": "qwen2",
        "parameter_dtypes": ["torch.float32"],
        "parameter_report": {
            "parameter_storage_bytes": 1_976_131_072,
            "total_parameters": 494_032_768,
            "trainable_fraction": 0.0,
            "trainable_parameters": 0,
        },
    }
    if report["model"] != expected_model:
        raise ValueError("recorded activation-patching model evidence drifted")

    protocol = QWEN2_5_0_5B_ACTIVATION_PATCHING_PROTOCOL
    clean_metric = 9.210310935974121
    corrupt_metric = -7.7003021240234375
    clean_logits_hash = (
        "sha256:7712a5226abbd04dc7a16b8e6e1147f85784d4a347c0b19595c3e9a0b53ef94a"
    )
    corrupt_logits_hash = (
        "sha256:885151b3b668ec4022cea46f39a997a2b244696a5ba9148e8cf545765c8cfe09"
    )
    source_layer_0_hash = (
        "sha256:ccdbcdf1c128b15f93c7ce9ab2d2068d28977b7d08bda09def4abcdf6c748442"
    )
    source_layer_11_hash = (
        "sha256:01c50947506138c10552a6a7767b3d9f9e86c83aa9f5fd825562b7002013da21"
    )
    future_logits_hash = (
        "sha256:6aeb0f070819656bf9828b30b44dbcb9fce183f72c93556fc61216b76adca45e"
    )
    expected_conditions = [
        {
            "clean_metric": clean_metric,
            "corrupt_metric": corrupt_metric,
            "layer_index": 0,
            "metric_position": 25,
            "name": "source_position_layer_0",
            "normalized_recovery": 1.0000241370674128,
            "patched_logits_sha256": source_layer_0_hash,
            "patched_metric": 9.210719108581543,
            "patched_positions": [19],
            "role": "preselected_source-site_intervention",
        },
        {
            "clean_metric": clean_metric,
            "corrupt_metric": corrupt_metric,
            "layer_index": 11,
            "metric_position": 25,
            "name": "source_position_layer_11",
            "normalized_recovery": 0.9922442752431005,
            "patched_logits_sha256": source_layer_11_hash,
            "patched_metric": 9.079156875610352,
            "patched_positions": [19],
            "role": "preselected_source-site_intervention",
        },
        {
            "clean_metric": clean_metric,
            "corrupt_metric": corrupt_metric,
            "layer_index": 23,
            "metric_position": 25,
            "name": "source_position_layer_23",
            "normalized_recovery": 0.0,
            "patched_logits_sha256": corrupt_logits_hash,
            "patched_metric": corrupt_metric,
            "patched_positions": [19],
            "role": "preselected_source-site_intervention",
        },
        {
            "clean_metric": clean_metric,
            "corrupt_metric": corrupt_metric,
            "layer_index": 0,
            "metric_position": 25,
            "name": "full_prefix_first_layer_positive_control",
            "normalized_recovery": 1.0,
            "patched_logits_sha256": clean_logits_hash,
            "patched_metric": clean_metric,
            "patched_positions": list(range(26)),
            "role": "constructive-positive-control",
        },
        {
            "clean_metric": clean_metric,
            "corrupt_metric": corrupt_metric,
            "layer_index": 23,
            "metric_position": 25,
            "name": "readout_position_final_layer_positive_control",
            "normalized_recovery": 1.0,
            "patched_logits_sha256": clean_logits_hash,
            "patched_metric": clean_metric,
            "patched_positions": [25],
            "role": "constructive-positive-control",
        },
        {
            "clean_metric": 9.210312843322754,
            "corrupt_metric": -7.700307846069336,
            "layer_index": 0,
            "metric_position": 25,
            "name": "future_position_first_layer_negative_control",
            "normalized_recovery": 0.0,
            "patched_logits_sha256": future_logits_hash,
            "patched_metric": -7.700307846069336,
            "patched_positions": [26],
            "role": "causal-future-negative-control",
        },
    ]
    expected_result = {
        "baseline": {
            "clean_metric": clean_metric,
            "clean_metric_logits_sha256": clean_logits_hash,
            "clean_minus_corrupt_gap": 16.91061305999756,
            "clean_top_token_id": 59_604,
            "clean_top_token_text": "Paris",
            "corrupt_metric": corrupt_metric,
            "corrupt_metric_logits_sha256": corrupt_logits_hash,
            "corrupt_top_token_id": 94_409,
            "corrupt_top_token_text": "Berlin",
        },
        "conditions": expected_conditions,
        "execution": {
            "future_augmented_forward_count": 3,
            "gradient_or_backward_executed": False,
            "hook_count_after_control": 0,
            "model_eval_mode": True,
            "normal_forward_count": 7,
            "parameters_frozen_for_control": True,
            "real_forward_hooks_executed": True,
            "total_forward_count": 10,
        },
        "input_contract": {
            "batch_size": 1,
            "clean_changed_token": " France",
            "clean_input_ids": list(protocol.clean_input_ids),
            "corrupt_changed_token": " Germany",
            "corrupt_input_ids": list(protocol.corrupt_input_ids),
            "future_control_position": 26,
            "metric_position": 25,
            "only_changed_position": 19,
            "prompt_token_count": 26,
        },
        "structural_controls": {
            "all_passed": True,
            "appended_future_clean_metric_delta": 1.9073486328125e-06,
            "appended_future_corrupt_metric_delta": -5.7220458984375e-06,
            "checks": {
                "appended_future_token_preserves_clean_past_metric": True,
                "appended_future_token_preserves_corrupt_past_metric": True,
                "final_layer_source_cannot_change_other_position": True,
                "full_prefix_first_layer_recovery_is_one": True,
                "future_position_cannot_change_past_metric": True,
                "readout_final_layer_recovery_is_one": True,
            },
            "control_tolerance": 1e-05,
        },
    }
    if report["result"] != expected_result:
        raise ValueError("recorded activation-patching result drifted")
    expected_scope = {
        "causal_structure_controls_passed": True,
        "constructive_controls_passed": True,
        "cuda_gpu_or_vllm_executed": False,
        "external_timestamped_preregistration": False,
        "fixed_behavior_contrast_observed": True,
        "model_quality_or_factual_reliability_proven": False,
        "performance_benchmark_performed": False,
        "production_safety_proven": False,
        "publisher_authenticated_by_signature": False,
        "real_transformers_forward_hooks_executed": True,
        "sae_experiment_executed": False,
        "target_checkpoint_weights_loaded": True,
        "unbiased_effect_estimate_proven": False,
        "unique_natural_circuit_proven": False,
        "verification_to_loader_reopen_toctou_eliminated": False,
    }
    if report["scope"] != expected_scope:
        raise ValueError("recorded activation-patching scope drifted")
    return report
