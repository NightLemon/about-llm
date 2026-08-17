"""Real-weight, selected-matrix quantization control for the reviewed Qwen snapshot.

The control deliberately quantizes one weight matrix rather than presenting a
partially quantized model as a deployable low-bit checkpoint.  It verifies the
reviewed snapshot, captures a real target-model activation, serializes and
reloads the selected weight through the repository's packed INT4 bundle, and
executes both the selected linear projection and a full model forward with the
dequantized replacement.  No fused low-bit kernel is used.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import platform
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.inference.quantization import quantize_symmetric_groupwise
from about_llm.inference.quantized_bundle import (
    QUANTIZED_BUNDLE_SCHEMA_VERSION,
    NamedQuantizedMatrix,
    QuantizedBundleIdentity,
    QuantizedMatrixBundle,
)
from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    download_checkpoint_snapshot,
    verify_checkpoint_snapshot,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

TARGET_WEIGHT_QUANTIZATION_REPORT_VERSION = (
    "about-llm.transformers-target-weight-quantization-report.v1"
)
TARGET_WEIGHT_QUANTIZATION_CHECKED_AT = "2026-08-15"
TARGET_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TARGET_MANIFEST_FINGERPRINT = (
    "sha256:ddf41f2cff963bc2a8fc186c28369abba8a920b850152fc815e2b17c7d037876"
)
TARGET_MODULE_NAME = "model.layers.0.self_attn.o_proj"
TARGET_PARAMETER_NAME = TARGET_MODULE_NAME + ".weight"
TARGET_WEIGHT_SHAPE = (896, 896)
TARGET_TOTAL_PARAMETERS = 494_032_768
TARGET_SELECTED_PARAMETERS = 802_816
TARGET_PROMPT_TOKENS = 31
TARGET_VOCABULARY_SIZE = 151_936
TARGET_BIT_WIDTH = 4
TARGET_GROUP_SIZE = 128
TARGET_SELECTED_FILE_COUNT = 7
TARGET_SELECTED_FILE_BYTES = 999_586_347

TARGET_WEIGHT_QUANTIZATION_EVIDENCE_BOUNDARY = (
    "This control verifies the reviewed immutable Qwen snapshot, loads it on CPU "
    "in FP32 with trust_remote_code disabled, captures one real activation, and "
    "quantizes only model.layers.0.self_attn.o_proj.weight with the repository's "
    "symmetric group-wise packed INT4 reference. It reloads that one-matrix "
    "artifact, executes the selected dequantized linear projection, and runs one "
    "full-model forward after replacing only that matrix with dequantized FP32 "
    "values. It does not produce or load a full low-bit checkpoint, execute a "
    "fused low-bit kernel, GPTQ, AWQ, SmoothQuant, calibration, generation, GPU, "
    "CUDA, vLLM, or a quality/performance evaluation. Selected-layer compression "
    "and two forward observations do not establish whole-model storage, runtime "
    "memory, latency, accuracy, effective context, licensing, provenance, or "
    "production safety."
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REPORT_FIELDS = {
    "report_version",
    "checked_at",
    "manifest_fingerprint",
    "source",
    "runtime",
    "selection",
    "artifact",
    "execution",
    "scope",
    "evidence_boundary",
    "report_fingerprint",
}
_SOURCE_FIELDS = {
    "model_id",
    "revision",
    "selected_file_count",
    "selected_file_bytes",
    "all_selected_file_bytes_verified_before_load",
    "loader_input",
}
_RUNTIME_FIELDS = {
    "python_implementation",
    "python_version",
    "platform",
    "torch_version",
    "transformers_version",
    "device",
    "source_dtype",
    "attention_implementation",
    "torch_num_threads",
    "cuda_executed",
}
_SELECTION_FIELDS = {
    "module_name",
    "parameter_name",
    "module_class",
    "bias_present",
    "weight_shape",
    "source_dtype",
    "bit_width",
    "group_size",
    "groups_per_row",
    "quantizer",
    "code_range",
    "selected_parameters",
    "total_model_parameters",
    "selected_parameter_fraction",
}
_ARTIFACT_FIELDS = {
    "bundle_schema_version",
    "tensor_count",
    "serialized_bundle_bytes",
    "serialized_bundle_sha256",
    "reference_fp32_weight_bytes",
    "ideal_packed_weight_bytes",
    "scale_metadata_bytes",
    "serialized_compression_ratio",
    "scale_dtype",
    "round_trip_codes_equal",
    "round_trip_scales_equal",
    "tamper_rejected_before_decode",
}
_EXECUTION_FIELDS = {
    "prompt_token_count",
    "captured_input_shape",
    "captured_output_shape",
    "last_logits_shape",
    "baseline_hook_vs_direct_max_abs_error",
    "weight_error",
    "selected_output_error",
    "last_logits_error",
    "baseline_last_logits_sha256",
    "partial_quantized_last_logits_sha256",
    "baseline_last_argmax_token_id",
    "partial_quantized_last_argmax_token_id",
    "last_argmax_match",
    "in_memory_vs_reloaded_selected_output_exact",
    "source_weight_restored_exactly",
}
_ERROR_FIELDS = {
    "mean_absolute_error",
    "root_mean_squared_error",
    "maximum_absolute_error",
    "relative_l2_error",
}
_SCOPE_FIELDS = {
    "target_checkpoint_weights_loaded",
    "real_target_activation_captured",
    "selected_target_weight_quantized_and_packed",
    "quantized_artifact_reloaded",
    "dequantized_selected_linear_executed",
    "partial_quantized_full_model_forward_executed",
    "full_checkpoint_quantized",
    "quantized_runtime_loaded",
    "fused_low_bit_kernel_executed",
    "gptq_awq_or_calibration_executed",
    "generation_executed",
    "gpu_cuda_or_vllm_executed",
    "whole_model_storage_or_runtime_memory_proven",
    "model_quality_or_effective_context_proven",
    "performance_benchmark_performed",
    "publisher_authenticated_by_signature",
    "license_compatibility_proven",
    "production_safety_proven",
}


def execute_selected_linear_quantization(
    model: Any,
    *,
    input_ids: Any,
    module_name: str,
    bundle_identity: QuantizedBundleIdentity,
    bit_width: int = TARGET_BIT_WIDTH,
    group_size: int = TARGET_GROUP_SIZE,
) -> dict[str, Any]:
    """Quantize one ``torch.nn.Linear`` weight and execute controlled forwards.

    The original weight, training mode, and per-parameter ``requires_grad`` flags
    are restored before return. The returned artifact data is path-free.
    """

    try:
        import numpy as np
        import torch
        from torch.nn import functional as torch_functional
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("NumPy and torch are required for quantization") from error

    if not isinstance(module_name, str) or not module_name:
        raise ValueError("module_name must be a non-empty string")
    modules = dict(model.named_modules())
    module = modules.get(module_name)
    if not isinstance(module, torch.nn.Linear):
        raise ValueError("selected module must be an existing torch.nn.Linear")
    if module.bias is not None:
        raise ValueError("selected v1 module must be bias-free")
    if module.weight.device.type != "cpu" or module.weight.dtype != torch.float32:
        raise ValueError("selected v1 weight must be CPU FP32")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long:
        raise TypeError("input_ids must be a torch.long Tensor")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence]")

    original_training = bool(model.training)
    original_requires_grad = [bool(parameter.requires_grad) for parameter in model.parameters()]
    reference_weight = module.weight.detach().clone()
    captured_inputs: list[Any] = []
    captured_outputs: list[Any] = []

    def capture(_module: Any, arguments: tuple[Any, ...], output: Any) -> None:
        if len(arguments) != 1 or not isinstance(arguments[0], torch.Tensor):
            raise RuntimeError("selected linear hook received unexpected arguments")
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("selected linear hook received non-tensor output")
        captured_inputs.append(arguments[0].detach().clone())
        captured_outputs.append(output.detach().clone())

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handle = module.register_forward_hook(capture)
    try:
        with torch.no_grad():
            baseline_result = model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    if len(captured_inputs) != 1 or len(captured_outputs) != 1:
        raise RuntimeError("selected linear module must execute exactly once")
    baseline_logits = getattr(baseline_result, "logits", None)
    if not isinstance(baseline_logits, torch.Tensor) or baseline_logits.ndim != 3:
        raise RuntimeError("model forward did not return rank-3 logits")
    captured_input = captured_inputs[0]
    captured_output = captured_outputs[0]
    direct_output = torch_functional.linear(captured_input, reference_weight)
    hook_direct_error = float((captured_output - direct_output).abs().max().item())
    if hook_direct_error > 1e-6:
        raise AssertionError("captured selected-linear output disagrees with direct FP32")

    reference_numpy = reference_weight.numpy().copy()
    quantized = quantize_symmetric_groupwise(
        reference_numpy,
        bit_width=bit_width,
        group_size=group_size,
    )
    packed = quantized.pack()
    bundle = QuantizedMatrixBundle(
        identity=bundle_identity,
        tensors=(NamedQuantizedMatrix(module_name + ".weight", packed),),
    )
    artifact = bundle.to_bytes()
    restored_bundle = QuantizedMatrixBundle.from_bytes(artifact)
    restored = restored_bundle.get(module_name + ".weight")
    round_trip_codes_equal = bool(
        np.array_equal(restored.unpack().values, quantized.values)
    )
    round_trip_scales_equal = bool(np.array_equal(restored.scales, quantized.scales))
    if not round_trip_codes_equal or not round_trip_scales_equal:
        raise AssertionError("quantized artifact round trip changed codes or scales")
    tampered = bytearray(artifact)
    tampered[-1] ^= 1
    try:
        QuantizedMatrixBundle.from_bytes(bytes(tampered))
    except ValueError:
        tamper_rejected = True
    else:  # pragma: no cover - defensive assertion
        tamper_rejected = False
    if not tamper_rejected:
        raise AssertionError("tampered quantized artifact was accepted")

    in_memory_weight = torch.from_numpy(packed.dequantize())
    restored_weight = torch.from_numpy(restored.dequantize())
    in_memory_output = torch_functional.linear(captured_input, in_memory_weight)
    restored_output = torch_functional.linear(captured_input, restored_weight)
    output_round_trip_exact = bool(torch.equal(in_memory_output, restored_output))
    if not output_round_trip_exact:
        raise AssertionError("reloaded quantized selected output drifted")

    partial_result: Any = None
    restored_source_exactly = False
    try:
        with torch.no_grad():
            module.weight.copy_(restored_weight)
            partial_result = model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
    finally:
        with torch.no_grad():
            module.weight.copy_(reference_weight)
        restored_source_exactly = bool(torch.equal(module.weight, reference_weight))
        for parameter, requires_grad in zip(
            model.parameters(), original_requires_grad, strict=True
        ):
            parameter.requires_grad_(requires_grad)
        model.train(original_training)
    if not restored_source_exactly:
        raise AssertionError("source weight was not restored after quantized forward")
    partial_logits = getattr(partial_result, "logits", None)
    if (
        not isinstance(partial_logits, torch.Tensor)
        or partial_logits.shape != baseline_logits.shape
    ):
        raise RuntimeError("partial-quantized forward returned inconsistent logits")

    baseline_last_logits = baseline_logits[0, -1].detach().clone()
    partial_last_logits = partial_logits[0, -1].detach().clone()
    baseline_argmax = int(torch.argmax(baseline_last_logits).item())
    partial_argmax = int(torch.argmax(partial_last_logits).item())
    reference_bytes = int(reference_weight.numel() * reference_weight.element_size())
    return {
        "module_class": type(module).__name__,
        "bias_present": module.bias is not None,
        "weight_shape": list(reference_weight.shape),
        "source_dtype": str(reference_weight.dtype),
        "groups_per_row": quantized.groups_per_row,
        "selected_parameters": int(reference_weight.numel()),
        "total_model_parameters": sum(
            int(parameter.numel()) for parameter in model.parameters()
        ),
        "artifact": {
            "serialized_bundle_bytes": len(artifact),
            "serialized_bundle_sha256": _bytes_sha256(artifact),
            "reference_fp32_weight_bytes": reference_bytes,
            "ideal_packed_weight_bytes": packed.packed_weight_bytes,
            "scale_metadata_bytes": packed.scale_metadata_bytes,
            "serialized_compression_ratio": reference_bytes / len(artifact),
            "round_trip_codes_equal": round_trip_codes_equal,
            "round_trip_scales_equal": round_trip_scales_equal,
            "tamper_rejected_before_decode": tamper_rejected,
        },
        "execution": {
            "captured_input_shape": list(captured_input.shape),
            "captured_output_shape": list(captured_output.shape),
            "last_logits_shape": list(baseline_logits.shape),
            "baseline_hook_vs_direct_max_abs_error": hook_direct_error,
            "weight_error": _torch_error(reference_weight, restored_weight),
            "selected_output_error": _torch_error(captured_output, restored_output),
            "last_logits_error": _torch_error(
                baseline_last_logits, partial_last_logits
            ),
            "baseline_last_logits_sha256": _tensor_sha256(baseline_last_logits),
            "partial_quantized_last_logits_sha256": _tensor_sha256(
                partial_last_logits
            ),
            "baseline_last_argmax_token_id": baseline_argmax,
            "partial_quantized_last_argmax_token_id": partial_argmax,
            "last_argmax_match": baseline_argmax == partial_argmax,
            "in_memory_vs_reloaded_selected_output_exact": output_round_trip_exact,
            "source_weight_restored_exactly": restored_source_exactly,
        },
    }


def run_target_weight_quantization_control(
    spec: CheckpointControlSpec,
    *,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Verify, load, partially quantize, and execute the reviewed Qwen control."""

    _require_target_spec(spec)
    try:
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("torch and transformers are required for execution") from error

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
        attn_implementation=spec.attention_implementation,
        **dtype_argument,
    )
    if type(model).__name__ != spec.expected_model_class:
        raise ValueError("loaded model class mismatch")
    if getattr(model.config, "model_type", None) != spec.expected_model_type:
        raise ValueError("loaded model type mismatch")
    architecture = _target_architecture(model.config)
    messages = [
        {"role": message.role, "content": message.content} for message in spec.messages
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    observation = execute_selected_linear_quantization(
        model,
        input_ids=input_ids,
        module_name=TARGET_MODULE_NAME,
        bundle_identity=QuantizedBundleIdentity(
            model_family=spec.model_id,
            model_revision=spec.revision,
            tokenizer_id=spec.model_id,
            tokenizer_revision=spec.revision,
            architecture_config=architecture,
        ),
    )
    if observation["weight_shape"] != list(TARGET_WEIGHT_SHAPE):
        raise AssertionError("selected target weight shape drifted")
    if observation["selected_parameters"] != TARGET_SELECTED_PARAMETERS:
        raise AssertionError("selected target parameter count drifted")
    if observation["total_model_parameters"] != TARGET_TOTAL_PARAMETERS:
        raise AssertionError("target total parameter count drifted")
    execution_observation = cast(dict[str, Any], observation["execution"])
    if execution_observation["captured_input_shape"] != [1, TARGET_PROMPT_TOKENS, 896]:
        raise AssertionError("target prompt or activation shape drifted")
    if execution_observation["last_logits_shape"] != [
        1,
        TARGET_PROMPT_TOKENS,
        TARGET_VOCABULARY_SIZE,
    ]:
        raise AssertionError("target logits shape drifted")

    artifact_observation = cast(dict[str, Any], observation["artifact"])
    projection: dict[str, object] = {
        "report_version": TARGET_WEIGHT_QUANTIZATION_REPORT_VERSION,
        "checked_at": TARGET_WEIGHT_QUANTIZATION_CHECKED_AT,
        "manifest_fingerprint": spec.manifest_fingerprint,
        "source": {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "selected_file_count": len(snapshot.files),
            "selected_file_bytes": sum(
                cast(int, item["size_bytes"]) for item in snapshot.files
            ),
            "all_selected_file_bytes_verified_before_load": True,
            "loader_input": "verified_local_snapshot_directory",
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": "cpu",
            "source_dtype": "torch.float32",
            "attention_implementation": spec.attention_implementation,
            "torch_num_threads": torch.get_num_threads(),
            "cuda_executed": False,
        },
        "selection": {
            "module_name": TARGET_MODULE_NAME,
            "parameter_name": TARGET_PARAMETER_NAME,
            "module_class": observation["module_class"],
            "bias_present": observation["bias_present"],
            "weight_shape": observation["weight_shape"],
            "source_dtype": observation["source_dtype"],
            "bit_width": TARGET_BIT_WIDTH,
            "group_size": TARGET_GROUP_SIZE,
            "groups_per_row": observation["groups_per_row"],
            "quantizer": "symmetric absmax per contiguous row group",
            "code_range": [-7, 7],
            "selected_parameters": observation["selected_parameters"],
            "total_model_parameters": observation["total_model_parameters"],
            "selected_parameter_fraction": (
                TARGET_SELECTED_PARAMETERS / TARGET_TOTAL_PARAMETERS
            ),
        },
        "artifact": {
            "bundle_schema_version": QUANTIZED_BUNDLE_SCHEMA_VERSION,
            "tensor_count": 1,
            **artifact_observation,
            "scale_dtype": "float32",
        },
        "execution": {
            "prompt_token_count": int(input_ids.shape[1]),
            **execution_observation,
        },
        "scope": {
            "target_checkpoint_weights_loaded": True,
            "real_target_activation_captured": True,
            "selected_target_weight_quantized_and_packed": True,
            "quantized_artifact_reloaded": True,
            "dequantized_selected_linear_executed": True,
            "partial_quantized_full_model_forward_executed": True,
            "full_checkpoint_quantized": False,
            "quantized_runtime_loaded": False,
            "fused_low_bit_kernel_executed": False,
            "gptq_awq_or_calibration_executed": False,
            "generation_executed": False,
            "gpu_cuda_or_vllm_executed": False,
            "whole_model_storage_or_runtime_memory_proven": False,
            "model_quality_or_effective_context_proven": False,
            "performance_benchmark_performed": False,
            "publisher_authenticated_by_signature": False,
            "license_compatibility_proven": False,
            "production_safety_proven": False,
        },
        "evidence_boundary": TARGET_WEIGHT_QUANTIZATION_EVIDENCE_BOUNDARY,
    }
    projection["report_fingerprint"] = _object_sha256(projection)
    return projection


def verify_recorded_target_weight_quantization_report(
    path: Path,
    *,
    expected_manifest_fingerprint: str = TARGET_MANIFEST_FINGERPRINT,
) -> Mapping[str, Any]:
    """Validate the closed report, arithmetic, scope, and self-fingerprint."""

    report = _load_json_file(path)
    _require_fields(report, _REPORT_FIELDS, "report")
    if report.get("report_version") != TARGET_WEIGHT_QUANTIZATION_REPORT_VERSION:
        raise ValueError("report.report_version is unsupported")
    if report.get("checked_at") != TARGET_WEIGHT_QUANTIZATION_CHECKED_AT:
        raise ValueError("report.checked_at drift")
    if report.get("evidence_boundary") != TARGET_WEIGHT_QUANTIZATION_EVIDENCE_BOUNDARY:
        raise ValueError("report.evidence_boundary drift")
    manifest_fingerprint = _sha256(
        report.get("manifest_fingerprint"), "report.manifest_fingerprint"
    )
    if not hmac.compare_digest(manifest_fingerprint, expected_manifest_fingerprint):
        raise ValueError("report.manifest_fingerprint mismatch")
    fingerprint = _sha256(report.get("report_fingerprint"), "report")
    projection = dict(report)
    del projection["report_fingerprint"]
    if not hmac.compare_digest(fingerprint, _object_sha256(projection)):
        raise ValueError("report fingerprint mismatch")

    source = _record(report.get("source"), _SOURCE_FIELDS, "report.source")
    if source != {
        "model_id": TARGET_MODEL_ID,
        "revision": TARGET_REVISION,
        "selected_file_count": TARGET_SELECTED_FILE_COUNT,
        "selected_file_bytes": TARGET_SELECTED_FILE_BYTES,
        "all_selected_file_bytes_verified_before_load": True,
        "loader_input": "verified_local_snapshot_directory",
    }:
        raise ValueError("report.source does not match the reviewed snapshot")

    runtime = _record(report.get("runtime"), _RUNTIME_FIELDS, "report.runtime")
    for name in (
        "python_implementation",
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
    ):
        _nonempty_string(runtime.get(name), f"report.runtime.{name}")
    if (
        runtime.get("device") != "cpu"
        or runtime.get("source_dtype") != "torch.float32"
        or runtime.get("attention_implementation") != "eager"
        or runtime.get("cuda_executed") is not False
    ):
        raise ValueError("report.runtime is not the reviewed CPU FP32 eager path")
    _positive_integer(runtime.get("torch_num_threads"), "report.runtime.torch_num_threads")

    selection = _record(
        report.get("selection"), _SELECTION_FIELDS, "report.selection"
    )
    expected_selection = {
        "module_name": TARGET_MODULE_NAME,
        "parameter_name": TARGET_PARAMETER_NAME,
        "module_class": "Linear",
        "bias_present": False,
        "weight_shape": list(TARGET_WEIGHT_SHAPE),
        "source_dtype": "torch.float32",
        "bit_width": TARGET_BIT_WIDTH,
        "group_size": TARGET_GROUP_SIZE,
        "groups_per_row": 7,
        "quantizer": "symmetric absmax per contiguous row group",
        "code_range": [-7, 7],
        "selected_parameters": TARGET_SELECTED_PARAMETERS,
        "total_model_parameters": TARGET_TOTAL_PARAMETERS,
        "selected_parameter_fraction": TARGET_SELECTED_PARAMETERS
        / TARGET_TOTAL_PARAMETERS,
    }
    if selection != expected_selection:
        raise ValueError("report.selection does not match the reviewed target matrix")

    artifact = _record(report.get("artifact"), _ARTIFACT_FIELDS, "report.artifact")
    if (
        artifact.get("bundle_schema_version") != QUANTIZED_BUNDLE_SCHEMA_VERSION
        or artifact.get("tensor_count") != 1
        or artifact.get("reference_fp32_weight_bytes") != 3_211_264
        or artifact.get("ideal_packed_weight_bytes") != 401_408
        or artifact.get("scale_metadata_bytes") != 25_088
        or artifact.get("scale_dtype") != "float32"
        or artifact.get("round_trip_codes_equal") is not True
        or artifact.get("round_trip_scales_equal") is not True
        or artifact.get("tamper_rejected_before_decode") is not True
    ):
        raise ValueError("report.artifact contract drift")
    serialized_bytes = _positive_integer(
        artifact.get("serialized_bundle_bytes"),
        "report.artifact.serialized_bundle_bytes",
    )
    if serialized_bytes <= 401_408 + 25_088:
        raise ValueError("report.artifact serialized bundle omitted framing metadata")
    _sha256(artifact.get("serialized_bundle_sha256"), "report.artifact")
    compression = _finite_number(
        artifact.get("serialized_compression_ratio"),
        "report.artifact.serialized_compression_ratio",
    )
    if not math.isclose(compression, 3_211_264 / serialized_bytes, rel_tol=1e-15):
        raise ValueError("report.artifact compression ratio is inconsistent")

    execution = _record(
        report.get("execution"), _EXECUTION_FIELDS, "report.execution"
    )
    if (
        execution.get("prompt_token_count") != TARGET_PROMPT_TOKENS
        or execution.get("captured_input_shape") != [1, TARGET_PROMPT_TOKENS, 896]
        or execution.get("captured_output_shape") != [1, TARGET_PROMPT_TOKENS, 896]
        or execution.get("last_logits_shape")
        != [1, TARGET_PROMPT_TOKENS, TARGET_VOCABULARY_SIZE]
        or execution.get("in_memory_vs_reloaded_selected_output_exact") is not True
        or execution.get("source_weight_restored_exactly") is not True
    ):
        raise ValueError("report.execution shape or restoration contract drift")
    hook_error = _finite_number(
        execution.get("baseline_hook_vs_direct_max_abs_error"),
        "report.execution.baseline_hook_vs_direct_max_abs_error",
    )
    if not 0 <= hook_error <= 1e-6:
        raise ValueError("report.execution baseline direct check failed")
    for name in ("weight_error", "selected_output_error", "last_logits_error"):
        metrics = _record(execution.get(name), _ERROR_FIELDS, f"report.execution.{name}")
        for metric_name, value in metrics.items():
            metric = _finite_number(value, f"report.execution.{name}.{metric_name}")
            if metric < 0:
                raise ValueError("report error metrics must be non-negative")
    _sha256(execution.get("baseline_last_logits_sha256"), "report.execution")
    _sha256(
        execution.get("partial_quantized_last_logits_sha256"), "report.execution"
    )
    baseline_argmax = _nonnegative_integer(
        execution.get("baseline_last_argmax_token_id"),
        "report.execution.baseline_last_argmax_token_id",
    )
    partial_argmax = _nonnegative_integer(
        execution.get("partial_quantized_last_argmax_token_id"),
        "report.execution.partial_quantized_last_argmax_token_id",
    )
    if baseline_argmax >= TARGET_VOCABULARY_SIZE or partial_argmax >= TARGET_VOCABULARY_SIZE:
        raise ValueError("report.execution argmax token is out of vocabulary")
    if execution.get("last_argmax_match") is not (baseline_argmax == partial_argmax):
        raise ValueError("report.execution last_argmax_match is inconsistent")

    scope = _record(report.get("scope"), _SCOPE_FIELDS, "report.scope")
    expected_true = {
        "target_checkpoint_weights_loaded",
        "real_target_activation_captured",
        "selected_target_weight_quantized_and_packed",
        "quantized_artifact_reloaded",
        "dequantized_selected_linear_executed",
        "partial_quantized_full_model_forward_executed",
    }
    for name, value in scope.items():
        if value is not (name in expected_true):
            raise ValueError(f"report.scope.{name} drift")
    return report


def _target_architecture(config: Any) -> dict[str, Any]:
    expected = {
        "model_type": "qwen2",
        "hidden_size": 896,
        "intermediate_size": 4_864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": TARGET_VOCABULARY_SIZE,
        "selected_module": TARGET_MODULE_NAME,
    }
    observed = {
        name: getattr(config, name)
        for name in expected
        if name != "selected_module"
    }
    observed["selected_module"] = TARGET_MODULE_NAME
    if observed != expected:
        raise ValueError("loaded target architecture drift")
    return expected


def _require_target_spec(spec: CheckpointControlSpec) -> None:
    if not isinstance(spec, CheckpointControlSpec):
        raise TypeError("spec must be CheckpointControlSpec")
    if (
        spec.model_id != TARGET_MODEL_ID
        or spec.revision != TARGET_REVISION
        or spec.manifest_fingerprint != TARGET_MANIFEST_FINGERPRINT
        or spec.expected_model_class != "Qwen2ForCausalLM"
        or spec.expected_model_type != "qwen2"
        or spec.dtype != "float32"
        or spec.device != "cpu"
        or spec.attention_implementation != "eager"
    ):
        raise ValueError("spec is not the reviewed Qwen target control")


def _torch_error(reference: Any, observed: Any) -> dict[str, float]:
    import torch

    if not isinstance(reference, torch.Tensor) or not isinstance(observed, torch.Tensor):
        raise TypeError("error inputs must be torch tensors")
    if reference.shape != observed.shape or reference.numel() == 0:
        raise ValueError("error inputs must have the same non-empty shape")
    expected = reference.detach().to(dtype=torch.float64)
    actual = observed.detach().to(dtype=torch.float64)
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise ValueError("error inputs must be finite")
    difference = actual - expected
    absolute = difference.abs()
    denominator = torch.linalg.vector_norm(expected.reshape(-1))
    relative = torch.linalg.vector_norm(difference.reshape(-1)) / denominator
    return {
        "mean_absolute_error": float(absolute.mean().item()),
        "root_mean_squared_error": float(difference.square().mean().sqrt().item()),
        "maximum_absolute_error": float(absolute.max().item()),
        "relative_l2_error": float(relative.item()),
    }


def _tensor_sha256(tensor: Any) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("fingerprinted value must be a torch Tensor")
    value = tensor.detach().cpu().contiguous().numpy()
    header = canonical_json_bytes(
        {"dtype": str(value.dtype), "shape": list(value.shape)}
    )
    return _bytes_sha256(header + b"\n" + value.tobytes(order="C"))


def _bytes_sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _object_sha256(value: Any) -> str:
    return "sha256:" + artifact_fingerprint(value)


def _load_json_file(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    with path.open("rb") as stream:
        value = stream.read(1_000_001)
    if len(value) > 1_000_000:
        raise ValueError("report exceeds byte limit")
    try:
        decoded = value.decode("utf-8")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("report must be strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("report must be a JSON object")
    return cast(dict[str, Any], parsed)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _require_fields(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name}: field set mismatch")


def _record(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    _require_fields(value, fields, name)
    return cast(dict[str, Any], value)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 fingerprint")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return cast(int, value)


def _positive_integer(value: Any, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
