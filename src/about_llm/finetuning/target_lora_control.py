"""Target-checkpoint LoRA training control with narrow, verifiable claims."""

from __future__ import annotations

import gc
import hashlib
import hmac
import json
import math
import os
import platform
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    download_checkpoint_snapshot,
    verify_checkpoint_snapshot,
    verify_recorded_checkpoint_report,
)
from about_llm.llmops import canonical_json_bytes

TARGET_LORA_CONTROL_VERSION = "about-llm.target-lora-control.v1"
TARGET_LORA_REPORT_VERSION = "about-llm.target-lora-control-report.v1"
TARGET_LORA_ARTIFACT_VERSION = "about-llm.target-lora-adapter-artifact.v1"
TARGET_LORA_ARTIFACT_MANIFEST = "about-llm-target-lora-manifest.json"
TARGET_LORA_EVIDENCE_BOUNDARY = (
    "This control rehashes the selected immutable-revision checkpoint files before each "
    "load, loads the reviewed Qwen checkpoint with trust_remote_code disabled, executes "
    "one CPU FP32 assistant-only LoRA backward and AdamW step, verifies a frozen-base "
    "parameter fingerprint, saves a standard PEFT adapter, and reloads that adapter into "
    "a freshly loaded copy of the same selected checkpoint. It does not authenticate the "
    "publisher or trainer, eliminate verification-to-loader-reopen TOCTOU, prove every "
    "repository file or license, export optimizer/RNG/resume state, merge full weights, "
    "execute quantization/QLoRA/CUDA/vLLM, establish convergence or model quality, use a "
    "representative dataset, benchmark memory/performance, or prove production safety."
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_TENSOR_KEY = re.compile(
    r"base_model\.model\.model\.layers\.(?P<layer>[0-9]+)\.self_attn\."
    r"(?P<module>q_proj|v_proj)\.lora_(?P<side>A|B)\.weight\Z"
)
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_REPORT_BYTES = 4_000_000
_MAX_MESSAGE_CHARACTERS = 4096
_ADAPTER_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
}
_CONTROL_FIELDS = {
    "adapter",
    "checked_at",
    "control_version",
    "evidence_boundary",
    "max_sequence_tokens",
    "messages",
    "model_contract",
    "model_id",
    "optimizer",
    "revision",
    "runtime",
    "source_checkpoint_manifest_fingerprint",
    "source_checkpoint_report_fingerprint",
    "torch_seed",
}
_MODEL_CONTRACT_FIELDS = {
    "base_parameter_count",
    "expected_model_class",
    "expected_model_type",
    "head_dim",
    "hidden_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
}
_RUNTIME_FIELDS = {"attention_implementation", "device", "dtype", "torch_num_threads"}
_ADAPTER_FIELDS = {
    "alpha",
    "bias",
    "dropout",
    "rank",
    "target_modules",
    "task_type",
}
_OPTIMIZER_FIELDS = {
    "beta1",
    "beta2",
    "epsilon",
    "learning_rate",
    "name",
    "steps",
    "weight_decay",
}
_MESSAGE_FIELDS = {"content", "role"}
_ARTIFACT_FIELDS = {
    "contract",
    "control_manifest_fingerprint",
    "files",
    "identity",
    "manifest_fingerprint",
    "schema_version",
    "source_checkpoint_manifest_fingerprint",
    "tensors",
}
_ARTIFACT_IDENTITY_FIELDS = {"adapter_name", "architecture", "model_id", "revision"}
_ARTIFACT_CONTRACT_FIELDS = {
    "adapter_type",
    "alpha",
    "base_reference_kind",
    "bias",
    "dropout",
    "full_weight_merge_performed",
    "quantized_base",
    "rank",
    "target_modules",
    "task_type",
}
_ARTIFACT_FILE_FIELDS = {"bytes", "path", "sha256"}
_ARTIFACT_TENSOR_FIELDS = {
    "a_tensor_count",
    "b_tensor_count",
    "descriptor_set_sha256",
    "dtypes",
    "nonzero_a_tensor_count",
    "nonzero_b_element_count",
    "nonzero_b_tensor_count",
    "tensor_count",
    "total_numel",
}
_ADAPTER_CONFIG_FIELDS = {
    "alora_invocation_tokens",
    "alpha_pattern",
    "arrow_config",
    "auto_mapping",
    "base_model_name_or_path",
    "bias",
    "corda_config",
    "ensure_weight_tying",
    "eva_config",
    "exclude_modules",
    "fan_in_fan_out",
    "inference_mode",
    "init_lora_weights",
    "layer_replication",
    "layers_pattern",
    "layers_to_transform",
    "loftq_config",
    "lora_alpha",
    "lora_bias",
    "lora_dropout",
    "lora_ga_config",
    "megatron_config",
    "megatron_core",
    "modules_to_save",
    "monteclora_config",
    "peft_type",
    "peft_version",
    "qalora_group_size",
    "r",
    "rank_pattern",
    "revision",
    "target_modules",
    "target_parameters",
    "task_type",
    "trainable_token_indices",
    "use_bdlora",
    "use_dora",
    "use_qalora",
    "use_rslora",
    "velora_config",
}
_REPORT_FIELDS = {
    "adapter_artifact",
    "checked_at",
    "control_manifest_fingerprint",
    "evidence_boundary",
    "execution",
    "model",
    "report_fingerprint",
    "report_version",
    "round_trip",
    "runtime",
    "sample",
    "scope",
    "source",
}
_REPORT_SOURCE_FIELDS = {
    "all_selected_file_bytes_verified_before_initial_load",
    "all_selected_file_bytes_verified_before_reload",
    "checkpoint_report_fingerprint",
    "model_id",
    "revision",
    "selected_file_count",
    "selected_total_bytes",
}
_REPORT_RUNTIME_FIELDS = {
    "attention_implementation",
    "cuda_executed",
    "device",
    "dtype",
    "peft_version",
    "platform",
    "python_implementation",
    "python_version",
    "torch_num_threads",
    "torch_version",
    "transformers_version",
}
_REPORT_SAMPLE_FIELDS = {
    "assistant_only_loss_mask_applied",
    "chat_template_fingerprint",
    "input_ids_fingerprint",
    "input_token_ids",
    "label_ids_fingerprint",
    "max_sequence_tokens",
    "prompt_token_count",
    "supervised_token_count",
}
_REPORT_MODEL_FIELDS = {
    "adapter_parameter_count",
    "base_parameter_count",
    "class",
    "head_dim",
    "hidden_size",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "parameter_dtypes",
    "total_parameter_count_with_adapter",
    "trainable_fraction_with_adapter",
}
_REPORT_EXECUTION_FIELDS = {
    "adapter_a_tensor_count",
    "adapter_b_tensor_count",
    "adapter_nonzero_a_tensor_count_after_step",
    "adapter_nonzero_b_element_count_after_step",
    "adapter_nonzero_b_tensor_count_after_step",
    "backward_executed",
    "base_last_logits_sha256",
    "finite_gradient_tensor_count",
    "frozen_base_gradient_tensor_count",
    "frozen_base_parameter_fingerprint_after",
    "frozen_base_parameter_fingerprint_before",
    "frozen_base_parameters_unchanged",
    "initial_adapter_base_max_abs_error",
    "initial_adapter_last_logits_sha256",
    "initial_loss",
    "optimizer",
    "optimizer_step_count",
    "post_step_loss",
    "post_step_vs_base_max_abs_error",
    "trained_last_logits_sha256",
    "trainable_gradient_tensor_count",
}
_REPORT_OPTIMIZER_FIELDS = {
    "beta1",
    "beta2",
    "epsilon",
    "learning_rate",
    "name",
    "weight_decay",
}
_REPORT_ROUND_TRIP_FIELDS = {
    "adapter_loaded_with_peft",
    "maximum_logit_error",
    "reloaded_last_logits_sha256",
    "reloaded_loss",
    "trained_and_reloaded_logits_exact",
}
_REPORT_SCOPE_FIELDS = {
    "assistant_only_training_boundary_observed",
    "cuda_executed",
    "full_weight_merge_executed",
    "license_compatibility_proven",
    "model_quality_or_convergence_proven",
    "optimizer_scheduler_rng_resume_state_exported",
    "peft_adapter_saved_and_reloaded",
    "performance_or_peak_memory_benchmarked",
    "production_safety_proven",
    "publisher_or_trainer_authenticated",
    "qlora_or_quantized_base_executed",
    "representative_dataset_used",
    "target_checkpoint_backward_executed",
    "target_checkpoint_weights_loaded",
    "trust_remote_code",
    "verification_to_loader_reopen_toctou_eliminated",
    "vllm_or_serving_runtime_executed",
}


@dataclass(frozen=True)
class TargetLoRAModelContract:
    expected_model_class: str
    expected_model_type: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    base_parameter_count: int

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class TargetLoRAAdapterContract:
    task_type: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: int
    dropout: float
    bias: str

    def to_dict(self) -> dict[str, object]:
        value = cast(dict[str, object], asdict(self))
        value["target_modules"] = list(self.target_modules)
        return value


@dataclass(frozen=True)
class TargetLoRAOptimizerContract:
    name: str
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    steps: int

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class TargetLoRAMessage:
    role: str
    content: str


@dataclass(frozen=True)
class TargetLoRAControlSpec:
    checked_at: str
    model_id: str
    revision: str
    source_checkpoint_manifest_fingerprint: str
    source_checkpoint_report_fingerprint: str
    model_contract: TargetLoRAModelContract
    device: str
    dtype: str
    attention_implementation: str
    torch_num_threads: int
    adapter: TargetLoRAAdapterContract
    optimizer: TargetLoRAOptimizerContract
    torch_seed: int
    max_sequence_tokens: int
    messages: tuple[TargetLoRAMessage, ...]
    manifest_fingerprint: str


@dataclass(frozen=True)
class TargetLoRAArtifactVerification:
    manifest_fingerprint: str
    manifest_bytes: int
    file_count: int
    total_file_bytes: int
    files: tuple[Mapping[str, object], ...]
    tensor_count: int
    total_tensor_numel: int
    tensor_dtypes: tuple[str, ...]
    descriptor_set_sha256: str
    a_tensor_count: int
    b_tensor_count: int
    nonzero_a_tensor_count: int
    nonzero_b_tensor_count: int
    nonzero_b_element_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_fingerprint": self.manifest_fingerprint,
            "manifest_bytes": self.manifest_bytes,
            "file_count": self.file_count,
            "total_file_bytes": self.total_file_bytes,
            "files": [dict(item) for item in self.files],
            "tensor_count": self.tensor_count,
            "total_tensor_numel": self.total_tensor_numel,
            "tensor_dtypes": list(self.tensor_dtypes),
            "descriptor_set_sha256": self.descriptor_set_sha256,
            "a_tensor_count": self.a_tensor_count,
            "b_tensor_count": self.b_tensor_count,
            "nonzero_a_tensor_count": self.nonzero_a_tensor_count,
            "nonzero_b_tensor_count": self.nonzero_b_tensor_count,
            "nonzero_b_element_count": self.nonzero_b_element_count,
        }


@dataclass(frozen=True)
class TargetLoRATrainingResult:
    sample: Mapping[str, object]
    model: Mapping[str, object]
    execution: Mapping[str, object]
    artifact: TargetLoRAArtifactVerification
    trained_last_logits: Any


def load_target_lora_control_spec(
    path: Path, *, checkpoint_spec: CheckpointControlSpec
) -> TargetLoRAControlSpec:
    """Load a strict control manifest and bind it to a reviewed checkpoint spec."""

    manifest = _load_strict_json_file(path, max_bytes=_MAX_MANIFEST_BYTES)
    _exact_fields(manifest, _CONTROL_FIELDS, "target LoRA manifest")
    if manifest.get("control_version") != TARGET_LORA_CONTROL_VERSION:
        raise ValueError("target LoRA manifest control_version is unsupported")
    if manifest.get("evidence_boundary") != TARGET_LORA_EVIDENCE_BOUNDARY:
        raise ValueError("target LoRA manifest evidence_boundary drift")
    checked_at = _iso_date(manifest.get("checked_at"), "manifest.checked_at")
    model_id = _string(manifest.get("model_id"), "manifest.model_id")
    revision = _string(manifest.get("revision"), "manifest.revision")
    if _MODEL_ID.fullmatch(model_id) is None or _REVISION.fullmatch(revision) is None:
        raise ValueError("target LoRA manifest model identity is invalid")
    if (model_id, revision) != (checkpoint_spec.model_id, checkpoint_spec.revision):
        raise ValueError("target LoRA manifest does not match checkpoint identity")
    checkpoint_manifest_fingerprint = _digest(
        manifest.get("source_checkpoint_manifest_fingerprint"),
        "manifest.source_checkpoint_manifest_fingerprint",
    )
    if checkpoint_manifest_fingerprint != checkpoint_spec.manifest_fingerprint:
        raise ValueError("target LoRA source checkpoint manifest fingerprint mismatch")
    checkpoint_report_fingerprint = _digest(
        manifest.get("source_checkpoint_report_fingerprint"),
        "manifest.source_checkpoint_report_fingerprint",
    )
    model_contract = _parse_model_contract(manifest.get("model_contract"))
    runtime = _object(manifest.get("runtime"), "manifest.runtime")
    _exact_fields(runtime, _RUNTIME_FIELDS, "manifest.runtime")
    device = _string(runtime.get("device"), "manifest.runtime.device")
    dtype = _string(runtime.get("dtype"), "manifest.runtime.dtype")
    attention = _string(
        runtime.get("attention_implementation"),
        "manifest.runtime.attention_implementation",
    )
    threads = _positive_integer(
        runtime.get("torch_num_threads"), "manifest.runtime.torch_num_threads"
    )
    if (device, dtype, attention, threads) != ("cpu", "float32", "eager", 8):
        raise ValueError("target LoRA runtime must be reviewed CPU/FP32/eager/8-thread")
    adapter = _parse_adapter_contract(manifest.get("adapter"))
    optimizer = _parse_optimizer_contract(manifest.get("optimizer"))
    seed = _positive_integer(manifest.get("torch_seed"), "manifest.torch_seed")
    if seed > 2**63 - 1:
        raise ValueError("manifest.torch_seed exceeds the supported range")
    maximum = _positive_integer(
        manifest.get("max_sequence_tokens"), "manifest.max_sequence_tokens"
    )
    if maximum > 4096:
        raise ValueError("manifest.max_sequence_tokens exceeds control limit")
    messages = _parse_messages(manifest.get("messages"))
    return TargetLoRAControlSpec(
        checked_at=checked_at,
        model_id=model_id,
        revision=revision,
        source_checkpoint_manifest_fingerprint=checkpoint_manifest_fingerprint,
        source_checkpoint_report_fingerprint=checkpoint_report_fingerprint,
        model_contract=model_contract,
        device=device,
        dtype=dtype,
        attention_implementation=attention,
        torch_num_threads=threads,
        adapter=adapter,
        optimizer=optimizer,
        torch_seed=seed,
        max_sequence_tokens=maximum,
        messages=messages,
        manifest_fingerprint=_canonical_sha256(manifest),
    )


def execute_loaded_target_lora_training(
    spec: TargetLoRAControlSpec,
    *,
    model: Any,
    tokenizer: Any,
    artifact_directory: Path,
) -> TargetLoRATrainingResult:
    """Run one reviewed LoRA step on an already loaded target-compatible model."""

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
    except ImportError as error:  # pragma: no cover - environment-specific dependency error
        raise RuntimeError("torch and peft are required for target LoRA execution") from error
    _validate_loaded_model(spec, model)
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("target tokenizer must provide a chat template")
    torch.set_num_threads(spec.torch_num_threads)
    torch.manual_seed(spec.torch_seed)
    model.to("cpu")
    model.requires_grad_(False)
    model.config.use_cache = False
    input_ids, labels, sample_report = _render_training_sample(spec, tokenizer, torch)
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.inference_mode():
        base_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        base_last_logits = base_output.logits[:, -1, :].detach().to(torch.float32).cpu()
    _finite_scalar(base_output.loss, "base loss")
    peft_model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=spec.adapter.rank,
            lora_alpha=spec.adapter.alpha,
            lora_dropout=spec.adapter.dropout,
            target_modules=list(spec.adapter.target_modules),
            bias=cast(Literal["none", "all", "lora_only"], spec.adapter.bias),
        ),
    )
    peft_config = peft_model.peft_config["default"]
    peft_config.base_model_name_or_path = spec.model_id
    peft_config.revision = spec.revision
    trainable = [parameter for parameter in peft_model.parameters() if parameter.requires_grad]
    adapter_parameter_count = sum(parameter.numel() for parameter in trainable)
    expected_adapter_parameters = _expected_adapter_parameter_count(spec)
    if adapter_parameter_count != expected_adapter_parameters:
        raise RuntimeError("loaded model LoRA trainable parameter count drift")
    frozen_before = _parameter_fingerprint(
        (name, parameter)
        for name, parameter in peft_model.named_parameters()
        if not parameter.requires_grad
    )
    peft_model.eval()
    with torch.inference_mode():
        initial_output = peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        initial_logits = initial_output.logits[:, -1, :].detach().to(torch.float32).cpu()
    initial_loss = _finite_scalar(initial_output.loss, "initial adapter loss")
    initial_error = _maximum_error(base_last_logits, initial_logits)
    if initial_error != 0.0:
        raise RuntimeError("zero-initialized LoRA adapter changed base logits")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=spec.optimizer.learning_rate,
        betas=(spec.optimizer.beta1, spec.optimizer.beta2),
        eps=spec.optimizer.epsilon,
        weight_decay=spec.optimizer.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    peft_model.train()
    train_output = peft_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    _finite_scalar(train_output.loss, "training loss")
    train_output.loss.backward()
    trainable_gradient_tensors = 0
    finite_gradient_tensors = 0
    frozen_gradient_tensors = 0
    for parameter in peft_model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            trainable_gradient_tensors += 1
            if bool(torch.isfinite(parameter.grad).all().item()):
                finite_gradient_tensors += 1
        elif not parameter.requires_grad and parameter.grad is not None:
            frozen_gradient_tensors += 1
    if trainable_gradient_tensors == 0:
        raise RuntimeError("target LoRA backward produced no trainable gradients")
    if finite_gradient_tensors != trainable_gradient_tensors:
        raise RuntimeError("target LoRA backward produced a non-finite gradient")
    if frozen_gradient_tensors != 0:
        raise RuntimeError("frozen base parameter unexpectedly received a gradient")
    optimizer.step()
    frozen_after = _parameter_fingerprint(
        (name, parameter)
        for name, parameter in peft_model.named_parameters()
        if not parameter.requires_grad
    )
    if not hmac.compare_digest(frozen_before, frozen_after):
        raise RuntimeError("frozen base parameters changed during LoRA optimizer step")
    peft_model.eval()
    with torch.inference_mode():
        post_output = peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        post_logits = post_output.logits[:, -1, :].detach().to(torch.float32).cpu()
    post_loss = _finite_scalar(post_output.loss, "post-step loss")
    changed_error = _maximum_error(base_last_logits, post_logits)
    if changed_error <= 0.0:
        raise RuntimeError("target LoRA optimizer step did not change observed logits")
    adapter_state = get_peft_model_state_dict(peft_model, save_embedding_layers=False)
    state_summary = _state_tensor_summary(adapter_state, torch)
    expected_side_count = spec.model_contract.num_hidden_layers * len(
        spec.adapter.target_modules
    )
    if (
        state_summary["a_tensor_count"] != expected_side_count
        or state_summary["b_tensor_count"] != expected_side_count
        or state_summary["nonzero_a_tensor_count"] != expected_side_count
        or state_summary["nonzero_b_tensor_count"] != expected_side_count
    ):
        raise RuntimeError("target LoRA adapter tensor coverage or update drift")
    artifact = _publish_adapter_artifact(
        artifact_directory,
        spec=spec,
        peft_model=peft_model,
    )
    total_parameter_count = spec.model_contract.base_parameter_count + adapter_parameter_count
    model_report = {
        "class": type(model).__name__,
        "model_type": getattr(model.config, "model_type", None),
        "num_hidden_layers": spec.model_contract.num_hidden_layers,
        "hidden_size": spec.model_contract.hidden_size,
        "num_attention_heads": spec.model_contract.num_attention_heads,
        "num_key_value_heads": spec.model_contract.num_key_value_heads,
        "head_dim": spec.model_contract.head_dim,
        "base_parameter_count": spec.model_contract.base_parameter_count,
        "adapter_parameter_count": adapter_parameter_count,
        "total_parameter_count_with_adapter": total_parameter_count,
        "trainable_fraction_with_adapter": adapter_parameter_count
        / total_parameter_count,
        "parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in peft_model.parameters()}
        ),
    }
    execution = {
        "base_last_logits_sha256": _tensor_sha256(base_last_logits),
        "initial_adapter_last_logits_sha256": _tensor_sha256(initial_logits),
        "initial_adapter_base_max_abs_error": initial_error,
        "trained_last_logits_sha256": _tensor_sha256(post_logits),
        "post_step_vs_base_max_abs_error": changed_error,
        "initial_loss": initial_loss,
        "post_step_loss": post_loss,
        "backward_executed": True,
        "optimizer_step_count": spec.optimizer.steps,
        "optimizer": {
            key: value
            for key, value in spec.optimizer.to_dict().items()
            if key != "steps"
        },
        "trainable_gradient_tensor_count": trainable_gradient_tensors,
        "finite_gradient_tensor_count": finite_gradient_tensors,
        "frozen_base_gradient_tensor_count": frozen_gradient_tensors,
        "frozen_base_parameter_fingerprint_before": frozen_before,
        "frozen_base_parameter_fingerprint_after": frozen_after,
        "frozen_base_parameters_unchanged": True,
        "adapter_a_tensor_count": state_summary["a_tensor_count"],
        "adapter_b_tensor_count": state_summary["b_tensor_count"],
        "adapter_nonzero_a_tensor_count_after_step": state_summary[
            "nonzero_a_tensor_count"
        ],
        "adapter_nonzero_b_tensor_count_after_step": state_summary[
            "nonzero_b_tensor_count"
        ],
        "adapter_nonzero_b_element_count_after_step": state_summary[
            "nonzero_b_element_count"
        ],
    }
    del optimizer, train_output, post_output, initial_output, base_output, peft_model, model
    gc.collect()
    return TargetLoRATrainingResult(
        sample=sample_report,
        model=model_report,
        execution=execution,
        artifact=artifact,
        trained_last_logits=post_logits,
    )


def run_target_lora_control(
    spec: TargetLoRAControlSpec,
    *,
    checkpoint_spec: CheckpointControlSpec,
    checkpoint_report_path: Path,
    artifact_directory: Path,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Hash, load, train, publish, reload, and report the reviewed target control."""

    try:
        import peft
        import torch
        import transformers
        from packaging.version import Version
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment-specific dependency error
        raise RuntimeError("torch, transformers, and peft are required") from error
    checkpoint_report = verify_recorded_checkpoint_report(
        checkpoint_report_path,
        expected_manifest_fingerprint=checkpoint_spec.manifest_fingerprint,
    )
    if checkpoint_report.get("report_fingerprint") != spec.source_checkpoint_report_fingerprint:
        raise ValueError("source checkpoint recorded report fingerprint mismatch")
    _require_spec_checkpoint_binding(spec, checkpoint_spec)
    snapshot_directory = download_checkpoint_snapshot(
        checkpoint_spec, local_files_only=local_files_only
    )
    initial_snapshot = verify_checkpoint_snapshot(checkpoint_spec, snapshot_directory)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        initial_snapshot.directory,
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
        initial_snapshot.directory,
        trust_remote_code=False,
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        attn_implementation=spec.attention_implementation,
        **dtype_argument,
    )
    training = execute_loaded_target_lora_training(
        spec,
        model=model,
        tokenizer=tokenizer,
        artifact_directory=artifact_directory,
    )
    del model
    gc.collect()
    reload_snapshot = verify_checkpoint_snapshot(checkpoint_spec, snapshot_directory)
    reload_base = AutoModelForCausalLM.from_pretrained(
        reload_snapshot.directory,
        trust_remote_code=False,
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        attn_implementation=spec.attention_implementation,
        **dtype_argument,
    )
    reload_base.config.use_cache = False
    reloaded = PeftModel.from_pretrained(
        reload_base,
        artifact_directory,
        is_trainable=False,
        local_files_only=True,
    ).eval()
    input_ids = torch.tensor(
        [cast(list[int], training.sample["input_token_ids"])], dtype=torch.long
    )
    prompt_count = cast(int, training.sample["prompt_token_count"])
    labels = input_ids.clone()
    labels[:, :prompt_count] = -100
    with torch.inference_mode():
        reloaded_output = reloaded(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        reloaded_logits = (
            reloaded_output.logits[:, -1, :].detach().to(torch.float32).cpu()
        )
    reloaded_loss = _finite_scalar(reloaded_output.loss, "reloaded adapter loss")
    reload_error = _maximum_error(training.trained_last_logits, reloaded_logits)
    if reload_error != 0.0 or not torch.equal(
        training.trained_last_logits, reloaded_logits
    ):
        raise RuntimeError("saved target LoRA adapter did not reload exactly")
    source_artifacts = _object(checkpoint_report.get("artifacts"), "checkpoint.artifacts")
    selected_file_count = _positive_integer(
        source_artifacts.get("selected_file_count"),
        "checkpoint.artifacts.selected_file_count",
    )
    selected_total_bytes = _positive_integer(
        source_artifacts.get("selected_total_bytes"),
        "checkpoint.artifacts.selected_total_bytes",
    )
    report: dict[str, object] = {
        "report_version": TARGET_LORA_REPORT_VERSION,
        "control_manifest_fingerprint": spec.manifest_fingerprint,
        "checked_at": spec.checked_at,
        "source": {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "checkpoint_report_fingerprint": spec.source_checkpoint_report_fingerprint,
            "selected_file_count": selected_file_count,
            "selected_total_bytes": selected_total_bytes,
            "all_selected_file_bytes_verified_before_initial_load": True,
            "all_selected_file_bytes_verified_before_reload": True,
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
            "device": spec.device,
            "dtype": spec.dtype,
            "attention_implementation": spec.attention_implementation,
            "torch_num_threads": torch.get_num_threads(),
            "cuda_executed": False,
        },
        "sample": dict(training.sample),
        "model": dict(training.model),
        "execution": dict(training.execution),
        "adapter_artifact": training.artifact.to_dict(),
        "round_trip": {
            "adapter_loaded_with_peft": True,
            "trained_and_reloaded_logits_exact": True,
            "maximum_logit_error": reload_error,
            "reloaded_last_logits_sha256": _tensor_sha256(reloaded_logits),
            "reloaded_loss": reloaded_loss,
        },
        "scope": _expected_scope(),
        "evidence_boundary": TARGET_LORA_EVIDENCE_BOUNDARY,
    }
    report["report_fingerprint"] = _canonical_sha256(report)
    verify_recorded_target_lora_report(
        spec,
        checkpoint_spec=checkpoint_spec,
        checkpoint_report_path=checkpoint_report_path,
        report=report,
        artifact_directory=artifact_directory,
    )
    return report


def verify_target_lora_adapter_artifact(
    root: Path, *, spec: TargetLoRAControlSpec
) -> TargetLoRAArtifactVerification:
    """Verify the exact adapter file set, config, tensors, and manifest identity."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("target LoRA artifact root must be a regular directory")
    manifest_path = root / TARGET_LORA_ARTIFACT_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("target LoRA artifact manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("target LoRA artifact manifest exceeds byte limit")
    manifest = _strict_json_object(manifest_bytes, "target LoRA artifact manifest")
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("target LoRA artifact manifest must be canonical JSON")
    _exact_fields(manifest, _ARTIFACT_FIELDS, "target LoRA artifact manifest")
    if manifest.get("schema_version") != TARGET_LORA_ARTIFACT_VERSION:
        raise ValueError("target LoRA artifact schema_version is unsupported")
    if manifest.get("control_manifest_fingerprint") != spec.manifest_fingerprint:
        raise ValueError("target LoRA artifact control fingerprint mismatch")
    if (
        manifest.get("source_checkpoint_manifest_fingerprint")
        != spec.source_checkpoint_manifest_fingerprint
    ):
        raise ValueError("target LoRA artifact checkpoint fingerprint mismatch")
    supplied_manifest_fingerprint = _digest(
        manifest.get("manifest_fingerprint"), "artifact.manifest_fingerprint"
    )
    unsigned = dict(manifest)
    del unsigned["manifest_fingerprint"]
    expected_manifest_fingerprint = _canonical_sha256(unsigned)
    if not hmac.compare_digest(supplied_manifest_fingerprint, expected_manifest_fingerprint):
        raise ValueError("target LoRA artifact manifest fingerprint mismatch")
    identity = _object(manifest.get("identity"), "artifact.identity")
    _exact_fields(identity, _ARTIFACT_IDENTITY_FIELDS, "artifact.identity")
    expected_identity = {
        "adapter_name": "default",
        "architecture": spec.model_contract.expected_model_class,
        "model_id": spec.model_id,
        "revision": spec.revision,
    }
    if identity != expected_identity:
        raise ValueError("target LoRA artifact identity drift")
    contract = _object(manifest.get("contract"), "artifact.contract")
    _exact_fields(contract, _ARTIFACT_CONTRACT_FIELDS, "artifact.contract")
    if contract != _artifact_contract(spec):
        raise ValueError("target LoRA artifact contract drift")
    files = _parse_artifact_files(manifest.get("files"))
    actual_files = _describe_payload_files(root)
    if files != actual_files:
        raise ValueError("target LoRA artifact file set, size, or digest mismatch")
    _validate_adapter_config(root / "adapter_config.json", spec)
    tensor_summary = _adapter_tensor_summary(root / "adapter_model.safetensors", spec)
    tensors = _object(manifest.get("tensors"), "artifact.tensors")
    _exact_fields(tensors, _ARTIFACT_TENSOR_FIELDS, "artifact.tensors")
    if tensors != tensor_summary:
        raise ValueError("target LoRA artifact tensor contract mismatch")
    return TargetLoRAArtifactVerification(
        manifest_fingerprint=supplied_manifest_fingerprint,
        manifest_bytes=len(manifest_bytes),
        file_count=len(files),
        total_file_bytes=sum(cast(int, item["bytes"]) for item in files),
        files=tuple(files),
        tensor_count=cast(int, tensors["tensor_count"]),
        total_tensor_numel=cast(int, tensors["total_numel"]),
        tensor_dtypes=tuple(cast(list[str], tensors["dtypes"])),
        descriptor_set_sha256=cast(str, tensors["descriptor_set_sha256"]),
        a_tensor_count=cast(int, tensors["a_tensor_count"]),
        b_tensor_count=cast(int, tensors["b_tensor_count"]),
        nonzero_a_tensor_count=cast(int, tensors["nonzero_a_tensor_count"]),
        nonzero_b_tensor_count=cast(int, tensors["nonzero_b_tensor_count"]),
        nonzero_b_element_count=cast(int, tensors["nonzero_b_element_count"]),
    )


def verify_recorded_target_lora_report(
    spec: TargetLoRAControlSpec,
    *,
    checkpoint_spec: CheckpointControlSpec,
    checkpoint_report_path: Path,
    report: Mapping[str, object],
    artifact_directory: Path,
) -> None:
    """Verify recorded identities and current adapter bytes without replaying training."""

    _require_spec_checkpoint_binding(spec, checkpoint_spec)
    checkpoint_report = verify_recorded_checkpoint_report(
        checkpoint_report_path,
        expected_manifest_fingerprint=checkpoint_spec.manifest_fingerprint,
    )
    if checkpoint_report.get("report_fingerprint") != spec.source_checkpoint_report_fingerprint:
        raise ValueError("recorded target LoRA source report fingerprint mismatch")
    _exact_fields(report, _REPORT_FIELDS, "target LoRA report")
    if report.get("report_version") != TARGET_LORA_REPORT_VERSION:
        raise ValueError("target LoRA report_version is unsupported")
    if report.get("control_manifest_fingerprint") != spec.manifest_fingerprint:
        raise ValueError("target LoRA report control fingerprint mismatch")
    if report.get("checked_at") != spec.checked_at:
        raise ValueError("target LoRA report checked_at drift")
    if report.get("evidence_boundary") != TARGET_LORA_EVIDENCE_BOUNDARY:
        raise ValueError("target LoRA report evidence boundary drift")
    supplied = _digest(report.get("report_fingerprint"), "report.report_fingerprint")
    unsigned = dict(report)
    del unsigned["report_fingerprint"]
    if not hmac.compare_digest(supplied, _canonical_sha256(unsigned)):
        raise ValueError("target LoRA report fingerprint mismatch")
    source = _object(report.get("source"), "report.source")
    _exact_fields(source, _REPORT_SOURCE_FIELDS, "report.source")
    checkpoint_artifacts = _object(
        checkpoint_report.get("artifacts"), "checkpoint report artifacts"
    )
    expected_source = {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "checkpoint_report_fingerprint": spec.source_checkpoint_report_fingerprint,
        "selected_file_count": checkpoint_artifacts.get("selected_file_count"),
        "selected_total_bytes": checkpoint_artifacts.get("selected_total_bytes"),
        "all_selected_file_bytes_verified_before_initial_load": True,
        "all_selected_file_bytes_verified_before_reload": True,
    }
    if source != expected_source:
        raise ValueError("target LoRA report source evidence drift")
    runtime = _object(report.get("runtime"), "report.runtime")
    _exact_fields(runtime, _REPORT_RUNTIME_FIELDS, "report.runtime")
    if (
        runtime.get("device"),
        runtime.get("dtype"),
        runtime.get("attention_implementation"),
        runtime.get("torch_num_threads"),
        runtime.get("cuda_executed"),
    ) != (spec.device, spec.dtype, spec.attention_implementation, spec.torch_num_threads, False):
        raise ValueError("target LoRA report runtime contract drift")
    for key in (
        "python_implementation",
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
        "peft_version",
    ):
        _string(runtime.get(key), f"report.runtime.{key}")
    _verify_sample_report(spec, _object(report.get("sample"), "report.sample"))
    _verify_model_report(spec, _object(report.get("model"), "report.model"))
    execution = _object(report.get("execution"), "report.execution")
    _verify_execution_report(spec, execution)
    artifact = verify_target_lora_adapter_artifact(artifact_directory, spec=spec)
    if report.get("adapter_artifact") != artifact.to_dict():
        raise ValueError("target LoRA report adapter artifact evidence drift")
    if (
        execution.get("adapter_a_tensor_count") != artifact.a_tensor_count
        or execution.get("adapter_b_tensor_count") != artifact.b_tensor_count
        or execution.get("adapter_nonzero_a_tensor_count_after_step")
        != artifact.nonzero_a_tensor_count
        or execution.get("adapter_nonzero_b_tensor_count_after_step")
        != artifact.nonzero_b_tensor_count
        or execution.get("adapter_nonzero_b_element_count_after_step")
        != artifact.nonzero_b_element_count
    ):
        raise ValueError("target LoRA execution/artifact tensor evidence drift")
    round_trip = _object(report.get("round_trip"), "report.round_trip")
    _exact_fields(round_trip, _REPORT_ROUND_TRIP_FIELDS, "report.round_trip")
    if (
        round_trip.get("adapter_loaded_with_peft") is not True
        or round_trip.get("trained_and_reloaded_logits_exact") is not True
        or round_trip.get("maximum_logit_error") != 0.0
        or round_trip.get("reloaded_last_logits_sha256")
        != execution.get("trained_last_logits_sha256")
        or round_trip.get("reloaded_loss") != execution.get("post_step_loss")
    ):
        raise ValueError("target LoRA report round-trip evidence drift")
    _finite_number(round_trip.get("reloaded_loss"), "report.round_trip.reloaded_loss")
    scope = _object(report.get("scope"), "report.scope")
    _exact_fields(scope, _REPORT_SCOPE_FIELDS, "report.scope")
    if scope != _expected_scope():
        raise ValueError("target LoRA report scope drift")


def load_recorded_target_lora_report(
    path: Path,
    *,
    spec: TargetLoRAControlSpec,
    checkpoint_spec: CheckpointControlSpec,
    checkpoint_report_path: Path,
    artifact_directory: Path,
) -> dict[str, object]:
    """Load and verify a recorded report with duplicate/non-finite rejection."""

    report = _load_strict_json_file(path, max_bytes=_MAX_REPORT_BYTES)
    verify_recorded_target_lora_report(
        spec,
        checkpoint_spec=checkpoint_spec,
        checkpoint_report_path=checkpoint_report_path,
        report=report,
        artifact_directory=artifact_directory,
    )
    return report


def _publish_adapter_artifact(
    target: Path, *, spec: TargetLoRAControlSpec, peft_model: Any
) -> TargetLoRAArtifactVerification:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace target LoRA artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.building-", dir=target.parent
    ) as temporary:
        temporary_path = Path(temporary)
        peft_model.save_pretrained(
            temporary_path,
            safe_serialization=True,
            save_embedding_layers=False,
        )
        actual_names = {
            item.name for item in temporary_path.iterdir() if item.is_file()
        }
        if actual_names != _ADAPTER_FILES:
            raise RuntimeError("PEFT adapter output file set drift")
        _write_artifact_manifest(temporary_path, spec=spec)
        verify_target_lora_adapter_artifact(temporary_path, spec=spec)
        os.replace(temporary_path, target)
    return verify_target_lora_adapter_artifact(target, spec=spec)


def _write_artifact_manifest(root: Path, *, spec: TargetLoRAControlSpec) -> None:
    manifest_path = root / TARGET_LORA_ARTIFACT_MANIFEST
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("target LoRA artifact manifest already exists")
    manifest: dict[str, object] = {
        "schema_version": TARGET_LORA_ARTIFACT_VERSION,
        "control_manifest_fingerprint": spec.manifest_fingerprint,
        "source_checkpoint_manifest_fingerprint": (
            spec.source_checkpoint_manifest_fingerprint
        ),
        "identity": {
            "adapter_name": "default",
            "architecture": spec.model_contract.expected_model_class,
            "model_id": spec.model_id,
            "revision": spec.revision,
        },
        "contract": _artifact_contract(spec),
        "files": _describe_payload_files(root),
        "tensors": _adapter_tensor_summary(root / "adapter_model.safetensors", spec),
    }
    manifest["manifest_fingerprint"] = _canonical_sha256(manifest)
    payload = canonical_json_bytes(manifest)
    with manifest_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_contract(spec: TargetLoRAControlSpec) -> dict[str, object]:
    return {
        "adapter_type": "LORA",
        "task_type": spec.adapter.task_type,
        "target_modules": list(spec.adapter.target_modules),
        "rank": spec.adapter.rank,
        "alpha": spec.adapter.alpha,
        "dropout": spec.adapter.dropout,
        "bias": spec.adapter.bias,
        "base_reference_kind": "model_id_plus_immutable_revision",
        "quantized_base": False,
        "full_weight_merge_performed": False,
    }


def _describe_payload_files(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for item in root.iterdir():
        if item.name == TARGET_LORA_ARTIFACT_MANIFEST:
            continue
        if item.is_symlink() or not item.is_file():
            raise ValueError("target LoRA artifact contains non-regular payload")
        if item.name not in _ADAPTER_FILES:
            raise ValueError("target LoRA artifact contains an unexpected payload file")
        size, digest = _hash_open_file(item)
        files.append({"path": item.name, "bytes": size, "sha256": digest})
    files.sort(key=lambda item: cast(str, item["path"]))
    if {cast(str, item["path"]) for item in files} != _ADAPTER_FILES:
        raise ValueError("target LoRA artifact payload file set is incomplete")
    return files


def _parse_artifact_files(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(_ADAPTER_FILES):
        raise ValueError("artifact.files must contain the exact adapter payload file set")
    files: list[dict[str, object]] = []
    for index, item in enumerate(value):
        record = _object(item, f"artifact.files[{index}]")
        _exact_fields(record, _ARTIFACT_FILE_FIELDS, f"artifact.files[{index}]")
        path = _string(record.get("path"), f"artifact.files[{index}].path")
        if path not in _ADAPTER_FILES:
            raise ValueError("artifact.files contains an unsupported path")
        size = _positive_integer(record.get("bytes"), f"artifact.files[{index}].bytes")
        digest = _digest(record.get("sha256"), f"artifact.files[{index}].sha256")
        files.append({"path": path, "bytes": size, "sha256": digest})
    if files != sorted(files, key=lambda item: cast(str, item["path"])):
        raise ValueError("artifact.files must be sorted by path")
    if len({cast(str, item["path"]) for item in files}) != len(files):
        raise ValueError("artifact.files contains a duplicate path")
    return files


def _validate_adapter_config(path: Path, spec: TargetLoRAControlSpec) -> None:
    config = _load_strict_json_file(path, max_bytes=_MAX_MANIFEST_BYTES)
    _exact_fields(config, _ADAPTER_CONFIG_FIELDS, "target LoRA adapter_config")
    expected: dict[str, object] = {
        "alora_invocation_tokens": None,
        "alpha_pattern": {},
        "arrow_config": None,
        "auto_mapping": None,
        "base_model_name_or_path": spec.model_id,
        "bias": spec.adapter.bias,
        "corda_config": None,
        "ensure_weight_tying": False,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": spec.adapter.alpha,
        "lora_bias": False,
        "lora_dropout": spec.adapter.dropout,
        "lora_ga_config": None,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "monteclora_config": None,
        "peft_type": "LORA",
        "peft_version": "0.20.0",
        "qalora_group_size": 16,
        "r": spec.adapter.rank,
        "rank_pattern": {},
        "revision": spec.revision,
        "target_parameters": None,
        "task_type": spec.adapter.task_type,
        "trainable_token_indices": None,
        "use_bdlora": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
        "velora_config": None,
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(f"target LoRA adapter_config {key} drift")
    target_modules = config.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or len(target_modules) != len(spec.adapter.target_modules)
        or any(not isinstance(item, str) for item in target_modules)
        or set(target_modules) != set(spec.adapter.target_modules)
    ):
        raise ValueError("target LoRA adapter_config target_modules drift")


def _adapter_tensor_summary(path: Path, spec: TargetLoRAControlSpec) -> dict[str, object]:
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - environment-specific dependency error
        raise RuntimeError("safetensors and torch are required for adapter verification") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError("target LoRA adapter weights must be a regular safetensors file")
    descriptors: list[dict[str, object]] = []
    coverage: dict[tuple[int, str, str], tuple[int, ...]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        if handle.metadata() != {"format": "pt"}:
            raise ValueError("target LoRA adapter safetensors metadata drift")
        keys = list(handle.keys())
        for key in keys:
            match = _TENSOR_KEY.fullmatch(key)
            if match is None:
                raise ValueError(f"target LoRA adapter contains unexpected tensor: {key}")
            layer = int(match.group("layer"))
            module = match.group("module")
            side = match.group("side")
            if layer >= spec.model_contract.num_hidden_layers:
                raise ValueError("target LoRA adapter tensor layer is out of range")
            shape = tuple(int(value) for value in handle.get_slice(key).get_shape())
            dtype = str(handle.get_slice(key).get_dtype())
            expected_shape = _expected_adapter_shape(spec, module=module, side=side)
            if shape != expected_shape or dtype != "F32":
                raise ValueError("target LoRA adapter tensor shape or dtype drift")
            identity = (layer, module, side)
            if identity in coverage:
                raise ValueError("target LoRA adapter tensor coverage is duplicated")
            coverage[identity] = shape
            descriptors.append({"name": key, "shape": list(shape), "dtype": dtype})
    expected_coverage = {
        (layer, module, side)
        for layer in range(spec.model_contract.num_hidden_layers)
        for module in spec.adapter.target_modules
        for side in ("A", "B")
    }
    if set(coverage) != expected_coverage:
        raise ValueError("target LoRA adapter does not cover every reviewed A/B target")
    descriptors.sort(key=lambda item: cast(str, item["name"]))
    state = load_file(path, device="cpu")
    nonzero_a = 0
    nonzero_b = 0
    nonzero_b_elements = 0
    for key, tensor in state.items():
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("target LoRA adapter contains non-finite tensor values")
        nonzero = int(torch.count_nonzero(tensor).item())
        if ".lora_A." in key and nonzero > 0:
            nonzero_a += 1
        if ".lora_B." in key:
            if nonzero > 0:
                nonzero_b += 1
            nonzero_b_elements += nonzero
    expected_side_count = spec.model_contract.num_hidden_layers * len(
        spec.adapter.target_modules
    )
    if nonzero_a != expected_side_count or nonzero_b != expected_side_count:
        raise ValueError("target LoRA adapter has an all-zero A or B tensor")
    return {
        "tensor_count": len(descriptors),
        "total_numel": sum(
            math.prod(cast(list[int], item["shape"])) for item in descriptors
        ),
        "dtypes": sorted({cast(str, item["dtype"]) for item in descriptors}),
        "descriptor_set_sha256": _canonical_sha256({"tensors": descriptors}),
        "a_tensor_count": expected_side_count,
        "b_tensor_count": expected_side_count,
        "nonzero_a_tensor_count": nonzero_a,
        "nonzero_b_tensor_count": nonzero_b,
        "nonzero_b_element_count": nonzero_b_elements,
    }


def _state_tensor_summary(state: Mapping[str, Any], torch: Any) -> dict[str, int]:
    a_count = 0
    b_count = 0
    nonzero_a = 0
    nonzero_b = 0
    nonzero_b_elements = 0
    for key, tensor in state.items():
        nonzero = int(torch.count_nonzero(tensor.detach()).item())
        if ".lora_A." in key:
            a_count += 1
            nonzero_a += int(nonzero > 0)
        elif ".lora_B." in key:
            b_count += 1
            nonzero_b += int(nonzero > 0)
            nonzero_b_elements += nonzero
        else:
            raise RuntimeError("target LoRA state contains a non-adapter tensor")
    return {
        "a_tensor_count": a_count,
        "b_tensor_count": b_count,
        "nonzero_a_tensor_count": nonzero_a,
        "nonzero_b_tensor_count": nonzero_b,
        "nonzero_b_element_count": nonzero_b_elements,
    }


def _render_training_sample(
    spec: TargetLoRAControlSpec, tokenizer: Any, torch: Any
) -> tuple[Any, Any, dict[str, object]]:
    messages = [{"role": item.role, "content": item.content} for item in spec.messages]
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    if not isinstance(prompt, torch.Tensor) or not isinstance(full, torch.Tensor):
        raise RuntimeError("target chat template must return tensors")
    prompt = prompt.to(device="cpu", dtype=torch.long)
    full = full.to(device="cpu", dtype=torch.long)
    if prompt.ndim != 2 or full.ndim != 2 or prompt.shape[0] != 1 or full.shape[0] != 1:
        raise RuntimeError("target chat template must return [1, tokens] tensors")
    prompt_count = int(prompt.shape[1])
    total_count = int(full.shape[1])
    if prompt_count < 1 or total_count <= prompt_count:
        raise RuntimeError("target assistant response must add supervised tokens")
    if total_count > spec.max_sequence_tokens:
        raise RuntimeError("target training sample exceeds reviewed sequence limit")
    if not torch.equal(full[:, :prompt_count], prompt):
        raise RuntimeError("target full conversation does not preserve generation prefix")
    if torch.any(full < 0) or torch.any(full >= len(tokenizer)):
        raise RuntimeError("target chat template returned out-of-vocabulary token ids")
    labels = full.clone()
    labels[:, :prompt_count] = -100
    token_ids = [int(value) for value in full[0].tolist()]
    label_ids = [-100] * prompt_count + token_ids[prompt_count:]
    chat_template = cast(str, tokenizer.chat_template)
    sample = {
        "input_token_ids": token_ids,
        "prompt_token_count": prompt_count,
        "supervised_token_count": total_count - prompt_count,
        "max_sequence_tokens": spec.max_sequence_tokens,
        "assistant_only_loss_mask_applied": True,
        "input_ids_fingerprint": _canonical_sha256({"token_ids": token_ids}),
        "label_ids_fingerprint": _canonical_sha256({"label_ids": label_ids}),
        "chat_template_fingerprint": _sha256_bytes(chat_template.encode("utf-8")),
    }
    return full, labels, sample


def _validate_loaded_model(spec: TargetLoRAControlSpec, model: Any) -> None:
    contract = spec.model_contract
    if type(model).__name__ != contract.expected_model_class:
        raise ValueError("loaded model class does not match target LoRA contract")
    config = model.config
    observed_hidden_size = getattr(config, "hidden_size", None)
    observed_attention_heads = getattr(config, "num_attention_heads", None)
    observed_head_dim = getattr(config, "head_dim", None)
    if (
        observed_head_dim is None
        and isinstance(observed_hidden_size, int)
        and isinstance(observed_attention_heads, int)
        and observed_attention_heads > 0
        and observed_hidden_size % observed_attention_heads == 0
    ):
        observed_head_dim = observed_hidden_size // observed_attention_heads
    observed = {
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": observed_hidden_size,
        "num_attention_heads": observed_attention_heads,
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": observed_head_dim,
    }
    expected = {
        "model_type": contract.expected_model_type,
        "num_hidden_layers": contract.num_hidden_layers,
        "hidden_size": contract.hidden_size,
        "num_attention_heads": contract.num_attention_heads,
        "num_key_value_heads": contract.num_key_value_heads,
        "head_dim": contract.head_dim,
    }
    if observed != expected:
        raise ValueError("loaded model architecture does not match target LoRA contract")
    parameters = list(model.parameters())
    if sum(parameter.numel() for parameter in parameters) != contract.base_parameter_count:
        raise ValueError("loaded model parameter count does not match target LoRA contract")
    if {str(parameter.dtype) for parameter in parameters} != {"torch.float32"}:
        raise ValueError("loaded target LoRA base parameters must all be float32")


def _expected_adapter_shape(
    spec: TargetLoRAControlSpec, *, module: str, side: str
) -> tuple[int, ...]:
    hidden = spec.model_contract.hidden_size
    rank = spec.adapter.rank
    if side == "A":
        return (rank, hidden)
    if module == "q_proj":
        return (spec.model_contract.num_attention_heads * spec.model_contract.head_dim, rank)
    if module == "v_proj":
        return (
            spec.model_contract.num_key_value_heads * spec.model_contract.head_dim,
            rank,
        )
    raise ValueError("unsupported target LoRA module")


def _expected_adapter_parameter_count(spec: TargetLoRAControlSpec) -> int:
    per_layer = sum(
        math.prod(_expected_adapter_shape(spec, module=module, side=side))
        for module in spec.adapter.target_modules
        for side in ("A", "B")
    )
    return spec.model_contract.num_hidden_layers * per_layer


def _verify_sample_report(
    spec: TargetLoRAControlSpec, sample: Mapping[str, object]
) -> None:
    _exact_fields(sample, _REPORT_SAMPLE_FIELDS, "report.sample")
    token_ids = sample.get("input_token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in token_ids
        )
    ):
        raise ValueError("report.sample.input_token_ids is invalid")
    prompt_count = _positive_integer(
        sample.get("prompt_token_count"), "report.sample.prompt_token_count"
    )
    supervised_count = _positive_integer(
        sample.get("supervised_token_count"), "report.sample.supervised_token_count"
    )
    if prompt_count + supervised_count != len(token_ids):
        raise ValueError("report.sample token boundary accounting mismatch")
    if sample.get("max_sequence_tokens") != spec.max_sequence_tokens:
        raise ValueError("report.sample sequence limit drift")
    if len(token_ids) > spec.max_sequence_tokens:
        raise ValueError("report.sample exceeds sequence limit")
    if sample.get("assistant_only_loss_mask_applied") is not True:
        raise ValueError("report.sample assistant-only mask evidence drift")
    if sample.get("input_ids_fingerprint") != _canonical_sha256(
        {"token_ids": token_ids}
    ):
        raise ValueError("report.sample input token fingerprint mismatch")
    label_ids = [-100] * prompt_count + token_ids[prompt_count:]
    if sample.get("label_ids_fingerprint") != _canonical_sha256(
        {"label_ids": label_ids}
    ):
        raise ValueError("report.sample label fingerprint mismatch")
    _digest(
        sample.get("chat_template_fingerprint"),
        "report.sample.chat_template_fingerprint",
    )


def _verify_model_report(
    spec: TargetLoRAControlSpec, model: Mapping[str, object]
) -> None:
    _exact_fields(model, _REPORT_MODEL_FIELDS, "report.model")
    contract = spec.model_contract
    expected_values = {
        "class": contract.expected_model_class,
        "model_type": contract.expected_model_type,
        "num_hidden_layers": contract.num_hidden_layers,
        "hidden_size": contract.hidden_size,
        "num_attention_heads": contract.num_attention_heads,
        "num_key_value_heads": contract.num_key_value_heads,
        "head_dim": contract.head_dim,
        "base_parameter_count": contract.base_parameter_count,
        "adapter_parameter_count": _expected_adapter_parameter_count(spec),
        "total_parameter_count_with_adapter": contract.base_parameter_count
        + _expected_adapter_parameter_count(spec),
        "parameter_dtypes": ["torch.float32"],
    }
    for key, expected in expected_values.items():
        if model.get(key) != expected:
            raise ValueError(f"report.model.{key} drift")
    expected_fraction = _expected_adapter_parameter_count(spec) / cast(
        int, expected_values["total_parameter_count_with_adapter"]
    )
    fraction = _finite_number(
        model.get("trainable_fraction_with_adapter"),
        "report.model.trainable_fraction_with_adapter",
    )
    if fraction != expected_fraction:
        raise ValueError("report.model trainable fraction drift")


def _verify_execution_report(
    spec: TargetLoRAControlSpec, execution: Mapping[str, object]
) -> None:
    _exact_fields(execution, _REPORT_EXECUTION_FIELDS, "report.execution")
    for key in (
        "base_last_logits_sha256",
        "initial_adapter_last_logits_sha256",
        "trained_last_logits_sha256",
        "frozen_base_parameter_fingerprint_before",
        "frozen_base_parameter_fingerprint_after",
    ):
        _digest(execution.get(key), f"report.execution.{key}")
    if execution.get("base_last_logits_sha256") != execution.get(
        "initial_adapter_last_logits_sha256"
    ):
        raise ValueError("report.execution initial adapter/base logit identity drift")
    if execution.get("initial_adapter_base_max_abs_error") != 0.0:
        raise ValueError("report.execution initial adapter neutrality drift")
    if _finite_number(
        execution.get("post_step_vs_base_max_abs_error"),
        "report.execution.post_step_vs_base_max_abs_error",
    ) <= 0:
        raise ValueError("report.execution does not observe an adapter logit change")
    _finite_number(execution.get("initial_loss"), "report.execution.initial_loss")
    _finite_number(execution.get("post_step_loss"), "report.execution.post_step_loss")
    if execution.get("backward_executed") is not True:
        raise ValueError("report.execution backward evidence drift")
    if execution.get("optimizer_step_count") != spec.optimizer.steps:
        raise ValueError("report.execution optimizer step count drift")
    optimizer = _object(execution.get("optimizer"), "report.execution.optimizer")
    _exact_fields(optimizer, _REPORT_OPTIMIZER_FIELDS, "report.execution.optimizer")
    expected_optimizer = spec.optimizer.to_dict()
    del expected_optimizer["steps"]
    if optimizer != expected_optimizer:
        raise ValueError("report.execution optimizer contract drift")
    trainable_gradient_count = _positive_integer(
        execution.get("trainable_gradient_tensor_count"),
        "report.execution.trainable_gradient_tensor_count",
    )
    if execution.get("finite_gradient_tensor_count") != trainable_gradient_count:
        raise ValueError("report.execution finite gradient count drift")
    if execution.get("frozen_base_gradient_tensor_count") != 0:
        raise ValueError("report.execution frozen gradient evidence drift")
    if (
        execution.get("frozen_base_parameters_unchanged") is not True
        or execution.get("frozen_base_parameter_fingerprint_before")
        != execution.get("frozen_base_parameter_fingerprint_after")
    ):
        raise ValueError("report.execution frozen base fingerprint drift")
    expected_side = spec.model_contract.num_hidden_layers * len(
        spec.adapter.target_modules
    )
    for key in (
        "adapter_a_tensor_count",
        "adapter_b_tensor_count",
        "adapter_nonzero_a_tensor_count_after_step",
        "adapter_nonzero_b_tensor_count_after_step",
    ):
        if execution.get(key) != expected_side:
            raise ValueError(f"report.execution {key} drift")
    if _positive_integer(
        execution.get("adapter_nonzero_b_element_count_after_step"),
        "report.execution.adapter_nonzero_b_element_count_after_step",
    ) < expected_side:
        raise ValueError("report.execution adapter B update evidence drift")


def _expected_scope() -> dict[str, bool]:
    return {
        "target_checkpoint_weights_loaded": True,
        "target_checkpoint_backward_executed": True,
        "assistant_only_training_boundary_observed": True,
        "peft_adapter_saved_and_reloaded": True,
        "trust_remote_code": False,
        "qlora_or_quantized_base_executed": False,
        "cuda_executed": False,
        "vllm_or_serving_runtime_executed": False,
        "full_weight_merge_executed": False,
        "optimizer_scheduler_rng_resume_state_exported": False,
        "representative_dataset_used": False,
        "model_quality_or_convergence_proven": False,
        "performance_or_peak_memory_benchmarked": False,
        "publisher_or_trainer_authenticated": False,
        "license_compatibility_proven": False,
        "production_safety_proven": False,
        "verification_to_loader_reopen_toctou_eliminated": False,
    }


def _require_spec_checkpoint_binding(
    spec: TargetLoRAControlSpec, checkpoint_spec: CheckpointControlSpec
) -> None:
    if (
        spec.model_id,
        spec.revision,
        spec.source_checkpoint_manifest_fingerprint,
    ) != (
        checkpoint_spec.model_id,
        checkpoint_spec.revision,
        checkpoint_spec.manifest_fingerprint,
    ):
        raise ValueError("target LoRA control/checkpoint binding mismatch")


def _parse_model_contract(value: object) -> TargetLoRAModelContract:
    record = _object(value, "manifest.model_contract")
    _exact_fields(record, _MODEL_CONTRACT_FIELDS, "manifest.model_contract")
    expected_class = _string(
        record.get("expected_model_class"), "manifest.model_contract.expected_model_class"
    )
    expected_type = _string(
        record.get("expected_model_type"), "manifest.model_contract.expected_model_type"
    )
    result = TargetLoRAModelContract(
        expected_model_class=expected_class,
        expected_model_type=expected_type,
        num_hidden_layers=_positive_integer(
            record.get("num_hidden_layers"), "manifest.model_contract.num_hidden_layers"
        ),
        hidden_size=_positive_integer(
            record.get("hidden_size"), "manifest.model_contract.hidden_size"
        ),
        num_attention_heads=_positive_integer(
            record.get("num_attention_heads"),
            "manifest.model_contract.num_attention_heads",
        ),
        num_key_value_heads=_positive_integer(
            record.get("num_key_value_heads"),
            "manifest.model_contract.num_key_value_heads",
        ),
        head_dim=_positive_integer(
            record.get("head_dim"), "manifest.model_contract.head_dim"
        ),
        base_parameter_count=_positive_integer(
            record.get("base_parameter_count"),
            "manifest.model_contract.base_parameter_count",
        ),
    )
    if result.hidden_size != result.num_attention_heads * result.head_dim:
        raise ValueError("manifest.model_contract hidden/head dimensions are inconsistent")
    if result.num_attention_heads % result.num_key_value_heads != 0:
        raise ValueError("manifest.model_contract attention head ratio is invalid")
    return result


def _parse_adapter_contract(value: object) -> TargetLoRAAdapterContract:
    record = _object(value, "manifest.adapter")
    _exact_fields(record, _ADAPTER_FIELDS, "manifest.adapter")
    modules = record.get("target_modules")
    if modules != ["q_proj", "v_proj"]:
        raise ValueError("manifest.adapter target_modules must be reviewed q_proj/v_proj")
    task_type = _string(record.get("task_type"), "manifest.adapter.task_type")
    rank = _positive_integer(record.get("rank"), "manifest.adapter.rank")
    alpha = _positive_integer(record.get("alpha"), "manifest.adapter.alpha")
    dropout = _finite_number(record.get("dropout"), "manifest.adapter.dropout")
    bias = _string(record.get("bias"), "manifest.adapter.bias")
    if (task_type, rank, alpha, dropout, bias) != ("CAUSAL_LM", 4, 8, 0.0, "none"):
        raise ValueError("manifest.adapter must match the reviewed LoRA contract")
    return TargetLoRAAdapterContract(
        task_type=task_type,
        target_modules=("q_proj", "v_proj"),
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=bias,
    )


def _parse_optimizer_contract(value: object) -> TargetLoRAOptimizerContract:
    record = _object(value, "manifest.optimizer")
    _exact_fields(record, _OPTIMIZER_FIELDS, "manifest.optimizer")
    result = TargetLoRAOptimizerContract(
        name=_string(record.get("name"), "manifest.optimizer.name"),
        learning_rate=_finite_number(
            record.get("learning_rate"), "manifest.optimizer.learning_rate"
        ),
        beta1=_finite_number(record.get("beta1"), "manifest.optimizer.beta1"),
        beta2=_finite_number(record.get("beta2"), "manifest.optimizer.beta2"),
        epsilon=_finite_number(
            record.get("epsilon"), "manifest.optimizer.epsilon"
        ),
        weight_decay=_finite_number(
            record.get("weight_decay"), "manifest.optimizer.weight_decay"
        ),
        steps=_positive_integer(record.get("steps"), "manifest.optimizer.steps"),
    )
    if result != TargetLoRAOptimizerContract(
        name="AdamW",
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        steps=1,
    ):
        raise ValueError("manifest.optimizer must match the reviewed one-step AdamW contract")
    return result


def _parse_messages(value: object) -> tuple[TargetLoRAMessage, ...]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("manifest.messages must contain one user and one assistant message")
    messages: list[TargetLoRAMessage] = []
    for index, item in enumerate(value):
        record = _object(item, f"manifest.messages[{index}]")
        _exact_fields(record, _MESSAGE_FIELDS, f"manifest.messages[{index}]")
        role = _string(record.get("role"), f"manifest.messages[{index}].role")
        content = _string(record.get("content"), f"manifest.messages[{index}].content")
        if len(content) > _MAX_MESSAGE_CHARACTERS:
            raise ValueError("manifest message exceeds character limit")
        messages.append(TargetLoRAMessage(role=role, content=content))
    if [item.role for item in messages] != ["user", "assistant"]:
        raise ValueError("manifest.messages roles must be user then assistant")
    return tuple(messages)


def _parameter_fingerprint(parameters: Any) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(parameters, key=lambda item: item[0]):
        tensor = parameter.detach().to(device="cpu").contiguous()
        descriptor = {
            "name": name,
            "shape": [int(value) for value in tensor.shape],
            "dtype": str(tensor.dtype),
        }
        payload = canonical_json_bytes(descriptor)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        digest.update(memoryview(tensor.numpy()).cast("B"))
        count += 1
    if count == 0:
        raise RuntimeError("parameter fingerprint received an empty parameter set")
    return "sha256:" + digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    metadata = canonical_json_bytes(
        {
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
    )
    digest.update(metadata)
    digest.update(memoryview(value.numpy()).cast("B"))
    return "sha256:" + digest.hexdigest()


def _maximum_error(left: Any, right: Any) -> float:
    return float((left - right).abs().max().item())


def _finite_scalar(value: Any, location: str) -> float:
    result = float(value.detach().to(device="cpu", dtype=None).item())
    if not math.isfinite(result):
        raise RuntimeError(f"{location} must be finite")
    return result


def _load_strict_json_file(path: Path, *, max_bytes: int) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON path must be a regular file: {path}")
    payload = path.read_bytes()
    if len(payload) > max_bytes:
        raise ValueError(f"JSON file exceeds byte limit: {path}")
    return _strict_json_object(payload, str(path))


def _strict_json_object(payload: bytes, location: str) -> dict[str, object]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{location} must be UTF-8") from error

    def pairs_hook(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{location} contains duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{location} contains non-finite number: {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{location} is not valid strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{location} root must be an object")
    return cast(dict[str, object], value)


def _hash_open_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, "sha256:" + digest.hexdigest()


def _canonical_sha256(value: Mapping[str, object]) -> str:
    return _sha256_bytes(canonical_json_bytes(dict(value)))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _exact_fields(
    value: Mapping[str, object], expected: set[str], location: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{location} fields mismatch: missing={missing}, extra={extra}")


def _object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return cast(dict[str, object], value)


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _digest(value: object, location: str) -> str:
    result = _string(value, location)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{location} must be lowercase SHA-256")
    return result


def _positive_integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return value


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    return result


def _iso_date(value: object, location: str) -> str:
    result = _string(value, location)
    try:
        date.fromisoformat(result)
    except ValueError as error:
        raise ValueError(f"{location} must be an ISO date") from error
    return result
