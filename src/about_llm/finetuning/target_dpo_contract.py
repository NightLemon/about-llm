"""Manifest schema and strict parsing for the target-checkpoint DPO control."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
)
from about_llm.llmops import artifact_fingerprint

TARGET_DPO_CONTROL_VERSION = "about-llm.target-dpo-control.v1"
TARGET_DPO_REPORT_VERSION = "about-llm.target-dpo-control-report.v1"
TARGET_DPO_EVIDENCE_BOUNDARY = (
    "This control rehashes one immutable-revision Qwen checkpoint snapshot and two "
    "authored preference/readiness artifacts, loads the verified local checkpoint with "
    "trust_remote_code disabled, normalizes Trainer-facing model/generation special-token "
    "config to tokenizer semantics before the baseline, and executes one CPU FP32 "
    "TRL 0.29.1 sigmoid-DPO "
    "optimizer step with a PEFT adapter-disabled frozen-base reference. It verifies "
    "exact target tokenization and completion masks, initial log(2), finite LoRA "
    "gradients, bitwise-unchanged frozen-base parameters and non-adapter state/config, "
    "adapter layers disabled during reference forwards, reported reference replay drift, "
    "and the one-batch chosen-relative "
    "margin after the step. It does not authenticate the model publisher, artifact "
    "author, labels, or trainer; eliminate verification-to-loader-reopen TOCTOU; "
    "establish human preference validity, annotator agreement, semantic deduplication, "
    "alignment quality, generalization, safety, convergence, or production readiness; "
    "execute QLoRA/CUDA/vLLM; export optimizer/RNG/resume state; or benchmark memory, "
    "throughput, latency, or cost."
)


@dataclass(frozen=True)
class TargetDPOArtifactSpec:
    filename: str
    size_bytes: int
    sha256: str
    manifest_fingerprint: str
    ordered_dataset_fingerprint: str | None = None
    record_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetDPOModelSpec:
    expected_model_class: str
    expected_model_type: str
    num_hidden_layers: int
    hidden_size: int
    base_parameter_count: int
    config_bos_token_id: int | None
    config_eos_token_id: int | None
    config_pad_token_id: int | None
    tokenizer_bos_token_id: int | None
    tokenizer_eos_token_id: int | None
    tokenizer_pad_token_id: int | None
    generation_config_bos_token_id: int | None
    generation_config_eos_token_ids: tuple[int, ...]
    generation_config_pad_token_id: int | None


@dataclass(frozen=True)
class TargetDPORuntimeSpec:
    python_implementation: str
    python_version: str
    platform: str
    torch_version: str
    transformers_version: str
    peft_version: str
    device: str
    dtype: str
    attention_implementation: str
    torch_num_threads: int


@dataclass(frozen=True)
class TargetDPOAdapterSpec:
    task_type: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: int
    dropout: float
    bias: str
    expected_trainable_parameters: int
    expected_trainable_tensor_count: int
    expected_a_tensor_count: int
    expected_b_tensor_count: int
    expected_b_element_count: int


@dataclass(frozen=True)
class TargetDPOTrainerSpec:
    trl_version: str
    loss_type: str
    beta: float
    max_length: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_steps: int
    gradient_checkpointing: bool
    optimizer: str
    reference_mode: str


@dataclass(frozen=True)
class TargetDPOPairSpec:
    record_id: str
    prompt_ids: tuple[int, ...]
    chosen_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]


@dataclass(frozen=True)
class TargetDPOControlSpec:
    checked_at: str
    model_id: str
    revision: str
    source_checkpoint_manifest_fingerprint: str
    source_checkpoint_report_fingerprint: str
    training_artifact: TargetDPOArtifactSpec
    readiness_artifact: TargetDPOArtifactSpec
    model_contract: TargetDPOModelSpec
    runtime: TargetDPORuntimeSpec
    adapter: TargetDPOAdapterSpec
    trainer: TargetDPOTrainerSpec
    torch_seed: int
    expected_pairs: tuple[TargetDPOPairSpec, ...]
    manifest_fingerprint: str


def _strict_json_object(path: Path, *, label: str = "control manifest") -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    raw = path.read_bytes()
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError(f"{label} exceeds the 5 MiB limit")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be strict UTF-8") from error
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} must be valid strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return cast(dict[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys mismatch")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return cast(int, value)


def _optional_integer(value: Any, label: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, label, minimum=minimum)


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _digest(value: Any, label: str) -> str:
    result = _string(value, label)
    if len(result) != 71 or not result.startswith("sha256:"):
        raise ValueError(f"{label} must be a sha256 digest")
    try:
        bytes.fromhex(result[7:])
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return result


def _token_ids(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    return tuple(_integer(item, label, minimum=0) for item in value)


def _artifact_spec(
    value: Any, label: str, *, training: bool
) -> TargetDPOArtifactSpec:
    item = _mapping(value, label)
    expected = {"filename", "size_bytes", "sha256", "manifest_fingerprint"}
    if training:
        expected |= {"ordered_dataset_fingerprint", "record_ids"}
    _exact_keys(item, expected, label)
    filename = _string(item["filename"], f"{label}.filename")
    if Path(filename).name != filename:
        raise ValueError(f"{label}.filename must be a basename")
    record_ids: tuple[str, ...] = ()
    ordered: str | None = None
    if training:
        values = item["record_ids"]
        if not isinstance(values, list) or not values:
            raise ValueError("training_artifact.record_ids must be a non-empty array")
        record_ids = tuple(_string(entry, "training record id") for entry in values)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("training record ids must be unique")
        ordered = _digest(
            item["ordered_dataset_fingerprint"],
            "training_artifact.ordered_dataset_fingerprint",
        )
    return TargetDPOArtifactSpec(
        filename=filename,
        size_bytes=_integer(item["size_bytes"], f"{label}.size_bytes"),
        sha256=_digest(item["sha256"], f"{label}.sha256"),
        manifest_fingerprint=_digest(
            item["manifest_fingerprint"], f"{label}.manifest_fingerprint"
        ),
        ordered_dataset_fingerprint=ordered,
        record_ids=record_ids,
    )


def load_target_dpo_control_spec(
    path: Path, *, checkpoint_spec: CheckpointControlSpec
) -> TargetDPOControlSpec:
    """Load the closed target-DPO manifest and bind it to the checkpoint spec."""

    manifest = _strict_json_object(path)
    _exact_keys(
        manifest,
        {
            "control_version",
            "checked_at",
            "model_id",
            "revision",
            "source_checkpoint_manifest_fingerprint",
            "source_checkpoint_report_fingerprint",
            "training_artifact",
            "readiness_artifact",
            "model_contract",
            "runtime",
            "adapter",
            "trainer",
            "torch_seed",
            "expected_pairs",
            "evidence_boundary",
        },
        "manifest",
    )
    if manifest["control_version"] != TARGET_DPO_CONTROL_VERSION:
        raise ValueError("target DPO control version mismatch")
    if manifest["evidence_boundary"] != TARGET_DPO_EVIDENCE_BOUNDARY:
        raise ValueError("target DPO evidence boundary mismatch")
    model_id = _string(manifest["model_id"], "manifest.model_id")
    revision = _string(manifest["revision"], "manifest.revision")
    checkpoint_fingerprint = _digest(
        manifest["source_checkpoint_manifest_fingerprint"],
        "manifest.source_checkpoint_manifest_fingerprint",
    )
    if (model_id, revision, checkpoint_fingerprint) != (
        checkpoint_spec.model_id,
        checkpoint_spec.revision,
        checkpoint_spec.manifest_fingerprint,
    ):
        raise ValueError("target DPO checkpoint binding mismatch")

    model = _mapping(manifest["model_contract"], "manifest.model_contract")
    _exact_keys(
        model,
        {
            "expected_model_class",
            "expected_model_type",
            "num_hidden_layers",
            "hidden_size",
            "base_parameter_count",
            "config_bos_token_id",
            "config_eos_token_id",
            "config_pad_token_id",
            "tokenizer_bos_token_id",
            "tokenizer_eos_token_id",
            "tokenizer_pad_token_id",
            "generation_config_bos_token_id",
            "generation_config_eos_token_ids",
            "generation_config_pad_token_id",
        },
        "manifest.model_contract",
    )
    model_spec = TargetDPOModelSpec(
        expected_model_class=_string(
            model["expected_model_class"], "model_contract.expected_model_class"
        ),
        expected_model_type=_string(
            model["expected_model_type"], "model_contract.expected_model_type"
        ),
        num_hidden_layers=_integer(
            model["num_hidden_layers"], "model_contract.num_hidden_layers"
        ),
        hidden_size=_integer(model["hidden_size"], "model_contract.hidden_size"),
        base_parameter_count=_integer(
            model["base_parameter_count"], "model_contract.base_parameter_count"
        ),
        config_bos_token_id=_optional_integer(
            model["config_bos_token_id"], "model_contract.config_bos_token_id"
        ),
        config_eos_token_id=_optional_integer(
            model["config_eos_token_id"], "model_contract.config_eos_token_id"
        ),
        config_pad_token_id=_optional_integer(
            model["config_pad_token_id"], "model_contract.config_pad_token_id"
        ),
        tokenizer_bos_token_id=_optional_integer(
            model["tokenizer_bos_token_id"],
            "model_contract.tokenizer_bos_token_id",
        ),
        tokenizer_eos_token_id=_optional_integer(
            model["tokenizer_eos_token_id"],
            "model_contract.tokenizer_eos_token_id",
        ),
        tokenizer_pad_token_id=_optional_integer(
            model["tokenizer_pad_token_id"],
            "model_contract.tokenizer_pad_token_id",
        ),
        generation_config_bos_token_id=_optional_integer(
            model["generation_config_bos_token_id"],
            "model_contract.generation_config_bos_token_id",
        ),
        generation_config_eos_token_ids=_token_ids(
            model["generation_config_eos_token_ids"],
            "model_contract.generation_config_eos_token_ids",
        ),
        generation_config_pad_token_id=_optional_integer(
            model["generation_config_pad_token_id"],
            "model_contract.generation_config_pad_token_id",
        ),
    )

    runtime = _mapping(manifest["runtime"], "manifest.runtime")
    _exact_keys(
        runtime,
        {
            "python_implementation",
            "python_version",
            "platform",
            "torch_version",
            "transformers_version",
            "peft_version",
            "device",
            "dtype",
            "attention_implementation",
            "torch_num_threads",
        },
        "manifest.runtime",
    )
    runtime_spec = TargetDPORuntimeSpec(
        python_implementation=_string(
            runtime["python_implementation"], "runtime.python_implementation"
        ),
        python_version=_string(runtime["python_version"], "runtime.python_version"),
        platform=_string(runtime["platform"], "runtime.platform"),
        torch_version=_string(runtime["torch_version"], "runtime.torch_version"),
        transformers_version=_string(
            runtime["transformers_version"], "runtime.transformers_version"
        ),
        peft_version=_string(runtime["peft_version"], "runtime.peft_version"),
        device=_string(runtime["device"], "runtime.device"),
        dtype=_string(runtime["dtype"], "runtime.dtype"),
        attention_implementation=_string(
            runtime["attention_implementation"], "runtime.attention_implementation"
        ),
        torch_num_threads=_integer(
            runtime["torch_num_threads"], "runtime.torch_num_threads"
        ),
    )
    if (
        runtime_spec.device,
        runtime_spec.dtype,
        runtime_spec.attention_implementation,
    ) != ("cpu", "float32", "eager"):
        raise ValueError("target DPO v1 requires CPU FP32 eager runtime")

    adapter = _mapping(manifest["adapter"], "manifest.adapter")
    _exact_keys(
        adapter,
        {
            "task_type",
            "target_modules",
            "rank",
            "alpha",
            "dropout",
            "bias",
            "expected_trainable_parameters",
            "expected_trainable_tensor_count",
            "expected_a_tensor_count",
            "expected_b_tensor_count",
            "expected_b_element_count",
        },
        "manifest.adapter",
    )
    modules = adapter["target_modules"]
    if not isinstance(modules, list) or not modules:
        raise ValueError("adapter.target_modules must be a non-empty array")
    target_modules = tuple(_string(item, "adapter target module") for item in modules)
    if len(target_modules) != len(set(target_modules)):
        raise ValueError("adapter target modules must be unique")
    adapter_spec = TargetDPOAdapterSpec(
        task_type=_string(adapter["task_type"], "adapter.task_type"),
        target_modules=target_modules,
        rank=_integer(adapter["rank"], "adapter.rank"),
        alpha=_integer(adapter["alpha"], "adapter.alpha"),
        dropout=_number(adapter["dropout"], "adapter.dropout"),
        bias=_string(adapter["bias"], "adapter.bias"),
        expected_trainable_parameters=_integer(
            adapter["expected_trainable_parameters"],
            "adapter.expected_trainable_parameters",
        ),
        expected_trainable_tensor_count=_integer(
            adapter["expected_trainable_tensor_count"],
            "adapter.expected_trainable_tensor_count",
        ),
        expected_a_tensor_count=_integer(
            adapter["expected_a_tensor_count"], "adapter.expected_a_tensor_count"
        ),
        expected_b_tensor_count=_integer(
            adapter["expected_b_tensor_count"], "adapter.expected_b_tensor_count"
        ),
        expected_b_element_count=_integer(
            adapter["expected_b_element_count"], "adapter.expected_b_element_count"
        ),
    )
    if adapter_spec.task_type != "CAUSAL_LM" or adapter_spec.bias != "none":
        raise ValueError("target DPO v1 requires CAUSAL_LM LoRA with no bias")

    trainer = _mapping(manifest["trainer"], "manifest.trainer")
    _exact_keys(
        trainer,
        {
            "trl_version",
            "loss_type",
            "beta",
            "max_length",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "max_steps",
            "gradient_checkpointing",
            "optimizer",
            "reference_mode",
        },
        "manifest.trainer",
    )
    if not isinstance(trainer["gradient_checkpointing"], bool):
        raise ValueError("trainer.gradient_checkpointing must be boolean")
    trainer_spec = TargetDPOTrainerSpec(
        trl_version=_string(trainer["trl_version"], "trainer.trl_version"),
        loss_type=_string(trainer["loss_type"], "trainer.loss_type"),
        beta=_number(trainer["beta"], "trainer.beta", positive=True),
        max_length=_integer(trainer["max_length"], "trainer.max_length"),
        batch_size=_integer(trainer["batch_size"], "trainer.batch_size"),
        gradient_accumulation_steps=_integer(
            trainer["gradient_accumulation_steps"],
            "trainer.gradient_accumulation_steps",
        ),
        learning_rate=_number(
            trainer["learning_rate"], "trainer.learning_rate", positive=True
        ),
        max_steps=_integer(trainer["max_steps"], "trainer.max_steps"),
        gradient_checkpointing=trainer["gradient_checkpointing"],
        optimizer=_string(trainer["optimizer"], "trainer.optimizer"),
        reference_mode=_string(
            trainer["reference_mode"], "trainer.reference_mode"
        ),
    )
    if (
        trainer_spec.loss_type,
        trainer_spec.max_steps,
        trainer_spec.reference_mode,
    ) != ("sigmoid", 1, "disable_current_peft_adapter"):
        raise ValueError("target DPO v1 trainer contract mismatch")

    raw_pairs = manifest["expected_pairs"]
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("expected_pairs must be a non-empty array")
    pair_specs: list[TargetDPOPairSpec] = []
    for index, raw_pair in enumerate(raw_pairs):
        pair = _mapping(raw_pair, f"expected_pairs[{index}]")
        _exact_keys(
            pair,
            {"record_id", "prompt_ids", "chosen_ids", "rejected_ids"},
            f"expected_pairs[{index}]",
        )

        pair_specs.append(
            TargetDPOPairSpec(
                record_id=_string(pair["record_id"], "pair.record_id"),
                prompt_ids=_token_ids(
                    pair["prompt_ids"], f"expected_pairs[{index}].prompt_ids"
                ),
                chosen_ids=_token_ids(
                    pair["chosen_ids"], f"expected_pairs[{index}].chosen_ids"
                ),
                rejected_ids=_token_ids(
                    pair["rejected_ids"], f"expected_pairs[{index}].rejected_ids"
                ),
            )
        )
    expected_pairs = tuple(pair_specs)
    training_spec = _artifact_spec(
        manifest["training_artifact"], "manifest.training_artifact", training=True
    )
    if tuple(item.record_id for item in expected_pairs) != training_spec.record_ids:
        raise ValueError("expected pair order differs from training artifact record order")
    if trainer_spec.batch_size != len(expected_pairs):
        raise ValueError("target DPO v1 requires one full-batch optimizer step")
    return TargetDPOControlSpec(
        checked_at=_string(manifest["checked_at"], "manifest.checked_at"),
        model_id=model_id,
        revision=revision,
        source_checkpoint_manifest_fingerprint=checkpoint_fingerprint,
        source_checkpoint_report_fingerprint=_digest(
            manifest["source_checkpoint_report_fingerprint"],
            "manifest.source_checkpoint_report_fingerprint",
        ),
        training_artifact=training_spec,
        readiness_artifact=_artifact_spec(
            manifest["readiness_artifact"],
            "manifest.readiness_artifact",
            training=False,
        ),
        model_contract=model_spec,
        runtime=runtime_spec,
        adapter=adapter_spec,
        trainer=trainer_spec,
        torch_seed=_integer(manifest["torch_seed"], "manifest.torch_seed", minimum=0),
        expected_pairs=expected_pairs,
        manifest_fingerprint="sha256:" + artifact_fingerprint(manifest),
    )
