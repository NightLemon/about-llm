"""Recorded target-Qwen SFT token, assistant-mask, and final-label control."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from about_llm.finetuning.data import SFTRecord, load_sft_records
from about_llm.finetuning.readiness import (
    load_sft_training_readiness,
    validate_sft_training_readiness,
)
from about_llm.finetuning.template import prepare_assistant_mask_features
from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    VerifiedCheckpointSnapshot,
    download_checkpoint_snapshot,
    verify_checkpoint_snapshot,
    verify_recorded_checkpoint_report,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

TARGET_SFT_LABEL_CONTROL_VERSION = "about-llm.target-sft-label-control.v2"
TARGET_SFT_LABEL_REPORT_VERSION = "about-llm.target-sft-label-control-report.v2"
TARGET_SFT_LABEL_EVIDENCE_BOUNDARY = (
    "This control rehashes one immutable-revision Qwen checkpoint snapshot, one authored "
    "tool-aware SFT train artifact, one held-out-free readiness artifact, and one reviewed "
    "local chat template. The checkpoint-native template is first shown to return an "
    "all-zero assistant mask for the fixed multi-turn, parallel-tool, and tool-preamble "
    "records. The reviewed template must render byte-token-identical input IDs while every "
    "generation span equals an independently authored per-record assistant serialization, "
    "including Qwen tool-call markup and message terminators. Tokenization and mask creation "
    "must happen in Python before Hugging Face Dataset construction so Arrow cannot widen "
    "heterogeneous tool argument objects with null fields. A real TRL 0.29.1 SFTTrainer "
    "collator must preserve exactly those tokens as labels, mask every other and padded "
    "token with -100, and feed the batch through the loaded CPU FP32 Qwen checkpoint for one "
    "finite no-grad forward loss. It does not authenticate the publisher, data author, "
    "readiness issuer, template reviewer, or runtime; eliminate verification-to-loader-"
    "reopen TOCTOU; prove arbitrary provider tool schemas, multimodal messages, records "
    "outside the fixed fixture, tool execution correctness, or tool-result truth; establish "
    "data legality, semantic quality, generalization, convergence, or safety; execute "
    "backward, optimizer, adapter export, QLoRA, CUDA, vLLM, or serving; or benchmark memory, "
    "throughput, latency, or cost."
)

_MAX_JSON_BYTES = 1_000_000
_LABEL_PAD_TOKEN_ID = -100


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) > _MAX_JSON_BYTES:
        raise ValueError(f"JSON artifact exceeds {_MAX_JSON_BYTES} bytes")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return cast(dict[str, Any], value)


def _closed_object(
    value: Any, *, field: str, required: set[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    typed = cast(dict[str, Any], value)
    missing = sorted(required - set(typed))
    unknown = sorted(set(typed) - required)
    if missing:
        raise ValueError(f"{field} is missing keys: {missing}")
    if unknown:
        raise ValueError(f"{field} has unknown keys: {unknown}")
    return typed


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return cast(int, value)


def _signed_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return cast(int, value)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _integer_sequence(value: Any, field: str, *, binary: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty integer array")
    items: list[int] = []
    for index, item in enumerate(value):
        number = _integer(item, f"{field}[{index}]")
        if binary and number not in (0, 1):
            raise ValueError(f"{field}[{index}] must be 0 or 1")
        items.append(number)
    return tuple(items)


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _has_generation_marker(template: str) -> bool:
    return "{% generation" in template or "{%- generation" in template


@dataclass(frozen=True)
class TargetSFTArtifactSpec:
    filename: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError("artifact filename must be one safe basename")
        if self.size_bytes <= 0:
            raise ValueError("artifact size_bytes must be positive")
        if len(self.sha256) != 71 or not self.sha256.startswith("sha256:"):
            raise ValueError("artifact sha256 must be one prefixed SHA-256 digest")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TargetSFTSampleSpec:
    record_id: str
    input_ids: tuple[int, ...]
    assistant_masks: tuple[int, ...]
    assistant_generation_text: str

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("sample record_id must not be empty")
        if not self.input_ids or len(self.input_ids) != len(self.assistant_masks):
            raise ValueError("sample input_ids and assistant_masks must be non-empty and aligned")
        if any(token < 0 for token in self.input_ids):
            raise ValueError("sample input_ids must be non-negative")
        if any(mask not in (0, 1) for mask in self.assistant_masks):
            raise ValueError("sample assistant_masks must be binary")
        if not any(self.assistant_masks) or all(self.assistant_masks):
            raise ValueError("sample must exercise both supervised and ignored tokens")
        if not isinstance(self.assistant_generation_text, str) or not (
            self.assistant_generation_text
        ):
            raise ValueError("sample assistant_generation_text must not be empty")
        try:
            self.assistant_generation_text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(
                "sample assistant_generation_text contains an unpaired Unicode surrogate"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "input_ids": list(self.input_ids),
            "assistant_masks": list(self.assistant_masks),
            "assistant_generation_text": self.assistant_generation_text,
        }


@dataclass(frozen=True)
class TargetSFTRuntimeSpec:
    python_version: str
    platform: str
    torch_version: str
    transformers_version: str
    trl_version: str
    device: str
    dtype: str
    attention_implementation: str
    torch_num_threads: int

    def __post_init__(self) -> None:
        values = (
            self.python_version,
            self.platform,
            self.torch_version,
            self.transformers_version,
            self.trl_version,
            self.device,
            self.dtype,
            self.attention_implementation,
        )
        if any(not value for value in values):
            raise ValueError("runtime string fields must not be empty")
        if self.device != "cpu" or self.dtype != "float32":
            raise ValueError("target SFT label v2 requires CPU FP32")
        if self.attention_implementation != "eager":
            raise ValueError("target SFT label v2 requires eager attention")
        if self.torch_num_threads <= 0:
            raise ValueError("torch_num_threads must be positive")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "trl_version": self.trl_version,
            "device": self.device,
            "dtype": self.dtype,
            "attention_implementation": self.attention_implementation,
            "torch_num_threads": self.torch_num_threads,
        }


@dataclass(frozen=True)
class TargetSFTLabelControlSpec:
    checked_at: str
    model_id: str
    revision: str
    source_checkpoint_manifest_fingerprint: str
    source_checkpoint_report_fingerprint: str
    expected_model_class: str
    expected_model_type: str
    expected_base_parameter_count: int
    training_artifact: TargetSFTArtifactSpec
    readiness_artifact: TargetSFTArtifactSpec
    template_artifact: TargetSFTArtifactSpec
    native_chat_template_sha256: str
    assistant_terminator: str
    max_length: int
    batch_size: int
    pad_token_id: int
    label_pad_token_id: int
    expected_batch_shape: tuple[int, int]
    expected_forward_loss: float
    forward_loss_atol: float
    runtime: TargetSFTRuntimeSpec
    samples: tuple[TargetSFTSampleSpec, ...]
    evidence_boundary: str

    def __post_init__(self) -> None:
        if (
            not self.checked_at
            or not self.model_id
            or not self.revision
            or not self.assistant_terminator
        ):
            raise ValueError("checked_at, model_id, and revision must not be empty")
        if not self.expected_model_class or not self.expected_model_type:
            raise ValueError("model contract strings must not be empty")
        if self.expected_base_parameter_count <= 0:
            raise ValueError("expected_base_parameter_count must be positive")
        if self.max_length <= 0 or self.batch_size <= 0:
            raise ValueError("max_length and batch_size must be positive")
        if self.pad_token_id < 0 or self.label_pad_token_id != _LABEL_PAD_TOKEN_ID:
            raise ValueError("reviewed pad-token contract mismatch")
        reviewed_shape = (
            self.batch_size,
            max(len(sample.input_ids) for sample in self.samples),
        )
        if self.expected_batch_shape != reviewed_shape:
            raise ValueError("expected_batch_shape does not match the reviewed samples")
        if len(self.samples) != self.batch_size:
            raise ValueError("sample count must equal batch_size")
        if len({sample.record_id for sample in self.samples}) != len(self.samples):
            raise ValueError("sample record IDs must be unique")
        if any(
            self.assistant_terminator not in sample.assistant_generation_text
            for sample in self.samples
        ):
            raise ValueError("every assistant generation projection must contain the terminator")
        if self.expected_forward_loss <= 0 or self.forward_loss_atol <= 0:
            raise ValueError("forward loss and tolerance must be positive")
        if self.evidence_boundary != TARGET_SFT_LABEL_EVIDENCE_BOUNDARY:
            raise ValueError("target SFT label evidence boundary drift")
        for digest in (
            self.source_checkpoint_manifest_fingerprint,
            self.source_checkpoint_report_fingerprint,
            self.native_chat_template_sha256,
        ):
            if len(digest) != 71 or not digest.startswith("sha256:"):
                raise ValueError("reviewed fingerprints must be prefixed SHA-256 digests")

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_version": TARGET_SFT_LABEL_CONTROL_VERSION,
            "checked_at": self.checked_at,
            "model_id": self.model_id,
            "revision": self.revision,
            "source_checkpoint_manifest_fingerprint": (
                self.source_checkpoint_manifest_fingerprint
            ),
            "source_checkpoint_report_fingerprint": self.source_checkpoint_report_fingerprint,
            "model_contract": {
                "expected_model_class": self.expected_model_class,
                "expected_model_type": self.expected_model_type,
                "expected_base_parameter_count": self.expected_base_parameter_count,
            },
            "artifacts": {
                "training": self.training_artifact.to_dict(),
                "readiness": self.readiness_artifact.to_dict(),
                "reviewed_template": self.template_artifact.to_dict(),
                "native_chat_template_sha256": self.native_chat_template_sha256,
                "assistant_terminator": self.assistant_terminator,
            },
            "collator_contract": {
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "padding_side": "right",
                "pad_token_id": self.pad_token_id,
                "label_pad_token_id": self.label_pad_token_id,
                "expected_batch_shape": list(self.expected_batch_shape),
            },
            "recorded_forward_contract": {
                "expected_loss": self.expected_forward_loss,
                "absolute_tolerance": self.forward_loss_atol,
            },
            "runtime": self.runtime.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
            "evidence_boundary": self.evidence_boundary,
        }


def _parse_artifact(value: Any, field: str) -> TargetSFTArtifactSpec:
    obj = _closed_object(value, field=field, required={"filename", "size_bytes", "sha256"})
    return TargetSFTArtifactSpec(
        filename=_string(obj["filename"], f"{field}.filename"),
        size_bytes=_integer(obj["size_bytes"], f"{field}.size_bytes", minimum=1),
        sha256=_string(obj["sha256"], f"{field}.sha256"),
    )


def load_target_sft_label_control_spec(
    path: Path, *, checkpoint_spec: CheckpointControlSpec | None = None
) -> TargetSFTLabelControlSpec:
    root = _closed_object(
        _load_json_object(path),
        field="control",
        required={
            "control_version",
            "checked_at",
            "model_id",
            "revision",
            "source_checkpoint_manifest_fingerprint",
            "source_checkpoint_report_fingerprint",
            "model_contract",
            "artifacts",
            "collator_contract",
            "recorded_forward_contract",
            "runtime",
            "samples",
            "evidence_boundary",
        },
    )
    if root["control_version"] != TARGET_SFT_LABEL_CONTROL_VERSION:
        raise ValueError("unsupported target SFT label control version")
    model = _closed_object(
        root["model_contract"],
        field="control.model_contract",
        required={
            "expected_model_class",
            "expected_model_type",
            "expected_base_parameter_count",
        },
    )
    artifacts = _closed_object(
        root["artifacts"],
        field="control.artifacts",
        required={
            "training",
            "readiness",
            "reviewed_template",
            "native_chat_template_sha256",
            "assistant_terminator",
        },
    )
    collator = _closed_object(
        root["collator_contract"],
        field="control.collator_contract",
        required={
            "max_length",
            "batch_size",
            "padding_side",
            "pad_token_id",
            "label_pad_token_id",
            "expected_batch_shape",
        },
    )
    if collator["padding_side"] != "right":
        raise ValueError("only reviewed right padding is supported")
    shape = _integer_sequence(
        collator["expected_batch_shape"], "control.collator_contract.expected_batch_shape"
    )
    if len(shape) != 2:
        raise ValueError("expected_batch_shape must have two dimensions")
    forward = _closed_object(
        root["recorded_forward_contract"],
        field="control.recorded_forward_contract",
        required={"expected_loss", "absolute_tolerance"},
    )
    runtime_obj = _closed_object(
        root["runtime"],
        field="control.runtime",
        required={
            "python_version",
            "platform",
            "torch_version",
            "transformers_version",
            "trl_version",
            "device",
            "dtype",
            "attention_implementation",
            "torch_num_threads",
        },
    )
    raw_samples = root["samples"]
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("control.samples must be a non-empty array")
    samples: list[TargetSFTSampleSpec] = []
    for index, value in enumerate(raw_samples):
        sample = _closed_object(
            value,
            field=f"control.samples[{index}]",
            required={
                "record_id",
                "input_ids",
                "assistant_masks",
                "assistant_generation_text",
            },
        )
        samples.append(
            TargetSFTSampleSpec(
                record_id=_string(sample["record_id"], f"control.samples[{index}].record_id"),
                input_ids=_integer_sequence(
                    sample["input_ids"], f"control.samples[{index}].input_ids"
                ),
                assistant_masks=_integer_sequence(
                    sample["assistant_masks"],
                    f"control.samples[{index}].assistant_masks",
                    binary=True,
                ),
                assistant_generation_text=_string(
                    sample["assistant_generation_text"],
                    f"control.samples[{index}].assistant_generation_text",
                ),
            )
        )
    spec = TargetSFTLabelControlSpec(
        checked_at=_string(root["checked_at"], "control.checked_at"),
        model_id=_string(root["model_id"], "control.model_id"),
        revision=_string(root["revision"], "control.revision"),
        source_checkpoint_manifest_fingerprint=_string(
            root["source_checkpoint_manifest_fingerprint"],
            "control.source_checkpoint_manifest_fingerprint",
        ),
        source_checkpoint_report_fingerprint=_string(
            root["source_checkpoint_report_fingerprint"],
            "control.source_checkpoint_report_fingerprint",
        ),
        expected_model_class=_string(
            model["expected_model_class"], "control.model_contract.expected_model_class"
        ),
        expected_model_type=_string(
            model["expected_model_type"], "control.model_contract.expected_model_type"
        ),
        expected_base_parameter_count=_integer(
            model["expected_base_parameter_count"],
            "control.model_contract.expected_base_parameter_count",
            minimum=1,
        ),
        training_artifact=_parse_artifact(artifacts["training"], "control.artifacts.training"),
        readiness_artifact=_parse_artifact(
            artifacts["readiness"], "control.artifacts.readiness"
        ),
        template_artifact=_parse_artifact(
            artifacts["reviewed_template"], "control.artifacts.reviewed_template"
        ),
        native_chat_template_sha256=_string(
            artifacts["native_chat_template_sha256"],
            "control.artifacts.native_chat_template_sha256",
        ),
        assistant_terminator=_string(
            artifacts["assistant_terminator"],
            "control.artifacts.assistant_terminator",
        ),
        max_length=_integer(
            collator["max_length"], "control.collator_contract.max_length", minimum=1
        ),
        batch_size=_integer(
            collator["batch_size"], "control.collator_contract.batch_size", minimum=1
        ),
        pad_token_id=_integer(collator["pad_token_id"], "control.collator_contract.pad_token_id"),
        label_pad_token_id=_signed_integer(
            collator["label_pad_token_id"],
            "control.collator_contract.label_pad_token_id",
        ),
        expected_batch_shape=(shape[0], shape[1]),
        expected_forward_loss=_finite_number(
            forward["expected_loss"], "control.recorded_forward_contract.expected_loss"
        ),
        forward_loss_atol=_finite_number(
            forward["absolute_tolerance"],
            "control.recorded_forward_contract.absolute_tolerance",
        ),
        runtime=TargetSFTRuntimeSpec(
            python_version=_string(runtime_obj["python_version"], "control.runtime.python_version"),
            platform=_string(runtime_obj["platform"], "control.runtime.platform"),
            torch_version=_string(runtime_obj["torch_version"], "control.runtime.torch_version"),
            transformers_version=_string(
                runtime_obj["transformers_version"], "control.runtime.transformers_version"
            ),
            trl_version=_string(runtime_obj["trl_version"], "control.runtime.trl_version"),
            device=_string(runtime_obj["device"], "control.runtime.device"),
            dtype=_string(runtime_obj["dtype"], "control.runtime.dtype"),
            attention_implementation=_string(
                runtime_obj["attention_implementation"],
                "control.runtime.attention_implementation",
            ),
            torch_num_threads=_integer(
                runtime_obj["torch_num_threads"],
                "control.runtime.torch_num_threads",
                minimum=1,
            ),
        ),
        samples=tuple(samples),
        evidence_boundary=_string(root["evidence_boundary"], "control.evidence_boundary"),
    )
    if checkpoint_spec is not None:
        if spec.model_id != checkpoint_spec.model_id or spec.revision != checkpoint_spec.revision:
            raise ValueError("target SFT label control does not match checkpoint identity")
        if (
            spec.source_checkpoint_manifest_fingerprint
            != checkpoint_spec.manifest_fingerprint
        ):
            raise ValueError("target SFT label control checkpoint fingerprint mismatch")
    return spec


def _verify_artifact(path: Path, artifact: TargetSFTArtifactSpec) -> dict[str, Any]:
    if path.name != artifact.filename:
        raise ValueError(f"artifact filename mismatch: {path.name!r}")
    if not path.is_file():
        raise ValueError(f"artifact does not exist: {path}")
    size = path.stat().st_size
    digest = _sha256_path(path)
    if size != artifact.size_bytes or digest != artifact.sha256:
        raise ValueError(f"artifact byte identity mismatch: {artifact.filename}")
    return {**artifact.to_dict(), "verified": True}


def _current_runtime(torch_module: Any) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch_module.__version__),
        "transformers_version": version("transformers"),
        "trl_version": version("trl"),
        "device": "cpu",
        "dtype": "float32",
        "attention_implementation": "eager",
        "torch_num_threads": int(torch_module.get_num_threads()),
    }


def _messages(record: SFTRecord) -> list[dict[str, Any]]:
    return [message.to_dict() for message in record.messages]


def _tools(record: SFTRecord) -> list[dict[str, Any]] | None:
    return [tool.to_dict() for tool in record.tools] or None


def _fixture_capabilities(records: Sequence[SFTRecord]) -> dict[str, Any]:
    roles = {message.role.value for record in records for message in record.messages}
    role_order = ("system", "user", "assistant", "tool")
    return {
        "supported_roles": [role for role in role_order if role in roles],
        "tool_calls_supported_by_evidence": any(record.tool_call_count for record in records),
        "multi_assistant_turns_supported_by_evidence": any(
            sum(message.role.value == "assistant" for message in record.messages) > 1
            for record in records
        ),
    }


def execute_loaded_target_sft_label_control(
    spec: TargetSFTLabelControlSpec,
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[SFTRecord],
    reviewed_template: str,
    torch_module: Any,
) -> dict[str, Any]:
    """Execute the reviewed tokenizer → TRL collator → target forward path."""

    if type(model).__name__ != spec.expected_model_class:
        raise ValueError("loaded model class does not match the control")
    if getattr(model.config, "model_type", None) != spec.expected_model_type:
        raise ValueError("loaded model_type does not match the control")
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if parameter_count != spec.expected_base_parameter_count:
        raise ValueError("loaded model parameter count does not match the control")
    snapshot = tuple(records)
    if tuple(record.record_id for record in snapshot) != tuple(
        sample.record_id for sample in spec.samples
    ):
        raise ValueError("ordered SFT record IDs do not match the control")
    native_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(native_template, str) or not native_template:
        raise ValueError("checkpoint tokenizer has no native chat template")
    if _sha256_bytes(native_template.encode("utf-8")) != spec.native_chat_template_sha256:
        raise ValueError("native chat template identity mismatch")
    if not reviewed_template.strip() or not _has_generation_marker(reviewed_template):
        raise ValueError("reviewed template must contain a generation span")
    if getattr(tokenizer, "pad_token_id", None) != spec.pad_token_id:
        raise ValueError("tokenizer pad_token_id does not match the control")

    sample_results: list[dict[str, Any]] = []
    for record, expected in zip(snapshot, spec.samples, strict=True):
        messages = _messages(record)
        tools = _tools(record)
        native = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            chat_template=native_template,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        native_ids = tuple(int(token) for token in native["input_ids"])
        native_masks = tuple(int(mask) for mask in native["assistant_masks"])
        if len(native_masks) != len(native_ids) or any(native_masks):
            raise RuntimeError(
                "checkpoint-native template no longer has the reviewed all-zero mask"
            )
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            chat_template=reviewed_template,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        input_ids = tuple(int(token) for token in rendered["input_ids"])
        assistant_masks = tuple(int(mask) for mask in rendered["assistant_masks"])
        if native_ids != input_ids:
            raise RuntimeError("reviewed template changed native input token IDs")
        if input_ids != expected.input_ids or assistant_masks != expected.assistant_masks:
            raise RuntimeError("target token or assistant-mask contract drift")
        if len(input_ids) > spec.max_length:
            raise RuntimeError("reviewed sample would be silently truncated")
        expected_assistant_ids = tuple(
            int(token)
            for token in tokenizer.encode(
                expected.assistant_generation_text,
                add_special_tokens=False,
            )
        )
        selected_ids = tuple(
            token for token, mask in zip(input_ids, assistant_masks, strict=True) if mask
        )
        if selected_ids != expected_assistant_ids:
            raise RuntimeError(
                "generation span does not equal the authored assistant serialization"
            )
        labels = tuple(
            token if mask else spec.label_pad_token_id
            for token, mask in zip(input_ids, assistant_masks, strict=True)
        )
        sample_results.append(
            {
                "record_id": record.record_id,
                "native_input_ids_equal": True,
                "assistant_generation_text_equal": True,
                "input_ids_canonical_sha256": _canonical_sha256(list(input_ids)),
                "assistant_masks_canonical_sha256": _canonical_sha256(
                    list(assistant_masks)
                ),
                "assistant_token_ids_canonical_sha256": _canonical_sha256(
                    list(selected_ids)
                ),
                "unpadded_labels_canonical_sha256": _canonical_sha256(list(labels)),
                "input_token_count": len(input_ids),
                "assistant_token_count": len(selected_ids),
            }
        )

    from datasets import Dataset, disable_progress_bars  # type: ignore[import-untyped]
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    tokenizer.chat_template = reviewed_template
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    torch_module.set_num_threads(spec.runtime.torch_num_threads)
    torch_module.manual_seed(20260813)
    disable_progress_bars()
    mask_preparation = prepare_assistant_mask_features(
        snapshot,
        render=lambda messages, tools: tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        ),
        renderer_identity={
            "model_id": spec.model_id,
            "revision": spec.revision,
            "native_chat_template_sha256": spec.native_chat_template_sha256,
            "reviewed_template_sha256": spec.template_artifact.sha256,
        },
        max_length=spec.max_length,
    )
    for prepared_feature, expected in zip(
        mask_preparation.features, spec.samples, strict=True
    ):
        if (
            prepared_feature.record_id != expected.record_id
            or prepared_feature.input_ids != expected.input_ids
            or prepared_feature.assistant_masks != expected.assistant_masks
        ):
            raise RuntimeError("pre-Arrow tokenized feature contract drift")
    with tempfile.TemporaryDirectory(prefix="about-llm-target-sft-label-") as directory:
        config = SFTConfig(
            output_dir=directory,
            max_length=spec.max_length,
            # TRL 0.29.1 rejects this preprocessing flag for an already-tokenized
            # dataset; its collator still consumes the materialized assistant_masks.
            assistant_only_loss=False,
            completion_only_loss=False,
            per_device_train_batch_size=spec.batch_size,
            max_steps=1,
            gradient_checkpointing=False,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            use_cpu=True,
            dataloader_pin_memory=False,
            optim="adamw_torch",
            seed=20260813,
            data_seed=20260813,
        )
        trainer = SFTTrainer(
            model=model,
            args=config,
            train_dataset=Dataset.from_list(mask_preparation.to_training_rows()),
            processing_class=tokenizer,
        )
        prepared: Any = trainer.train_dataset
        if prepared is None or len(prepared) != len(snapshot):
            raise RuntimeError("TRL did not preserve the reviewed training rows")
        features = [dict(prepared[index]) for index in range(len(prepared))]
        for feature, expected in zip(features, spec.samples, strict=True):
            if tuple(feature["input_ids"]) != expected.input_ids:
                raise RuntimeError("TRL prepared input_ids drifted from the control")
            if tuple(feature["assistant_masks"]) != expected.assistant_masks:
                raise RuntimeError("TRL prepared assistant_masks drifted from the control")
        batch = trainer.data_collator(features)
        input_rows = cast(list[list[int]], batch["input_ids"].detach().cpu().tolist())
        attention_rows = cast(
            list[list[int]], batch["attention_mask"].detach().cpu().tolist()
        )
        label_rows = cast(list[list[int]], batch["labels"].detach().cpu().tolist())
        shape = (len(input_rows), len(input_rows[0]))
        if shape != spec.expected_batch_shape:
            raise RuntimeError(f"TRL collated shape drifted: {shape}")
        for row_index, expected in enumerate(spec.samples):
            for column in range(shape[1]):
                if column < len(expected.input_ids):
                    expected_input = expected.input_ids[column]
                    expected_attention = 1
                    expected_label = (
                        expected_input
                        if expected.assistant_masks[column]
                        else spec.label_pad_token_id
                    )
                else:
                    expected_input = spec.pad_token_id
                    expected_attention = 0
                    expected_label = spec.label_pad_token_id
                if (
                    input_rows[row_index][column] != expected_input
                    or attention_rows[row_index][column] != expected_attention
                    or label_rows[row_index][column] != expected_label
                ):
                    raise RuntimeError("TRL final collator label contract drift")
        model.eval()
        with torch_module.no_grad():
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
        forward_loss = float(output.loss.detach().cpu().item())
    if not math.isfinite(forward_loss):
        raise RuntimeError("target SFT forward loss is non-finite")

    supervised = sum(label != spec.label_pad_token_id for row in label_rows for label in row)
    ignored = sum(label == spec.label_pad_token_id for row in label_rows for label in row)
    attention = sum(value for row in attention_rows for value in row)
    padding = sum(value == 0 for row in attention_rows for value in row)
    capabilities = _fixture_capabilities(snapshot)
    return {
        "model": {
            "class": type(model).__name__,
            "model_type": str(model.config.model_type),
            "base_parameter_count": parameter_count,
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "target_checkpoint_weights_loaded": True,
        },
        "template": {
            "checkpoint_native_generation_marker_present": _has_generation_marker(
                native_template
            ),
            "checkpoint_native_all_zero_assistant_mask_observed": True,
            "reviewed_generation_marker_present": True,
            "reviewed_render_matches_native_input_ids": True,
            "reviewed_mask_matches_authored_assistant_generation_text": True,
            **capabilities,
            "record_tools_forwarded_to_chat_template": True,
            "arrow_pre_tokenization_executed": True,
            "raw_nested_records_passed_to_arrow": False,
        },
        "samples": sample_results,
        "collator": {
            "batch_shape": list(shape),
            "input_ids_canonical_sha256": _canonical_sha256(input_rows),
            "attention_mask_canonical_sha256": _canonical_sha256(attention_rows),
            "labels_canonical_sha256": _canonical_sha256(label_rows),
            "attention_token_count": attention,
            "padding_token_count": padding,
            "supervised_label_count": supervised,
            "ignored_label_count": ignored,
            "non_assistant_and_padding_labels_are_minus_100": True,
            "assistant_labels_equal_input_ids": True,
            "silent_truncation_observed": False,
        },
        "execution": {
            "target_forward_executed": True,
            "backward_executed": False,
            "optimizer_step_count": 0,
            "forward_loss": forward_loss,
            "forward_loss_finite": True,
        },
    }


def _source_projection(
    spec: TargetSFTLabelControlSpec,
    checkpoint_spec: CheckpointControlSpec,
    snapshot: VerifiedCheckpointSnapshot,
) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "checkpoint_manifest_fingerprint": checkpoint_spec.manifest_fingerprint,
        "checkpoint_report_fingerprint": spec.source_checkpoint_report_fingerprint,
        "loader_input": "verified_local_snapshot_directory",
        "all_selected_checkpoint_file_bytes_verified_before_load": True,
        "selected_checkpoint_total_bytes": sum(
            cast(int, item["size_bytes"]) for item in snapshot.files
        ),
        "selected_checkpoint_files": [dict(item) for item in snapshot.files],
    }


def _scope_projection(records: Sequence[SFTRecord]) -> dict[str, bool]:
    capabilities = _fixture_capabilities(records)
    return {
        "target_checkpoint_weights_loaded": True,
        "target_tokenizer_and_native_template_executed": True,
        "reviewed_template_exact_fixed_subset_executed": True,
        "fixed_tool_calls_executed": cast(
            bool, capabilities["tool_calls_supported_by_evidence"]
        ),
        "fixed_multi_assistant_turns_executed": cast(
            bool, capabilities["multi_assistant_turns_supported_by_evidence"]
        ),
        "pre_arrow_tokenization_executed": True,
        "real_trl_sft_collator_executed": True,
        "target_forward_loss_executed": True,
        "backward_or_optimizer_executed": False,
        "arbitrary_provider_tool_schemas_or_multimodal_proven": False,
        "tool_execution_or_result_truth_proven": False,
        "data_legality_or_semantic_quality_proven": False,
        "convergence_generalization_or_safety_proven": False,
        "qlora_cuda_or_vllm_executed": False,
        "performance_or_memory_benchmark_performed": False,
        "publisher_data_template_or_runtime_authenticated": False,
        "verification_to_loader_reopen_toctou_eliminated": False,
        "production_readiness_proven": False,
    }


def run_target_sft_label_control(
    spec: TargetSFTLabelControlSpec,
    *,
    checkpoint_spec: CheckpointControlSpec,
    checkpoint_report_path: Path,
    training_path: Path,
    readiness_path: Path,
    template_path: Path,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load the reviewed checkpoint and produce one target SFT label report."""

    checkpoint_report = verify_recorded_checkpoint_report(
        checkpoint_report_path,
        expected_manifest_fingerprint=checkpoint_spec.manifest_fingerprint,
    )
    if checkpoint_report["report_fingerprint"] != spec.source_checkpoint_report_fingerprint:
        raise ValueError("checkpoint report fingerprint does not match the SFT control")
    training_artifact = _verify_artifact(training_path, spec.training_artifact)
    readiness_artifact = _verify_artifact(readiness_path, spec.readiness_artifact)
    template_artifact = _verify_artifact(template_path, spec.template_artifact)
    records = load_sft_records(training_path)
    readiness = load_sft_training_readiness(readiness_path)
    data_audit = validate_sft_training_readiness(records, readiness)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(spec.runtime.torch_num_threads)
    runtime = _current_runtime(torch)
    if runtime != spec.runtime.to_dict():
        raise RuntimeError(f"runtime does not match the reviewed control: {runtime}")
    snapshot_directory = download_checkpoint_snapshot(
        checkpoint_spec, local_files_only=local_files_only
    )
    snapshot = verify_checkpoint_snapshot(checkpoint_spec, snapshot_directory)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        snapshot.directory,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot.directory,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation=spec.runtime.attention_implementation,
    )
    result = execute_loaded_target_sft_label_control(
        spec,
        model=model,
        tokenizer=tokenizer,
        records=records,
        reviewed_template=template_path.read_text(encoding="utf-8"),
        torch_module=torch,
    )
    observed_loss = cast(float, cast(dict[str, Any], result["execution"])["forward_loss"])
    if not math.isclose(
        observed_loss,
        spec.expected_forward_loss,
        rel_tol=0.0,
        abs_tol=spec.forward_loss_atol,
    ):
        raise RuntimeError(
            f"target SFT forward loss drifted: expected {spec.expected_forward_loss}, "
            f"observed {observed_loss}"
        )
    payload: dict[str, Any] = {
        "report_version": TARGET_SFT_LABEL_REPORT_VERSION,
        "checked_at": spec.checked_at,
        "control_manifest_fingerprint": spec.manifest_fingerprint,
        "evidence_boundary": spec.evidence_boundary,
        "source": _source_projection(spec, checkpoint_spec, snapshot),
        "data": {
            "training_artifact": training_artifact,
            "readiness_artifact": readiness_artifact,
            "training_manifest_fingerprint": data_audit.manifest_fingerprint,
            "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
            "record_ids": [record.record_id for record in records],
            "authored_fixture_records": True,
            "held_out_plaintext_embedded": False,
        },
        "template_artifact": template_artifact,
        "runtime": runtime,
        "result": result,
        "scope": _scope_projection(records),
    }
    payload["report_fingerprint"] = "sha256:" + artifact_fingerprint(payload)
    return payload


def _expected_batch(
    spec: TargetSFTLabelControlSpec,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    width = spec.expected_batch_shape[1]
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for sample in spec.samples:
        padding = width - len(sample.input_ids)
        input_rows.append([*sample.input_ids, *([spec.pad_token_id] * padding)])
        attention_rows.append([*([1] * len(sample.input_ids)), *([0] * padding)])
        label_rows.append(
            [
                token if mask else spec.label_pad_token_id
                for token, mask in zip(sample.input_ids, sample.assistant_masks, strict=True)
            ]
            + [spec.label_pad_token_id] * padding
        )
    return input_rows, attention_rows, label_rows


def verify_recorded_target_sft_label_report(
    path: Path,
    *,
    spec: TargetSFTLabelControlSpec,
    checkpoint_spec: CheckpointControlSpec,
    checkpoint_report_path: Path,
    training_path: Path,
    readiness_path: Path,
    template_path: Path,
) -> dict[str, Any]:
    """Verify a recorded report without loading tokenizer or model weights."""

    report = _closed_object(
        _load_json_object(path),
        field="report",
        required={
            "report_version",
            "checked_at",
            "control_manifest_fingerprint",
            "evidence_boundary",
            "source",
            "data",
            "template_artifact",
            "runtime",
            "result",
            "scope",
            "report_fingerprint",
        },
    )
    if report["report_version"] != TARGET_SFT_LABEL_REPORT_VERSION:
        raise ValueError("unsupported target SFT label report version")
    if report["checked_at"] != spec.checked_at:
        raise ValueError("report checked_at does not match the control")
    if report["control_manifest_fingerprint"] != spec.manifest_fingerprint:
        raise ValueError("report control fingerprint mismatch")
    if report["evidence_boundary"] != spec.evidence_boundary:
        raise ValueError("report evidence boundary drift")
    claimed_fingerprint = _string(report["report_fingerprint"], "report.report_fingerprint")
    fingerprint_payload = dict(report)
    del fingerprint_payload["report_fingerprint"]
    expected_fingerprint = "sha256:" + artifact_fingerprint(fingerprint_payload)
    if claimed_fingerprint != expected_fingerprint:
        raise ValueError("report fingerprint mismatch")

    checkpoint_report = verify_recorded_checkpoint_report(
        checkpoint_report_path,
        expected_manifest_fingerprint=checkpoint_spec.manifest_fingerprint,
    )
    if checkpoint_report["report_fingerprint"] != spec.source_checkpoint_report_fingerprint:
        raise ValueError("checkpoint report fingerprint mismatch")
    training_artifact = _verify_artifact(training_path, spec.training_artifact)
    readiness_artifact = _verify_artifact(readiness_path, spec.readiness_artifact)
    template_artifact = _verify_artifact(template_path, spec.template_artifact)
    records = load_sft_records(training_path)
    readiness = load_sft_training_readiness(readiness_path)
    data_audit = validate_sft_training_readiness(records, readiness)
    if report["template_artifact"] != template_artifact:
        raise ValueError("report template artifact projection mismatch")
    if report["runtime"] != spec.runtime.to_dict():
        raise ValueError("report runtime projection mismatch")

    source = _closed_object(
        report["source"],
        field="report.source",
        required={
            "model_id",
            "revision",
            "checkpoint_manifest_fingerprint",
            "checkpoint_report_fingerprint",
            "loader_input",
            "all_selected_checkpoint_file_bytes_verified_before_load",
            "selected_checkpoint_total_bytes",
            "selected_checkpoint_files",
        },
    )
    expected_files = [
        {
            "filename": item.filename,
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
            "verified": True,
        }
        for item in checkpoint_spec.files
    ]
    if source != {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "checkpoint_manifest_fingerprint": checkpoint_spec.manifest_fingerprint,
        "checkpoint_report_fingerprint": spec.source_checkpoint_report_fingerprint,
        "loader_input": "verified_local_snapshot_directory",
        "all_selected_checkpoint_file_bytes_verified_before_load": True,
        "selected_checkpoint_total_bytes": sum(item.size_bytes for item in checkpoint_spec.files),
        "selected_checkpoint_files": expected_files,
    }:
        raise ValueError("report source projection mismatch")
    data = _closed_object(
        report["data"],
        field="report.data",
        required={
            "training_artifact",
            "readiness_artifact",
            "training_manifest_fingerprint",
            "readiness_manifest_fingerprint",
            "record_ids",
            "authored_fixture_records",
            "held_out_plaintext_embedded",
        },
    )
    if data != {
        "training_artifact": training_artifact,
        "readiness_artifact": readiness_artifact,
        "training_manifest_fingerprint": data_audit.manifest_fingerprint,
        "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
        "record_ids": [sample.record_id for sample in spec.samples],
        "authored_fixture_records": True,
        "held_out_plaintext_embedded": False,
    }:
        raise ValueError("report data projection mismatch")

    result = _closed_object(
        report["result"],
        field="report.result",
        required={"model", "template", "samples", "collator", "execution"},
    )
    model = _closed_object(
        result["model"],
        field="report.result.model",
        required={
            "class",
            "model_type",
            "base_parameter_count",
            "parameter_dtypes",
            "target_checkpoint_weights_loaded",
        },
    )
    if model != {
        "class": spec.expected_model_class,
        "model_type": spec.expected_model_type,
        "base_parameter_count": spec.expected_base_parameter_count,
        "parameter_dtypes": ["torch.float32"],
        "target_checkpoint_weights_loaded": True,
    }:
        raise ValueError("report model projection mismatch")
    template = _closed_object(
        result["template"],
        field="report.result.template",
        required={
            "checkpoint_native_generation_marker_present",
            "checkpoint_native_all_zero_assistant_mask_observed",
            "reviewed_generation_marker_present",
            "reviewed_render_matches_native_input_ids",
            "reviewed_mask_matches_authored_assistant_generation_text",
            "supported_roles",
            "tool_calls_supported_by_evidence",
            "multi_assistant_turns_supported_by_evidence",
            "record_tools_forwarded_to_chat_template",
            "arrow_pre_tokenization_executed",
            "raw_nested_records_passed_to_arrow",
        },
    )
    capabilities = _fixture_capabilities(records)
    expected_template = {
        "checkpoint_native_generation_marker_present": False,
        "checkpoint_native_all_zero_assistant_mask_observed": True,
        "reviewed_generation_marker_present": True,
        "reviewed_render_matches_native_input_ids": True,
        "reviewed_mask_matches_authored_assistant_generation_text": True,
        **capabilities,
        "record_tools_forwarded_to_chat_template": True,
        "arrow_pre_tokenization_executed": True,
        "raw_nested_records_passed_to_arrow": False,
    }
    if template != expected_template:
        raise ValueError("report template result projection mismatch")
    samples = result["samples"]
    if not isinstance(samples, list) or len(samples) != len(spec.samples):
        raise ValueError("report sample count mismatch")
    for index, (sample, expected) in enumerate(zip(samples, spec.samples, strict=True)):
        observed = _closed_object(
            sample,
            field=f"report.result.samples[{index}]",
            required={
                "record_id",
                "native_input_ids_equal",
                "assistant_generation_text_equal",
                "input_ids_canonical_sha256",
                "assistant_masks_canonical_sha256",
                "assistant_token_ids_canonical_sha256",
                "unpadded_labels_canonical_sha256",
                "input_token_count",
                "assistant_token_count",
            },
        )
        assistant_ids = [
            token
            for token, mask in zip(expected.input_ids, expected.assistant_masks, strict=True)
            if mask
        ]
        labels = [
            token if mask else spec.label_pad_token_id
            for token, mask in zip(expected.input_ids, expected.assistant_masks, strict=True)
        ]
        if observed != {
            "record_id": expected.record_id,
            "native_input_ids_equal": True,
            "assistant_generation_text_equal": True,
            "input_ids_canonical_sha256": _canonical_sha256(list(expected.input_ids)),
            "assistant_masks_canonical_sha256": _canonical_sha256(
                list(expected.assistant_masks)
            ),
            "assistant_token_ids_canonical_sha256": _canonical_sha256(assistant_ids),
            "unpadded_labels_canonical_sha256": _canonical_sha256(labels),
            "input_token_count": len(expected.input_ids),
            "assistant_token_count": sum(expected.assistant_masks),
        }:
            raise ValueError(f"report sample {index} projection mismatch")

    input_rows, attention_rows, label_rows = _expected_batch(spec)
    collator = _closed_object(
        result["collator"],
        field="report.result.collator",
        required={
            "batch_shape",
            "input_ids_canonical_sha256",
            "attention_mask_canonical_sha256",
            "labels_canonical_sha256",
            "attention_token_count",
            "padding_token_count",
            "supervised_label_count",
            "ignored_label_count",
            "non_assistant_and_padding_labels_are_minus_100",
            "assistant_labels_equal_input_ids",
            "silent_truncation_observed",
        },
    )
    expected_collator = {
        "batch_shape": list(spec.expected_batch_shape),
        "input_ids_canonical_sha256": _canonical_sha256(input_rows),
        "attention_mask_canonical_sha256": _canonical_sha256(attention_rows),
        "labels_canonical_sha256": _canonical_sha256(label_rows),
        "attention_token_count": sum(value for row in attention_rows for value in row),
        "padding_token_count": sum(value == 0 for row in attention_rows for value in row),
        "supervised_label_count": sum(
            label != spec.label_pad_token_id for row in label_rows for label in row
        ),
        "ignored_label_count": sum(
            label == spec.label_pad_token_id for row in label_rows for label in row
        ),
        "non_assistant_and_padding_labels_are_minus_100": True,
        "assistant_labels_equal_input_ids": True,
        "silent_truncation_observed": False,
    }
    if collator != expected_collator:
        raise ValueError("report collator projection mismatch")
    execution = _closed_object(
        result["execution"],
        field="report.result.execution",
        required={
            "target_forward_executed",
            "backward_executed",
            "optimizer_step_count",
            "forward_loss",
            "forward_loss_finite",
        },
    )
    loss = _finite_number(execution["forward_loss"], "report.result.execution.forward_loss")
    if not math.isclose(
        loss,
        spec.expected_forward_loss,
        rel_tol=0.0,
        abs_tol=spec.forward_loss_atol,
    ) or execution != {
        "target_forward_executed": True,
        "backward_executed": False,
        "optimizer_step_count": 0,
        "forward_loss": loss,
        "forward_loss_finite": True,
    }:
        raise ValueError("report execution projection mismatch")
    if report["scope"] != _scope_projection(records):
        raise ValueError("report scope projection mismatch")
    return report
