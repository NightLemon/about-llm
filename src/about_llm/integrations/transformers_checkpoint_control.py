"""Real-weight Transformers checkpoint control with explicit evidence limits."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, NoReturn, cast
from urllib.parse import urlsplit

from about_llm.integrations.transformers_tools import parameter_report
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes
from about_llm.model_config import inspect_decoder_config, load_model_config_json

TRANSFORMERS_CHECKPOINT_CONTROL_VERSION = (
    "about-llm.transformers-checkpoint-control.v1"
)
TRANSFORMERS_CHECKPOINT_REPORT_VERSION = (
    "about-llm.transformers-checkpoint-control-report.v1"
)
TRANSFORMERS_CHECKPOINT_EVIDENCE_BOUNDARY = (
    "This control hashes selected files from an immutable-revision repository snapshot, "
    "then loads from the verified local snapshot paths with trust_remote_code disabled, "
    "and executes one fixed CPU FP32 eager forward/cache/generate case. It does not "
    "authenticate the publisher, prove every repository file or config-weight semantic "
    "match, eliminate the verification-to-loader-reopen TOCTOU window, reproduce "
    "training, establish model quality or effective context, test other prompts/"
    "languages/dtypes/devices/runtimes, benchmark performance, validate licensing, or "
    "prove production safety."
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_FILES = 32
_MAX_MESSAGES = 16
_MAX_MESSAGE_CHARACTERS = 4096
_CONTROL_NEW_TOKENS = 2
_CACHE_EQUIVALENCE_ATOL = 1e-4
_MANIFEST_FIELDS = {
    "control_version",
    "checked_at",
    "model_id",
    "revision",
    "source_base_url",
    "expected_model_class",
    "expected_model_type",
    "dtype",
    "device",
    "attention_implementation",
    "max_new_tokens",
    "messages",
    "files",
    "evidence_boundary",
}
_FILE_FIELDS = {"filename", "size_bytes", "sha256"}
_MESSAGE_FIELDS = {"role", "content"}
_REQUIRED_EXECUTION_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}
_REPORT_FIELDS = {
    "report_version",
    "manifest_fingerprint",
    "checked_at",
    "source",
    "artifacts",
    "runtime",
    "model",
    "tokenizer",
    "execution",
    "scope",
    "evidence_boundary",
    "report_fingerprint",
}
_REPORT_SOURCE_FIELDS = {
    "model_id",
    "revision",
    "source_base_url",
    "public_anonymous_download",
    "all_selected_file_bytes_verified_before_load",
    "loader_input",
}
_REPORT_ARTIFACT_FIELDS = {"selected_file_count", "selected_total_bytes", "files"}
_REPORT_ARTIFACT_FILE_FIELDS = {"filename", "size_bytes", "sha256", "verified"}
_REPORT_RUNTIME_FIELDS = {
    "python_implementation",
    "python_version",
    "platform",
    "torch_version",
    "transformers_version",
    "device",
    "dtype",
    "attention_implementation",
    "torch_num_threads",
    "cuda_executed",
}
_REPORT_MODEL_FIELDS = {
    "class",
    "model_type",
    "architectures",
    "verified_raw_config_semantic_fingerprint",
    "config_snapshot_source",
    "parameter_report",
    "parameter_dtypes",
    "eval_mode",
}
_REPORT_PARAMETER_FIELDS = {
    "total_parameters",
    "trainable_parameters",
    "trainable_fraction",
    "parameter_storage_bytes",
}
_REPORT_TOKENIZER_FIELDS = {
    "class",
    "vocabulary_size_with_added_tokens",
    "chat_template_fingerprint",
    "pad_token_id",
    "eos_token_id",
}
_REPORT_EXECUTION_FIELDS = {
    "prompt_token_count",
    "prompt_token_ids_fingerprint",
    "prefill_logits_shape",
    "prefill_last_logits_sha256",
    "cached_second_logits_sha256",
    "generated_token_ids",
    "decoded_continuation",
    "manual_prefill_argmax_matches_generate",
    "manual_cached_argmax_matches_generate",
    "cached_full_argmax_match",
    "cached_full_max_abs_error",
    "cached_full_tolerance",
    "past_key_values_executed",
    "parameters_frozen_for_control",
    "generation_eos_token_id",
    "generated_ended_with_eos",
    "generation_sampling_disabled",
}
_REPORT_SCOPE_FIELDS = {
    "target_checkpoint_weights_loaded",
    "real_model_forward_executed",
    "framework_generate_executed",
    "manual_kv_cache_path_executed",
    "trust_remote_code",
    "training_or_backward_executed",
    "model_quality_proven",
    "effective_context_proven",
    "gpu_or_vllm_executed",
    "performance_benchmark_performed",
    "publisher_authenticated_by_signature",
    "license_compatibility_proven",
    "production_safety_proven",
    "verification_to_loader_reopen_toctou_eliminated",
}


@dataclass(frozen=True)
class CheckpointFileEvidence:
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CheckpointMessage:
    role: str
    content: str


@dataclass(frozen=True)
class CheckpointControlSpec:
    checked_at: str
    model_id: str
    revision: str
    source_base_url: str
    expected_model_class: str
    expected_model_type: str
    dtype: str
    device: str
    attention_implementation: str
    max_new_tokens: int
    messages: tuple[CheckpointMessage, ...]
    files: tuple[CheckpointFileEvidence, ...]
    manifest_fingerprint: str


@dataclass(frozen=True)
class VerifiedCheckpointSnapshot:
    directory: Path
    files: tuple[Mapping[str, object], ...]


def load_checkpoint_control_spec(path: Path) -> CheckpointControlSpec:
    """Load a strict, immutable checkpoint-control manifest."""

    manifest = _load_json_file(path)
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "manifest")
    if manifest.get("control_version") != TRANSFORMERS_CHECKPOINT_CONTROL_VERSION:
        raise ValueError("manifest.control_version is unsupported")
    if manifest.get("evidence_boundary") != TRANSFORMERS_CHECKPOINT_EVIDENCE_BOUNDARY:
        raise ValueError("manifest.evidence_boundary drift")
    checked_at = _required_string(manifest, "checked_at", "manifest")
    try:
        date.fromisoformat(checked_at)
    except ValueError as error:
        raise ValueError("manifest.checked_at must be an ISO date") from error
    model_id = _required_string(manifest, "model_id", "manifest")
    if _MODEL_ID.fullmatch(model_id) is None:
        raise ValueError("manifest.model_id must be a simple owner/repository id")
    revision = _required_string(manifest, "revision", "manifest")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("manifest.revision must be a 40-character commit id")
    source_base_url = _required_string(manifest, "source_base_url", "manifest")
    expected_url = f"https://huggingface.co/{model_id}/resolve/{revision}/"
    if source_base_url != expected_url:
        raise ValueError("manifest.source_base_url does not match model_id/revision")
    parsed_url = urlsplit(source_base_url)
    if (
        parsed_url.hostname != "huggingface.co"
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.port not in (None, 443)
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError("manifest.source_base_url must be public Hugging Face HTTPS")
    expected_model_class = _required_string(
        manifest, "expected_model_class", "manifest"
    )
    expected_model_type = _required_string(
        manifest, "expected_model_type", "manifest"
    )
    dtype = _required_string(manifest, "dtype", "manifest")
    device = _required_string(manifest, "device", "manifest")
    attention_implementation = _required_string(
        manifest, "attention_implementation", "manifest"
    )
    if (dtype, device, attention_implementation) != ("float32", "cpu", "eager"):
        raise ValueError("manifest runtime must be the reviewed float32/cpu/eager control")
    max_new_tokens = _positive_integer(
        manifest.get("max_new_tokens"), "manifest.max_new_tokens"
    )
    if max_new_tokens != _CONTROL_NEW_TOKENS:
        raise ValueError("manifest.max_new_tokens must be exactly 2 for control v1")
    messages = _parse_messages(manifest.get("messages"))
    files = _parse_files(manifest.get("files"))
    manifest_fingerprint = _canonical_sha256(manifest)
    return CheckpointControlSpec(
        checked_at=checked_at,
        model_id=model_id,
        revision=revision,
        source_base_url=source_base_url,
        expected_model_class=expected_model_class,
        expected_model_type=expected_model_type,
        dtype=dtype,
        device=device,
        attention_implementation=attention_implementation,
        max_new_tokens=max_new_tokens,
        messages=messages,
        files=files,
        manifest_fingerprint=manifest_fingerprint,
    )


def download_checkpoint_snapshot(
    spec: CheckpointControlSpec,
    *,
    local_files_only: bool = False,
) -> Path:
    """Resolve reviewed public files into one immutable Hugging Face snapshot."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("huggingface_hub is required for checkpoint download") from error
    directory = snapshot_download(
        repo_id=spec.model_id,
        revision=spec.revision,
        repo_type="model",
        allow_patterns=[item.filename for item in spec.files],
        local_files_only=local_files_only,
        token=False,
        max_workers=4,
    )
    return Path(directory)


def verify_checkpoint_snapshot(
    spec: CheckpointControlSpec, snapshot_directory: Path
) -> VerifiedCheckpointSnapshot:
    """Hash every selected file and return a path-free public evidence projection."""

    root = snapshot_directory.resolve()
    if not root.is_dir():
        raise ValueError("checkpoint snapshot directory does not exist")
    verified: list[Mapping[str, object]] = []
    for expected in spec.files:
        path = root / expected.filename
        if not path.is_file():
            raise ValueError(f"checkpoint file is missing: {expected.filename}")
        actual_size, actual_sha256 = _hash_open_file(path)
        if actual_size != expected.size_bytes:
            raise ValueError(f"checkpoint file size mismatch: {expected.filename}")
        if not hmac.compare_digest(actual_sha256, expected.sha256):
            raise ValueError(f"checkpoint file SHA-256 mismatch: {expected.filename}")
        verified.append(
            {
                "filename": expected.filename,
                "size_bytes": actual_size,
                "sha256": actual_sha256,
                "verified": True,
            }
        )
    return VerifiedCheckpointSnapshot(directory=root, files=tuple(verified))


def execute_loaded_checkpoint_control(
    spec: CheckpointControlSpec,
    *,
    model: Any,
    tokenizer: Any,
) -> dict[str, object]:
    """Execute manual prefill/cache steps and compare them with framework generate."""

    try:
        import torch
        from transformers import GenerationConfig
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("torch and transformers are required for execution") from error
    if type(model).__name__ != spec.expected_model_class:
        raise ValueError(
            f"loaded model class mismatch: expected {spec.expected_model_class}, "
            f"got {type(model).__name__}"
        )
    if getattr(model.config, "model_type", None) != spec.expected_model_type:
        raise ValueError("loaded model_type does not match the reviewed manifest")
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("target tokenizer must provide a chat template")
    model.to("cpu")
    model.requires_grad_(False)
    model.eval()
    messages = [
        {"role": message.role, "content": message.content}
        for message in spec.messages
    ]
    raw_input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(raw_input_ids, torch.Tensor):
        raise RuntimeError("chat template must return a tensor for this control")
    input_ids = raw_input_ids.to(device="cpu", dtype=torch.long)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise RuntimeError("chat template returned an invalid [1, tokens] tensor")
    if torch.any(input_ids < 0) or torch.any(input_ids >= len(tokenizer)):
        raise RuntimeError("chat template returned an out-of-vocabulary token id")
    attention_mask = torch.ones_like(input_ids)
    generation_config = GenerationConfig(  # type: ignore[no-untyped-call]
        do_sample=False,
        max_new_tokens=spec.max_new_tokens,
        repetition_penalty=1.0,
        use_cache=True,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        bos_token_id=None,
    )
    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        prefill_logits = prefill.logits[:, -1, :].to(dtype=torch.float32)
        if prefill_logits.ndim != 2 or prefill_logits.shape[0] != 1:
            raise RuntimeError("prefill logits have an invalid shape")
        if prefill.past_key_values is None:
            raise RuntimeError("prefill did not return a cache")
        first_argmax = int(torch.argmax(prefill_logits, dim=-1).item())
        first_token = torch.tensor([[first_argmax]], dtype=torch.long)
        cached_attention_mask = torch.ones(
            (1, input_ids.shape[1] + 1), dtype=torch.long
        )
        cached = model(
            input_ids=first_token,
            attention_mask=cached_attention_mask,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        cached_logits = cached.logits[:, -1, :].to(dtype=torch.float32)
        full_input_ids = torch.cat((input_ids, first_token), dim=1)
        full = model(
            input_ids=full_input_ids,
            attention_mask=torch.ones_like(full_input_ids),
            use_cache=False,
            return_dict=True,
        )
        full_logits = full.logits[:, -1, :].to(dtype=torch.float32)
        max_abs_cache_error = float(
            torch.max(torch.abs(cached_logits - full_logits)).item()
        )
        cached_second_argmax = int(torch.argmax(cached_logits, dim=-1).item())
        full_second_argmax = int(torch.argmax(full_logits, dim=-1).item())
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
    sequences = generated.sequences
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        raise RuntimeError("generate returned invalid sequences")
    continuation = sequences[0, input_ids.shape[1] :].to(dtype=torch.long)
    generated_ids = [int(value) for value in continuation.tolist()]
    if len(generated_ids) != spec.max_new_tokens:
        raise RuntimeError("generate did not return the reviewed fixed token count")
    if generated_ids[0] != first_argmax:
        raise RuntimeError(
            "generate first token differs from raw prefill argmax: "
            f"generated={generated_ids[0]}, manual={first_argmax}"
        )
    if generated_ids[1] != cached_second_argmax:
        raise RuntimeError(
            "generate second token differs from manual cached argmax: "
            f"generated={generated_ids[1]}, manual={cached_second_argmax}"
        )
    if cached_second_argmax != full_second_argmax:
        raise RuntimeError("cached and full-recompute second-token argmax differ")
    if max_abs_cache_error > _CACHE_EQUIVALENCE_ATOL:
        raise RuntimeError("cached/full logits exceed the reviewed FP32 tolerance")
    decoded = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        raise RuntimeError("tokenizer.decode returned a non-string value")
    input_id_list = [int(value) for value in input_ids[0].tolist()]
    return {
        "prompt_token_count": len(input_id_list),
        "prompt_token_ids_fingerprint": _canonical_sha256(
            {"token_ids": input_id_list}
        ),
        "prefill_logits_shape": [int(value) for value in prefill.logits.shape],
        "prefill_last_logits_sha256": _tensor_sha256(prefill_logits),
        "cached_second_logits_sha256": _tensor_sha256(cached_logits),
        "generated_token_ids": generated_ids,
        "decoded_continuation": decoded,
        "manual_prefill_argmax_matches_generate": True,
        "manual_cached_argmax_matches_generate": True,
        "cached_full_argmax_match": True,
        "cached_full_max_abs_error": max_abs_cache_error,
        "cached_full_tolerance": _CACHE_EQUIVALENCE_ATOL,
        "past_key_values_executed": True,
        "parameters_frozen_for_control": True,
        "generation_eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "generated_ended_with_eos": bool(
            generated_ids
            and generated_ids[-1] == getattr(tokenizer, "eos_token_id", None)
        ),
        "generation_sampling_disabled": True,
    }


def run_checkpoint_control(
    spec: CheckpointControlSpec,
    *,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Download, hash, load and execute the reviewed target checkpoint."""

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - dependency error is environment-specific
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
    from packaging.version import Version

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
    execution = execute_loaded_checkpoint_control(spec, model=model, tokenizer=tokenizer)
    raw_config_snapshot = load_model_config_json(snapshot.directory / "config.json")
    config_inspection = inspect_decoder_config(raw_config_snapshot)
    parameters = parameter_report(model)
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise RuntimeError("loaded tokenizer chat template disappeared")
    projection: dict[str, object] = {
        "report_version": TRANSFORMERS_CHECKPOINT_REPORT_VERSION,
        "manifest_fingerprint": spec.manifest_fingerprint,
        "checked_at": spec.checked_at,
        "source": {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "source_base_url": spec.source_base_url,
            "public_anonymous_download": True,
            "all_selected_file_bytes_verified_before_load": True,
            "loader_input": "verified_local_snapshot_directory",
        },
        "artifacts": {
            "selected_file_count": len(snapshot.files),
            "selected_total_bytes": sum(
                cast(int, item["size_bytes"]) for item in snapshot.files
            ),
            "files": [dict(item) for item in snapshot.files],
        },
        "runtime": {
            "python_implementation": __import__("platform").python_implementation(),
            "python_version": __import__("platform").python_version(),
            "platform": __import__("platform").platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": spec.device,
            "dtype": spec.dtype,
            "attention_implementation": spec.attention_implementation,
            "torch_num_threads": torch.get_num_threads(),
            "cuda_executed": False,
        },
        "model": {
            "class": type(model).__name__,
            "model_type": getattr(model.config, "model_type", None),
            "architectures": getattr(model.config, "architectures", None),
            "verified_raw_config_semantic_fingerprint": (
                config_inspection.config_fingerprint
            ),
            "config_snapshot_source": (
                "strict parsed config.json after its exact bytes were verified"
            ),
            "parameter_report": parameters,
            "parameter_dtypes": parameter_dtypes,
            "eval_mode": model.training is False,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocabulary_size_with_added_tokens": len(tokenizer),
            "chat_template_fingerprint": _canonical_sha256(
                {"chat_template": chat_template}
            ),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        },
        "execution": execution,
        "scope": {
            "target_checkpoint_weights_loaded": True,
            "real_model_forward_executed": True,
            "framework_generate_executed": True,
            "manual_kv_cache_path_executed": True,
            "trust_remote_code": False,
            "training_or_backward_executed": False,
            "model_quality_proven": False,
            "effective_context_proven": False,
            "gpu_or_vllm_executed": False,
            "performance_benchmark_performed": False,
            "publisher_authenticated_by_signature": False,
            "license_compatibility_proven": False,
            "production_safety_proven": False,
            "verification_to_loader_reopen_toctou_eliminated": False,
        },
        "evidence_boundary": TRANSFORMERS_CHECKPOINT_EVIDENCE_BOUNDARY,
    }
    projection["report_fingerprint"] = _canonical_sha256(projection)
    return projection


def verify_recorded_checkpoint_report(
    path: Path,
    *,
    expected_manifest_fingerprint: str | None = None,
) -> Mapping[str, Any]:
    """Verify the closed report schema, internal consistency and self-fingerprint."""

    report = _load_json_file(path)
    _require_exact_fields(report, _REPORT_FIELDS, "report")
    if report.get("report_version") != TRANSFORMERS_CHECKPOINT_REPORT_VERSION:
        raise ValueError("report.report_version is unsupported")
    if report.get("evidence_boundary") != TRANSFORMERS_CHECKPOINT_EVIDENCE_BOUNDARY:
        raise ValueError("report.evidence_boundary drift")
    fingerprint = _sha256(report.get("report_fingerprint"), "report")
    projection = dict(report)
    del projection["report_fingerprint"]
    if not hmac.compare_digest(fingerprint, _canonical_sha256(projection)):
        raise ValueError("report fingerprint mismatch")
    manifest_fingerprint = _sha256(report.get("manifest_fingerprint"), "report")
    if (
        expected_manifest_fingerprint is not None
        and manifest_fingerprint != expected_manifest_fingerprint
    ):
        raise ValueError("report manifest fingerprint mismatch")
    _validate_recorded_report_schema(report)
    scope = _mapping(report.get("scope"), "report.scope")
    required_true = (
        "target_checkpoint_weights_loaded",
        "real_model_forward_executed",
        "framework_generate_executed",
        "manual_kv_cache_path_executed",
    )
    required_false = (
        "trust_remote_code",
        "training_or_backward_executed",
        "model_quality_proven",
        "effective_context_proven",
        "gpu_or_vllm_executed",
        "performance_benchmark_performed",
        "publisher_authenticated_by_signature",
        "license_compatibility_proven",
        "production_safety_proven",
        "verification_to_loader_reopen_toctou_eliminated",
    )
    if any(scope.get(name) is not True for name in required_true):
        raise ValueError("report is missing required executed-scope evidence")
    if any(scope.get(name) is not False for name in required_false):
        raise ValueError("report overstates an excluded evidence scope")
    return report


def _validate_recorded_report_schema(report: Mapping[str, Any]) -> None:
    checked_at = _required_string(report, "checked_at", "report")
    try:
        date.fromisoformat(checked_at)
    except ValueError as error:
        raise ValueError("report.checked_at must be an ISO date") from error

    source = _mapping(report.get("source"), "report.source")
    _require_exact_fields(source, _REPORT_SOURCE_FIELDS, "report.source")
    model_id = _required_string(source, "model_id", "report.source")
    revision = _required_string(source, "revision", "report.source")
    source_base_url = _required_string(source, "source_base_url", "report.source")
    if _MODEL_ID.fullmatch(model_id) is None or _REVISION.fullmatch(revision) is None:
        raise ValueError("report.source model_id/revision format mismatch")
    if source_base_url != f"https://huggingface.co/{model_id}/resolve/{revision}/":
        raise ValueError("report.source source_base_url does not match model_id/revision")
    _require_boolean(source, "public_anonymous_download", "report.source", expected=True)
    _require_boolean(
        source,
        "all_selected_file_bytes_verified_before_load",
        "report.source",
        expected=True,
    )
    if source.get("loader_input") != "verified_local_snapshot_directory":
        raise ValueError("report.source.loader_input is unsupported")

    artifacts = _mapping(report.get("artifacts"), "report.artifacts")
    _require_exact_fields(artifacts, _REPORT_ARTIFACT_FIELDS, "report.artifacts")
    selected_file_count = _positive_integer(
        artifacts.get("selected_file_count"), "report.artifacts.selected_file_count"
    )
    if selected_file_count > _MAX_FILES:
        raise ValueError("report.artifacts.selected_file_count exceeds the file limit")
    selected_total_bytes = _positive_integer(
        artifacts.get("selected_total_bytes"), "report.artifacts.selected_total_bytes"
    )
    artifact_files = _bounded_array(
        artifacts.get("files"), "report.artifacts.files", maximum=_MAX_FILES
    )
    seen_files: set[str] = set()
    observed_total_bytes = 0
    for index, item in enumerate(artifact_files):
        location = f"report.artifacts.files[{index}]"
        file_report = _mapping(item, location)
        _require_exact_fields(file_report, _REPORT_ARTIFACT_FILE_FIELDS, location)
        filename = _required_string(file_report, "filename", location)
        if _FILENAME.fullmatch(filename) is None or filename in seen_files:
            raise ValueError(f"{location}.filename is invalid or duplicated")
        seen_files.add(filename)
        observed_total_bytes += _positive_integer(
            file_report.get("size_bytes"), f"{location}.size_bytes"
        )
        _sha256(file_report.get("sha256"), location)
        _require_boolean(file_report, "verified", location, expected=True)
    if len(artifact_files) != selected_file_count:
        raise ValueError("report.artifacts.selected_file_count does not match files")
    if observed_total_bytes != selected_total_bytes:
        raise ValueError("report.artifacts.selected_total_bytes does not match files")
    if not _REQUIRED_EXECUTION_FILES.issubset(seen_files):
        raise ValueError("report.artifacts.files misses required execution files")

    runtime = _mapping(report.get("runtime"), "report.runtime")
    _require_exact_fields(runtime, _REPORT_RUNTIME_FIELDS, "report.runtime")
    for field in (
        "python_implementation",
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
    ):
        _required_string(runtime, field, "report.runtime")
    if (
        runtime.get("device"),
        runtime.get("dtype"),
        runtime.get("attention_implementation"),
    ) != ("cpu", "float32", "eager"):
        raise ValueError("report.runtime is not the reviewed cpu/float32/eager runtime")
    _positive_integer(runtime.get("torch_num_threads"), "report.runtime.torch_num_threads")
    _require_boolean(runtime, "cuda_executed", "report.runtime", expected=False)

    model = _mapping(report.get("model"), "report.model")
    _require_exact_fields(model, _REPORT_MODEL_FIELDS, "report.model")
    _required_string(model, "class", "report.model")
    _required_string(model, "model_type", "report.model")
    _string_array(model.get("architectures"), "report.model.architectures", maximum=32)
    _sha256(
        model.get("verified_raw_config_semantic_fingerprint"),
        "report.model.verified_raw_config_semantic_fingerprint",
    )
    if model.get("config_snapshot_source") != (
        "strict parsed config.json after its exact bytes were verified"
    ):
        raise ValueError("report.model.config_snapshot_source is unsupported")
    parameters = _mapping(model.get("parameter_report"), "report.model.parameter_report")
    _require_exact_fields(
        parameters, _REPORT_PARAMETER_FIELDS, "report.model.parameter_report"
    )
    total_parameters = _positive_integer(
        parameters.get("total_parameters"), "report.model.parameter_report.total_parameters"
    )
    trainable_parameters = _non_negative_integer(
        parameters.get("trainable_parameters"),
        "report.model.parameter_report.trainable_parameters",
    )
    trainable_fraction = _finite_number(
        parameters.get("trainable_fraction"),
        "report.model.parameter_report.trainable_fraction",
    )
    parameter_storage_bytes = _positive_integer(
        parameters.get("parameter_storage_bytes"),
        "report.model.parameter_report.parameter_storage_bytes",
    )
    if trainable_parameters != 0 or trainable_fraction != 0.0:
        raise ValueError("report.model parameters were not fully frozen for inference")
    if trainable_parameters > total_parameters:
        raise ValueError("report.model trainable parameters exceed total parameters")
    if parameter_storage_bytes != total_parameters * 4:
        raise ValueError("report.model FP32 parameter storage is inconsistent")
    if _string_array(
        model.get("parameter_dtypes"), "report.model.parameter_dtypes", maximum=16
    ) != ["torch.float32"]:
        raise ValueError("report.model.parameter_dtypes is not the reviewed FP32 contract")
    _require_boolean(model, "eval_mode", "report.model", expected=True)

    tokenizer = _mapping(report.get("tokenizer"), "report.tokenizer")
    _require_exact_fields(tokenizer, _REPORT_TOKENIZER_FIELDS, "report.tokenizer")
    _required_string(tokenizer, "class", "report.tokenizer")
    tokenizer_vocabulary_size = _positive_integer(
        tokenizer.get("vocabulary_size_with_added_tokens"),
        "report.tokenizer.vocabulary_size_with_added_tokens",
    )
    _sha256(
        tokenizer.get("chat_template_fingerprint"),
        "report.tokenizer.chat_template_fingerprint",
    )
    pad_token_id = _non_negative_integer(
        tokenizer.get("pad_token_id"), "report.tokenizer.pad_token_id"
    )
    eos_token_id = _non_negative_integer(
        tokenizer.get("eos_token_id"), "report.tokenizer.eos_token_id"
    )
    if pad_token_id == eos_token_id:
        raise ValueError("report.tokenizer PAD and EOS ids unexpectedly coincide")
    if max(pad_token_id, eos_token_id) >= tokenizer_vocabulary_size:
        raise ValueError("report.tokenizer special token id exceeds its vocabulary")

    execution = _mapping(report.get("execution"), "report.execution")
    _require_exact_fields(execution, _REPORT_EXECUTION_FIELDS, "report.execution")
    prompt_token_count = _positive_integer(
        execution.get("prompt_token_count"), "report.execution.prompt_token_count"
    )
    _sha256(
        execution.get("prompt_token_ids_fingerprint"),
        "report.execution.prompt_token_ids_fingerprint",
    )
    prefill_shape_raw = _bounded_array(
        execution.get("prefill_logits_shape"),
        "report.execution.prefill_logits_shape",
        maximum=3,
    )
    prefill_shape = [
        _positive_integer(value, f"report.execution.prefill_logits_shape[{index}]")
        for index, value in enumerate(prefill_shape_raw)
    ]
    if len(prefill_shape) != 3 or prefill_shape[:2] != [1, prompt_token_count]:
        raise ValueError("report.execution.prefill_logits_shape is inconsistent")
    _sha256(
        execution.get("prefill_last_logits_sha256"),
        "report.execution.prefill_last_logits_sha256",
    )
    _sha256(
        execution.get("cached_second_logits_sha256"),
        "report.execution.cached_second_logits_sha256",
    )
    generated_raw = _bounded_array(
        execution.get("generated_token_ids"),
        "report.execution.generated_token_ids",
        maximum=_CONTROL_NEW_TOKENS,
    )
    generated_ids = [
        _non_negative_integer(value, f"report.execution.generated_token_ids[{index}]")
        for index, value in enumerate(generated_raw)
    ]
    if len(generated_ids) != _CONTROL_NEW_TOKENS:
        raise ValueError("report.execution.generated_token_ids must contain exactly 2 ids")
    if any(token_id >= prefill_shape[2] for token_id in generated_ids):
        raise ValueError("report.execution generated token id exceeds the logits vocabulary")
    if not isinstance(execution.get("decoded_continuation"), str):
        raise ValueError("report.execution.decoded_continuation must be a string")
    for field in (
        "manual_prefill_argmax_matches_generate",
        "manual_cached_argmax_matches_generate",
        "cached_full_argmax_match",
        "past_key_values_executed",
        "parameters_frozen_for_control",
        "generated_ended_with_eos",
        "generation_sampling_disabled",
    ):
        _require_boolean(execution, field, "report.execution", expected=True)
    max_abs_error = _finite_number(
        execution.get("cached_full_max_abs_error"),
        "report.execution.cached_full_max_abs_error",
    )
    tolerance = _finite_number(
        execution.get("cached_full_tolerance"),
        "report.execution.cached_full_tolerance",
    )
    if tolerance != _CACHE_EQUIVALENCE_ATOL:
        raise ValueError("report.execution cached/full tolerance is not the reviewed value")
    if max_abs_error < 0.0 or max_abs_error > tolerance:
        raise ValueError("report.execution cached/full error exceeds its tolerance")
    generation_eos_token_id = _non_negative_integer(
        execution.get("generation_eos_token_id"),
        "report.execution.generation_eos_token_id",
    )
    if generation_eos_token_id != eos_token_id or generated_ids[-1] != eos_token_id:
        raise ValueError("report.execution generated EOS is inconsistent with tokenizer")

    scope = _mapping(report.get("scope"), "report.scope")
    _require_exact_fields(scope, _REPORT_SCOPE_FIELDS, "report.scope")


def _parse_messages(value: Any) -> tuple[CheckpointMessage, ...]:
    if not isinstance(value, list) or not value or len(value) > _MAX_MESSAGES:
        raise ValueError("manifest.messages must be a bounded non-empty array")
    messages: list[CheckpointMessage] = []
    for index, item in enumerate(value):
        location = f"manifest.messages[{index}]"
        mapping = _mapping(item, location)
        _require_exact_fields(mapping, _MESSAGE_FIELDS, location)
        role = _required_string(mapping, "role", location)
        content = _required_string(mapping, "content", location)
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"{location}.role is unsupported")
        if len(content) > _MAX_MESSAGE_CHARACTERS:
            raise ValueError(f"{location}.content exceeds the character limit")
        messages.append(CheckpointMessage(role=role, content=content))
    if messages[-1].role != "user":
        raise ValueError("manifest.messages must end with a user message")
    return tuple(messages)


def _parse_files(value: Any) -> tuple[CheckpointFileEvidence, ...]:
    if not isinstance(value, list) or not value or len(value) > _MAX_FILES:
        raise ValueError("manifest.files must be a bounded non-empty array")
    files: list[CheckpointFileEvidence] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        location = f"manifest.files[{index}]"
        mapping = _mapping(item, location)
        _require_exact_fields(mapping, _FILE_FIELDS, location)
        filename = _required_string(mapping, "filename", location)
        if _FILENAME.fullmatch(filename) is None:
            raise ValueError(f"{location}.filename must be a simple file name")
        if filename in seen:
            raise ValueError(f"{location}.filename is duplicated")
        seen.add(filename)
        files.append(
            CheckpointFileEvidence(
                filename=filename,
                size_bytes=_positive_integer(
                    mapping.get("size_bytes"), f"{location}.size_bytes"
                ),
                sha256=_sha256(mapping.get("sha256"), location),
            )
        )
    if not _REQUIRED_EXECUTION_FILES.issubset(seen):
        missing = sorted(_REQUIRED_EXECUTION_FILES - seen)
        raise ValueError(f"manifest.files is missing required execution files: {missing}")
    return tuple(files)


def _hash_open_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as handle:
            initial_size = os.fstat(handle.fileno()).st_size
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                digest.update(chunk)
            final_size = os.fstat(handle.fileno()).st_size
    except OSError as error:
        raise ValueError(f"cannot read checkpoint file {path.name}: {error}") from error
    if initial_size != final_size or total != final_size:
        raise ValueError(f"checkpoint file changed while hashing: {path.name}")
    return total, "sha256:" + digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().to(device="cpu", dtype=__import__("torch").float32)
    contiguous = contiguous.contiguous()
    return "sha256:" + hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def _load_json_file(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{path}: cannot read JSON: {error}") from error
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError(f"{path}: JSON exceeds the size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: JSON is not valid UTF-8") from error
    try:
        value: Any = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    canonical_json_bytes(value)
    return cast(dict[str, Any], value)


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], location: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{location}: field set mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return cast(Mapping[str, Any], value)


def _required_string(value: Mapping[str, Any], name: str, location: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{location}.{name} must be a non-empty string")
    return result


def _positive_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return cast(int, value)


def _non_negative_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return cast(int, value)


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    return result


def _bounded_array(value: Any, location: str, *, maximum: int) -> list[Any]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise ValueError(f"{location} must be a bounded non-empty array")
    return value


def _string_array(value: Any, location: str, *, maximum: int) -> list[str]:
    raw = _bounded_array(value, location, maximum=maximum)
    result: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{location}[{index}] must be a non-empty string")
        result.append(item)
    return result


def _require_boolean(
    value: Mapping[str, Any],
    name: str,
    location: str,
    *,
    expected: bool,
) -> None:
    if value.get(name) is not expected:
        raise ValueError(f"{location}.{name} must be {expected}")


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{location}: expected canonical sha256 fingerprint")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return "sha256:" + artifact_fingerprint(cast(Mapping[str, object], value))


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
