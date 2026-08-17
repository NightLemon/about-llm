"""Recorded target-Qwen loopback control for the OpenAI reference service.

The live runner loads one hash-reviewed Qwen checkpoint into a Transformers CPU
backend and exercises real TCP/HTTP.  The default content checks only verify the
recorded artifact; they do not reload the 0.5B model.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hmac
import importlib.metadata
import json
import logging
import os
import platform
import re
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, NoReturn, cast

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from about_llm.inference.openai_reference import (
    CHAT_COMPLETIONS_PATH,
    HEALTH_PATH,
    MODELS_PATH,
    OPENAI_REFERENCE_SERVICE_VERSION,
    ChatMessage,
    TransformersCPUBackend,
    build_reference_app,
    decode_strict_json_object,
)
from about_llm.inference.sse import STREAM_FINISHED, parse_sse_data_line
from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    VerifiedCheckpointSnapshot,
    download_checkpoint_snapshot,
    load_checkpoint_control_spec,
    verify_checkpoint_snapshot,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

TARGET_SERVICE_CONTROL_VERSION: Final = "about-llm.target-service-control.v1"
TARGET_SERVICE_REPORT_VERSION: Final = "about-llm.target-service-control-report.v1"
CONTROL_AUDIT_PATH: Final = "/control/audit"
CONTROL_TOKEN_ENV: Final = "ABOUT_LLM_TARGET_SERVICE_TOKEN"
MAX_RECORDED_REPORT_BYTES: Final = 128_000
REQUEST_TIMEOUT_SECONDS: Final = 120.0
SERVER_START_TIMEOUT_SECONDS: Final = 300.0
EXPECTED_PARAMETER_COUNT: Final = 494_032_768
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")

TARGET_SERVICE_EVIDENCE_BOUNDARY: Final = (
    "This control verifies the selected immutable-revision checkpoint files, loads "
    "Qwen with trust_remote_code disabled into a CPU FP32 eager Transformers backend, "
    "and executes one non-streaming and one SSE request over real IPv4 loopback TCP/HTTP. "
    "The backend completes generation before emitting SSE chunks, so it does not prove "
    "incremental decode streaming or cancellation. It does not use vLLM, CUDA, TLS, a "
    "reverse proxy, OAuth/JWT/IAM, multiple workers, remote clients, a representative "
    "workload, performance benchmarking, publisher signatures, license compatibility, "
    "model quality, full OpenAI API compatibility, or production safety, and it does not "
    "eliminate the verification-to-loader-reopen TOCTOU window."
)

_MANIFEST_FIELDS = frozenset(
    {
        "control_version",
        "checked_at",
        "checkpoint_manifest_fingerprint",
        "model_id",
        "revision",
        "expected_model_class",
        "expected_model_type",
        "maximum_prompt_tokens",
        "max_tokens",
        "messages",
        "expected_prompt_token_count",
        "expected_completion_token_ids",
        "expected_completion_text_fingerprint",
        "expected_finish_reason",
        "reviewed_runtime",
        "evidence_boundary",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "python",
        "torch",
        "transformers",
        "httpx",
        "starlette",
        "uvicorn",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "report_version",
        "control_manifest_fingerprint",
        "checkpoint_manifest_fingerprint",
        "checked_at",
        "source",
        "artifacts",
        "runtime",
        "network",
        "api",
        "execution",
        "server_process",
        "scope",
        "report_fingerprint",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "model_id",
        "revision",
        "checkpoint_files_verified_before_load",
        "trust_remote_code",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {"selected_file_count", "selected_total_bytes", "files"}
)
_FILE_FIELDS = frozenset({"filename", "size_bytes", "sha256", "verified"})
_NETWORK_FIELDS = frozenset(
    {"scheme", "address_scope", "real_tcp_http", "tls", "endpoint"}
)
_API_FIELDS = frozenset(
    {
        "service_version",
        "models_endpoint_executed",
        "chat_nonstream_executed",
        "chat_sse_executed",
        "unauthorized_status",
        "unknown_field_status",
        "wrong_model_status",
        "sse_done_observed",
        "stream_usage_observed",
        "raw_request_or_response_published",
    }
)
_EXECUTION_FIELDS = frozenset(
    {
        "model_class",
        "model_type",
        "parameter_count",
        "parameters_frozen",
        "framework_generate_call_count",
        "prompt_token_count",
        "completion_token_ids",
        "completion_text_fingerprint",
        "finish_reason",
        "nonstream_stream_content_match",
        "nonstream_stream_usage_match",
        "stream_content_delta_count",
        "generation_completed_before_sse_emission",
    }
)
_PROCESS_FIELDS = frozenset(
    {"subprocess_used", "stdout_bytes", "stderr_bytes", "stdout_stderr_empty"}
)
_SCOPE_FIELDS = frozenset(
    {
        "target_checkpoint_weights_loaded",
        "transformers_generate_executed",
        "real_ipv4_loopback_tcp_http_executed",
        "vllm_executed",
        "cuda_executed",
        "incremental_model_decode_streaming_proven",
        "client_disconnect_cancellation_proven",
        "tls_reverse_proxy_or_remote_network_proven",
        "oauth_jwt_iam_or_business_authorization_proven",
        "multiple_workers_or_distributed_serving_proven",
        "performance_capacity_or_slo_proven",
        "full_openai_api_compatibility_proven",
        "model_quality_proven",
        "publisher_authenticated_by_signature",
        "license_compatibility_proven",
        "production_safety_proven",
        "verification_to_loader_reopen_toctou_eliminated",
        "report_fingerprint_proves_authenticity",
    }
)


@dataclass(frozen=True)
class TargetServiceControlSpec:
    checked_at: str
    checkpoint_manifest_fingerprint: str
    model_id: str
    revision: str
    expected_model_class: str
    expected_model_type: str
    maximum_prompt_tokens: int
    max_tokens: int
    messages: tuple[ChatMessage, ...]
    expected_prompt_token_count: int
    expected_completion_token_ids: tuple[int, ...]
    expected_completion_text_fingerprint: str
    expected_finish_reason: str
    reviewed_runtime: Mapping[str, str]
    manifest_fingerprint: str


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _load_json(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read JSON file: {path}") from error
    if not raw or len(raw) > maximum_bytes:
        raise ValueError("JSON file has invalid byte size")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("JSON file is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return cast(dict[str, Any], value)


def _exact(value: Mapping[str, Any], fields: frozenset[str], location: str) -> None:
    if frozenset(value) != fields:
        raise ValueError(f"{location} fields are invalid")


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return cast(int, value)


def _sha256(value: Any, location: str) -> str:
    parsed = _string(value, location)
    if _SHA256.fullmatch(parsed) is None:
        raise ValueError(f"{location} must be a sha256 fingerprint")
    return parsed


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return "sha256:" + artifact_fingerprint(value)


def load_target_service_control_spec(path: Path) -> TargetServiceControlSpec:
    """Load the closed target-service manifest."""

    manifest = _load_json(path, maximum_bytes=64_000)
    _exact(manifest, _MANIFEST_FIELDS, "manifest")
    if manifest.get("control_version") != TARGET_SERVICE_CONTROL_VERSION:
        raise ValueError("manifest control version is unsupported")
    if manifest.get("evidence_boundary") != TARGET_SERVICE_EVIDENCE_BOUNDARY:
        raise ValueError("manifest evidence boundary drift")
    checked_at = _string(manifest.get("checked_at"), "manifest.checked_at")
    try:
        date.fromisoformat(checked_at)
    except ValueError as error:
        raise ValueError("manifest.checked_at must be an ISO date") from error
    checkpoint_fingerprint = _sha256(
        manifest.get("checkpoint_manifest_fingerprint"),
        "manifest.checkpoint_manifest_fingerprint",
    )
    model_id = _string(manifest.get("model_id"), "manifest.model_id")
    revision = _string(manifest.get("revision"), "manifest.revision")
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("manifest.revision must be a commit id")
    expected_model_class = _string(
        manifest.get("expected_model_class"), "manifest.expected_model_class"
    )
    expected_model_type = _string(
        manifest.get("expected_model_type"), "manifest.expected_model_type"
    )
    maximum_prompt_tokens = _positive_int(
        manifest.get("maximum_prompt_tokens"), "manifest.maximum_prompt_tokens"
    )
    max_tokens = _positive_int(manifest.get("max_tokens"), "manifest.max_tokens")
    raw_messages = manifest.get("messages")
    if (
        not isinstance(raw_messages, list)
        or not raw_messages
        or len(raw_messages) > 8
    ):
        raise ValueError("manifest.messages is invalid")
    messages: list[ChatMessage] = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, Mapping):
            raise ValueError(f"manifest.messages[{index}] must be an object")
        _exact(raw_message, frozenset({"role", "content"}), "manifest.message")
        role = _string(raw_message.get("role"), "manifest.message.role")
        content = _string(raw_message.get("content"), "manifest.message.content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("manifest message role is unsupported")
        messages.append(ChatMessage(role, content))
    if messages[-1].role != "user":
        raise ValueError("manifest final message must have role user")
    expected_prompt_token_count = _positive_int(
        manifest.get("expected_prompt_token_count"),
        "manifest.expected_prompt_token_count",
    )
    raw_completion_ids = manifest.get("expected_completion_token_ids")
    if (
        not isinstance(raw_completion_ids, list)
        or len(raw_completion_ids) != max_tokens
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_completion_ids
        )
    ):
        raise ValueError("manifest expected completion token ids are invalid")
    expected_text_fingerprint = _sha256(
        manifest.get("expected_completion_text_fingerprint"),
        "manifest.expected_completion_text_fingerprint",
    )
    expected_finish_reason = _string(
        manifest.get("expected_finish_reason"), "manifest.expected_finish_reason"
    )
    if expected_finish_reason not in {"stop", "length"}:
        raise ValueError("manifest finish reason is invalid")
    raw_runtime = manifest.get("reviewed_runtime")
    if not isinstance(raw_runtime, Mapping):
        raise ValueError("manifest.reviewed_runtime must be an object")
    _exact(raw_runtime, _RUNTIME_FIELDS, "manifest.reviewed_runtime")
    reviewed_runtime = {
        name: _string(raw_runtime.get(name), f"manifest.reviewed_runtime.{name}")
        for name in sorted(_RUNTIME_FIELDS)
    }
    return TargetServiceControlSpec(
        checked_at=checked_at,
        checkpoint_manifest_fingerprint=checkpoint_fingerprint,
        model_id=model_id,
        revision=revision,
        expected_model_class=expected_model_class,
        expected_model_type=expected_model_type,
        maximum_prompt_tokens=maximum_prompt_tokens,
        max_tokens=max_tokens,
        messages=tuple(messages),
        expected_prompt_token_count=expected_prompt_token_count,
        expected_completion_token_ids=tuple(cast(list[int], raw_completion_ids)),
        expected_completion_text_fingerprint=expected_text_fingerprint,
        expected_finish_reason=expected_finish_reason,
        reviewed_runtime=reviewed_runtime,
        manifest_fingerprint=_canonical_sha256(manifest),
    )


def _runtime_versions() -> dict[str, str]:
    import torch
    import transformers

    return {
        "httpx": importlib.metadata.version("httpx"),
        "python": platform.python_version(),
        "starlette": importlib.metadata.version("starlette"),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "uvicorn": importlib.metadata.version("uvicorn"),
    }


def _validate_checkpoint_binding(
    spec: TargetServiceControlSpec,
    checkpoint_spec: CheckpointControlSpec,
) -> None:
    if checkpoint_spec.manifest_fingerprint != spec.checkpoint_manifest_fingerprint:
        raise ValueError("checkpoint manifest fingerprint drift")
    if (
        checkpoint_spec.model_id != spec.model_id
        or checkpoint_spec.revision != spec.revision
        or checkpoint_spec.expected_model_class != spec.expected_model_class
        or checkpoint_spec.expected_model_type != spec.expected_model_type
    ):
        raise ValueError("checkpoint identity drift")
    checkpoint_messages = tuple(
        ChatMessage(message.role, message.content) for message in checkpoint_spec.messages
    )
    if checkpoint_messages != spec.messages or checkpoint_spec.max_new_tokens != (
        spec.max_tokens
    ):
        raise ValueError("checkpoint workload drift")


def _backend_fingerprint(spec: TargetServiceControlSpec) -> str:
    return _canonical_sha256(
        {
            "service_manifest": spec.manifest_fingerprint,
            "checkpoint_manifest": spec.checkpoint_manifest_fingerprint,
            "backend": "transformers.GenerationMixin.generate/cpu/fp32/eager",
        }
    )


def load_target_backend(
    spec: TargetServiceControlSpec,
    checkpoint_spec: CheckpointControlSpec,
    *,
    local_files_only: bool,
) -> tuple[TransformersCPUBackend, dict[str, Any]]:
    """Verify selected bytes, load Qwen, and return a frozen CPU backend."""

    import torch
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _validate_checkpoint_binding(spec, checkpoint_spec)
    runtime = _runtime_versions()
    if runtime != dict(spec.reviewed_runtime):
        raise ValueError("runtime version drift")
    snapshot_directory = download_checkpoint_snapshot(
        checkpoint_spec, local_files_only=local_files_only
    )
    snapshot = verify_checkpoint_snapshot(checkpoint_spec, snapshot_directory)
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
    if type(model).__name__ != spec.expected_model_class:
        raise ValueError("loaded model class drift")
    if getattr(model.config, "model_type", None) != spec.expected_model_type:
        raise ValueError("loaded model type drift")
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError("loaded parameter count drift")
    if {str(parameter.dtype) for parameter in model.parameters()} != {"torch.float32"}:
        raise ValueError("loaded model dtype drift")
    backend = TransformersCPUBackend(
        model_id=spec.model_id,
        backend_fingerprint=_backend_fingerprint(spec),
        model=model,
        tokenizer=tokenizer,
        maximum_prompt_tokens=spec.maximum_prompt_tokens,
    )
    artifacts = _artifact_projection(snapshot)
    metadata: dict[str, Any] = {
        "runtime": runtime,
        "artifacts": artifacts,
        "model_class": type(model).__name__,
        "model_type": getattr(model.config, "model_type", None),
        "parameter_count": parameter_count,
        "parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
    }
    return backend, metadata


def _artifact_projection(snapshot: VerifiedCheckpointSnapshot) -> dict[str, Any]:
    files = [dict(item) for item in snapshot.files]
    return {
        "selected_file_count": len(files),
        "selected_total_bytes": sum(cast(int, item["size_bytes"]) for item in files),
        "files": files,
    }


def _authorized(request: Request, token: str) -> bool:
    expected = b"authorization"
    headers = cast(Sequence[tuple[bytes, bytes]], request.scope["headers"])
    values = [value.decode("latin-1") for key, value in headers if key == expected]
    if len(values) != 1:
        return False
    scheme, separator, supplied = values[0].partition(" ")
    return (
        bool(separator)
        and scheme.lower() == "bearer"
        and hmac.compare_digest(supplied, token)
    )


def build_target_app(
    backend: TransformersCPUBackend,
    metadata: Mapping[str, Any],
    *,
    bearer_token: str,
    maximum_new_tokens: int,
) -> Starlette:
    """Build the reference API plus one protected content-free control endpoint."""

    app = build_reference_app(
        backend,
        bearer_token=bearer_token,
        maximum_new_tokens=maximum_new_tokens,
    )

    async def audit(request: Request) -> Response:
        if not _authorized(request, bearer_token):
            return Response(status_code=401)
        payload = {
            "service": app.state.reference_service.audit_projection(),
            "load": copy.deepcopy(dict(metadata)),
        }
        return Response(canonical_json_bytes(payload), media_type="application/json")

    app.router.routes.append(Route(CONTROL_AUDIT_PATH, audit, methods=["GET"]))
    return app


def serve_target(
    host: str,
    port: int,
    *,
    service_manifest: Path,
    checkpoint_manifest: Path,
    bearer_token: str,
    local_files_only: bool,
) -> int:
    """Load the target backend and serve it on IPv4 loopback."""

    if host != "127.0.0.1":
        raise ValueError("target service control only permits IPv4 loopback")
    if not 0 < port < 65_536:
        raise ValueError("port must be between 1 and 65535")
    spec = load_target_service_control_spec(service_manifest)
    checkpoint_spec = load_checkpoint_control_spec(checkpoint_manifest)
    backend, metadata = load_target_backend(
        spec,
        checkpoint_spec,
        local_files_only=local_files_only,
    )
    logging.disable(logging.CRITICAL)
    server = uvicorn.Server(
        uvicorn.Config(
            build_target_app(
                backend,
                metadata,
                bearer_token=bearer_token,
                maximum_new_tokens=spec.max_tokens,
            ),
            host=host,
            port=port,
            access_log=False,
            log_level="critical",
            lifespan="off",
            ws="none",
        )
    )
    asyncio.run(server.serve())
    return 0


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return cast(tuple[str, int], candidate.getsockname())[1]


async def _wait_ready(
    client: httpx.AsyncClient,
    base_url: str,
    process: subprocess.Popen[bytes],
    headers: Mapping[str, str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("target service exited before readiness")
        try:
            response = await client.get(f"{base_url}{HEALTH_PATH}", headers=headers)
            if response.status_code == 200:
                return decode_strict_json_object(response.content)
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.1)
    raise TimeoutError("target service readiness timed out")


def _request_body(spec: TargetServiceControlSpec, *, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": spec.model_id,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in spec.messages
        ],
        "max_tokens": spec.max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def _extract_nonstream(response: httpx.Response) -> dict[str, Any]:
    if response.status_code != 200:
        raise ValueError("non-streaming request failed")
    payload = decode_strict_json_object(response.content)
    choices = payload.get("choices")
    usage = payload.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(
        choices[0], Mapping
    ):
        raise ValueError("non-streaming choices are invalid")
    choice = cast(Mapping[str, Any], choices[0])
    message = choice.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        raise ValueError("non-streaming assistant message is invalid")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("non-streaming content is invalid")
    if not isinstance(usage, Mapping):
        raise ValueError("non-streaming usage is invalid")
    return {
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "usage": dict(usage),
        "system_fingerprint": payload.get("system_fingerprint"),
    }


async def _extract_stream(response: httpx.Response) -> dict[str, Any]:
    if response.status_code != 200 or not response.headers.get(
        "content-type", ""
    ).startswith("text/event-stream"):
        raise ValueError("streaming request failed")
    content = ""
    role_seen = False
    finish_reason: Any = None
    usage: dict[str, Any] | None = None
    system_fingerprints: set[str] = set()
    content_delta_count = 0
    done = False
    async for line in response.aiter_lines():
        event = parse_sse_data_line(line)
        if event is None:
            continue
        if event is STREAM_FINISHED:
            done = True
            continue
        if not isinstance(event, dict):
            raise ValueError("stream event is invalid")
        fingerprint = event.get("system_fingerprint")
        if isinstance(fingerprint, str):
            system_fingerprints.add(fingerprint)
        choices = event.get("choices") or []
        if choices:
            if not isinstance(choices, list) or not isinstance(choices[0], Mapping):
                raise ValueError("stream choices are invalid")
            choice = cast(Mapping[str, Any], choices[0])
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                raise ValueError("stream delta is invalid")
            if delta.get("role") == "assistant":
                role_seen = True
            fragment = delta.get("content")
            if fragment is not None:
                if not isinstance(fragment, str) or not fragment:
                    raise ValueError("stream content fragment is invalid")
                content += fragment
                content_delta_count += 1
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
        if event.get("usage") is not None:
            if not isinstance(event["usage"], Mapping):
                raise ValueError("stream usage is invalid")
            usage = dict(cast(Mapping[str, Any], event["usage"]))
    if (
        not done
        or not role_seen
        or not content
        or usage is None
        or len(system_fingerprints) != 1
    ):
        raise ValueError("stream did not complete the reviewed protocol")
    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
        "system_fingerprint": next(iter(system_fingerprints)),
        "content_delta_count": content_delta_count,
        "done": done,
    }


async def _run_live_control_async(
    spec: TargetServiceControlSpec,
    checkpoint_spec: CheckpointControlSpec,
    *,
    service_manifest: Path,
    checkpoint_manifest: Path,
    local_files_only: bool,
    request_timeout_seconds: float,
    server_start_timeout_seconds: float,
) -> dict[str, Any]:
    _validate_checkpoint_binding(spec, checkpoint_spec)
    host = "127.0.0.1"
    port = _reserve_port()
    base_url = f"http://{host}:{port}"
    token = secrets.token_urlsafe(32)
    environment = os.environ.copy()
    environment[CONTROL_TOKEN_ENV] = token
    command = [
        sys.executable,
        "-m",
        "about_llm.inference.target_service_control",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
        "--service-manifest",
        str(service_manifest.resolve()),
        "--checkpoint-manifest",
        str(checkpoint_manifest.resolve()),
    ]
    if local_files_only:
        command.append("--local-files-only")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    stdout = b""
    stderr = b""
    try:
        auth = {"Authorization": f"Bearer {token}"}
        timeout = httpx.Timeout(request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            health = await _wait_ready(
                client,
                base_url,
                process,
                auth,
                timeout_seconds=server_start_timeout_seconds,
            )
            unauthorized = await client.get(f"{base_url}{MODELS_PATH}")
            models = await client.get(f"{base_url}{MODELS_PATH}", headers=auth)
            if unauthorized.status_code != 401 or models.status_code != 200:
                raise ValueError("model discovery authentication control failed")
            model_payload = decode_strict_json_object(models.content)
            data = model_payload.get("data")
            if (
                not isinstance(data, list)
                or len(data) != 1
                or not isinstance(data[0], Mapping)
                or data[0].get("id") != spec.model_id
            ):
                raise ValueError("model discovery payload drift")

            invalid_body = _request_body(spec, stream=False)
            invalid_body["unexpected"] = True
            unknown_field = await client.post(
                f"{base_url}{CHAT_COMPLETIONS_PATH}", headers=auth, json=invalid_body
            )
            wrong_model_body = _request_body(spec, stream=False)
            wrong_model_body["model"] = "unavailable/model"
            wrong_model = await client.post(
                f"{base_url}{CHAT_COMPLETIONS_PATH}",
                headers=auth,
                json=wrong_model_body,
            )
            if unknown_field.status_code != 422 or wrong_model.status_code != 404:
                raise ValueError("request rejection controls failed")

            nonstream_response = await client.post(
                f"{base_url}{CHAT_COMPLETIONS_PATH}",
                headers=auth,
                json=_request_body(spec, stream=False),
            )
            nonstream = _extract_nonstream(nonstream_response)
            async with client.stream(
                "POST",
                f"{base_url}{CHAT_COMPLETIONS_PATH}",
                headers=auth,
                json=_request_body(spec, stream=True),
            ) as stream_response:
                stream = await _extract_stream(stream_response)
            audit_response = await client.get(
                f"{base_url}{CONTROL_AUDIT_PATH}", headers=auth
            )
            if audit_response.status_code != 200:
                raise ValueError("control audit endpoint failed")
            audit = decode_strict_json_object(audit_response.content)

        expected_usage = {
            "prompt_tokens": spec.expected_prompt_token_count,
            "completion_tokens": len(spec.expected_completion_token_ids),
            "total_tokens": (
                spec.expected_prompt_token_count
                + len(spec.expected_completion_token_ids)
            ),
        }
        expected_fingerprint = _backend_fingerprint(spec)
        for result in (nonstream, stream):
            actual_text_fingerprint = _canonical_sha256({"text": result["content"]})
            if actual_text_fingerprint != spec.expected_completion_text_fingerprint:
                raise ValueError("completion text fingerprint drift")
            if result["finish_reason"] != spec.expected_finish_reason:
                raise ValueError("completion finish reason drift")
            if result["usage"] != expected_usage:
                raise ValueError("completion usage drift")
            if result["system_fingerprint"] != expected_fingerprint:
                raise ValueError("backend fingerprint drift")
        if nonstream["content"] != stream["content"]:
            raise ValueError("non-stream and stream content differ")

        service_audit = audit.get("service")
        load_audit = audit.get("load")
        if not isinstance(service_audit, Mapping) or not isinstance(load_audit, Mapping):
            raise ValueError("control audit payload is invalid")
        backend_audit = service_audit.get("backend")
        if not isinstance(backend_audit, Mapping):
            raise ValueError("backend audit payload is invalid")
        last_execution = backend_audit.get("last_execution")
        if not isinstance(last_execution, Mapping):
            raise ValueError("backend execution audit is missing")
        if (
            service_audit.get("accepted_requests") != 2
            or service_audit.get("stream_requests") != 1
            or service_audit.get("nonstream_requests") != 1
            or service_audit.get("failed_backend_requests") != 0
            or backend_audit.get("generation_call_count") != 2
            or last_execution.get("prompt_token_count")
            != spec.expected_prompt_token_count
            or last_execution.get("completion_token_ids")
            != list(spec.expected_completion_token_ids)
            or last_execution.get("completion_text_fingerprint")
            != spec.expected_completion_text_fingerprint
            or last_execution.get("finish_reason") != spec.expected_finish_reason
        ):
            raise ValueError("backend execution audit drift")
        if health.get("backend_fingerprint") != expected_fingerprint:
            raise ValueError("readiness backend identity drift")
        if process.poll() is not None:
            raise RuntimeError("target service exited during control")
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=10.0)

    artifacts = cast(Mapping[str, Any], load_audit["artifacts"])
    runtime = cast(Mapping[str, Any], load_audit["runtime"])
    report: dict[str, Any] = {
        "report_version": TARGET_SERVICE_REPORT_VERSION,
        "control_manifest_fingerprint": spec.manifest_fingerprint,
        "checkpoint_manifest_fingerprint": spec.checkpoint_manifest_fingerprint,
        "checked_at": spec.checked_at,
        "source": {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "checkpoint_files_verified_before_load": True,
            "trust_remote_code": False,
        },
        "artifacts": copy.deepcopy(dict(artifacts)),
        "runtime": copy.deepcopy(dict(runtime)),
        "network": {
            "scheme": "http",
            "address_scope": "IPv4 loopback",
            "real_tcp_http": True,
            "tls": False,
            "endpoint": CHAT_COMPLETIONS_PATH,
        },
        "api": {
            "service_version": OPENAI_REFERENCE_SERVICE_VERSION,
            "models_endpoint_executed": True,
            "chat_nonstream_executed": True,
            "chat_sse_executed": True,
            "unauthorized_status": 401,
            "unknown_field_status": 422,
            "wrong_model_status": 404,
            "sse_done_observed": bool(stream["done"]),
            "stream_usage_observed": True,
            "raw_request_or_response_published": False,
        },
        "execution": {
            "model_class": load_audit["model_class"],
            "model_type": load_audit["model_type"],
            "parameter_count": load_audit["parameter_count"],
            "parameters_frozen": load_audit["parameters_frozen"],
            "framework_generate_call_count": backend_audit["generation_call_count"],
            "prompt_token_count": spec.expected_prompt_token_count,
            "completion_token_ids": list(spec.expected_completion_token_ids),
            "completion_text_fingerprint": spec.expected_completion_text_fingerprint,
            "finish_reason": spec.expected_finish_reason,
            "nonstream_stream_content_match": True,
            "nonstream_stream_usage_match": True,
            "stream_content_delta_count": stream["content_delta_count"],
            "generation_completed_before_sse_emission": True,
        },
        "server_process": {
            "subprocess_used": True,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_stderr_empty": not stdout and not stderr,
        },
        "scope": {
            "target_checkpoint_weights_loaded": True,
            "transformers_generate_executed": True,
            "real_ipv4_loopback_tcp_http_executed": True,
            "vllm_executed": False,
            "cuda_executed": False,
            "incremental_model_decode_streaming_proven": False,
            "client_disconnect_cancellation_proven": False,
            "tls_reverse_proxy_or_remote_network_proven": False,
            "oauth_jwt_iam_or_business_authorization_proven": False,
            "multiple_workers_or_distributed_serving_proven": False,
            "performance_capacity_or_slo_proven": False,
            "full_openai_api_compatibility_proven": False,
            "model_quality_proven": False,
            "publisher_authenticated_by_signature": False,
            "license_compatibility_proven": False,
            "production_safety_proven": False,
            "verification_to_loader_reopen_toctou_eliminated": False,
            "report_fingerprint_proves_authenticity": False,
        },
    }
    report["report_fingerprint"] = _canonical_sha256(report)
    return report


def run_live_target_service_control(
    service_manifest: Path,
    checkpoint_manifest: Path,
    *,
    local_files_only: bool = False,
    request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    server_start_timeout_seconds: float = SERVER_START_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Execute the real target checkpoint service control."""

    if request_timeout_seconds <= 0 or server_start_timeout_seconds <= 0:
        raise ValueError("timeouts must be positive")
    spec = load_target_service_control_spec(service_manifest)
    checkpoint_spec = load_checkpoint_control_spec(checkpoint_manifest)
    return asyncio.run(
        _run_live_control_async(
            spec,
            checkpoint_spec,
            service_manifest=service_manifest,
            checkpoint_manifest=checkpoint_manifest,
            local_files_only=local_files_only,
            request_timeout_seconds=request_timeout_seconds,
            server_start_timeout_seconds=server_start_timeout_seconds,
        )
    )


def verify_recorded_target_service_report(
    spec: TargetServiceControlSpec,
    checkpoint_spec: CheckpointControlSpec,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the closed recorded report without loading model weights."""

    _validate_checkpoint_binding(spec, checkpoint_spec)
    _validate_report_schema(report)
    supplied_fingerprint = _sha256(
        report.get("report_fingerprint"), "report.report_fingerprint"
    )
    unsigned = {key: copy.deepcopy(value) for key, value in report.items()}
    del unsigned["report_fingerprint"]
    if not hmac.compare_digest(supplied_fingerprint, _canonical_sha256(unsigned)):
        raise ValueError("recorded report fingerprint mismatch")
    if (
        report.get("report_version") != TARGET_SERVICE_REPORT_VERSION
        or report.get("control_manifest_fingerprint") != spec.manifest_fingerprint
        or report.get("checkpoint_manifest_fingerprint")
        != spec.checkpoint_manifest_fingerprint
        or report.get("checked_at") != spec.checked_at
    ):
        raise ValueError("recorded report top-level identity drift")
    source = cast(Mapping[str, Any], report["source"])
    if source != {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "checkpoint_files_verified_before_load": True,
        "trust_remote_code": False,
    }:
        raise ValueError("recorded report source drift")
    expected_files = [
        {
            "filename": item.filename,
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
            "verified": True,
        }
        for item in checkpoint_spec.files
    ]
    expected_total_bytes = sum(item.size_bytes for item in checkpoint_spec.files)
    artifacts = cast(Mapping[str, Any], report["artifacts"])
    if artifacts != {
        "selected_file_count": len(expected_files),
        "selected_total_bytes": expected_total_bytes,
        "files": expected_files,
    }:
        raise ValueError("recorded report artifact drift")
    if report["runtime"] != dict(spec.reviewed_runtime):
        raise ValueError("recorded report runtime drift")
    if report["network"] != {
        "scheme": "http",
        "address_scope": "IPv4 loopback",
        "real_tcp_http": True,
        "tls": False,
        "endpoint": CHAT_COMPLETIONS_PATH,
    }:
        raise ValueError("recorded report network drift")
    if report["api"] != {
        "service_version": OPENAI_REFERENCE_SERVICE_VERSION,
        "models_endpoint_executed": True,
        "chat_nonstream_executed": True,
        "chat_sse_executed": True,
        "unauthorized_status": 401,
        "unknown_field_status": 422,
        "wrong_model_status": 404,
        "sse_done_observed": True,
        "stream_usage_observed": True,
        "raw_request_or_response_published": False,
    }:
        raise ValueError("recorded report API drift")
    execution = cast(Mapping[str, Any], report["execution"])
    if execution != {
        "model_class": spec.expected_model_class,
        "model_type": spec.expected_model_type,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameters_frozen": True,
        "framework_generate_call_count": 2,
        "prompt_token_count": spec.expected_prompt_token_count,
        "completion_token_ids": list(spec.expected_completion_token_ids),
        "completion_text_fingerprint": spec.expected_completion_text_fingerprint,
        "finish_reason": spec.expected_finish_reason,
        "nonstream_stream_content_match": True,
        "nonstream_stream_usage_match": True,
        "stream_content_delta_count": len(spec.expected_completion_token_ids),
        "generation_completed_before_sse_emission": True,
    }:
        raise ValueError("recorded report execution drift")
    process = cast(Mapping[str, Any], report["server_process"])
    if process != {
        "subprocess_used": True,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_stderr_empty": True,
    }:
        raise ValueError("recorded report process drift")
    scope = cast(Mapping[str, Any], report["scope"])
    expected_true = {
        "target_checkpoint_weights_loaded",
        "transformers_generate_executed",
        "real_ipv4_loopback_tcp_http_executed",
    }
    if any(scope[name] is not (name in expected_true) for name in _SCOPE_FIELDS):
        raise ValueError("recorded report scope drift")
    return copy.deepcopy(dict(report))


def load_and_verify_recorded_target_service_report(
    service_manifest: Path,
    checkpoint_manifest: Path,
    report_path: Path,
) -> dict[str, Any]:
    spec = load_target_service_control_spec(service_manifest)
    checkpoint_spec = load_checkpoint_control_spec(checkpoint_manifest)
    report = _load_json(report_path, maximum_bytes=MAX_RECORDED_REPORT_BYTES)
    return verify_recorded_target_service_report(spec, checkpoint_spec, report)


def _validate_report_schema(report: Mapping[str, Any]) -> None:
    _exact(report, _REPORT_FIELDS, "report")
    mappings = (
        ("source", _SOURCE_FIELDS),
        ("artifacts", _ARTIFACT_FIELDS),
        ("runtime", _RUNTIME_FIELDS),
        ("network", _NETWORK_FIELDS),
        ("api", _API_FIELDS),
        ("execution", _EXECUTION_FIELDS),
        ("server_process", _PROCESS_FIELDS),
        ("scope", _SCOPE_FIELDS),
    )
    for name, fields in mappings:
        value = report.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"report.{name} must be an object")
        _exact(value, fields, f"report.{name}")
    artifacts = cast(Mapping[str, Any], report["artifacts"])
    files = artifacts.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("report.artifacts.files must be a non-empty array")
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("report artifact file must be an object")
        _exact(item, _FILE_FIELDS, "report.artifact.file")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, required=True)
    serve_parser.add_argument("--service-manifest", type=Path, required=True)
    serve_parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    serve_parser.add_argument("--local-files-only", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--service-manifest", type=Path, required=True)
    run_parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    run_parser.add_argument("--local-files-only", action="store_true")

    verify_parser = subparsers.add_parser("verify-recorded")
    verify_parser.add_argument("--service-manifest", type=Path, required=True)
    verify_parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    verify_parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        token = os.environ.get(CONTROL_TOKEN_ENV, "")
        if not token:
            raise ValueError("target service control token is missing")
        return serve_target(
            args.host,
            args.port,
            service_manifest=args.service_manifest,
            checkpoint_manifest=args.checkpoint_manifest,
            bearer_token=token,
            local_files_only=args.local_files_only,
        )
    if args.command == "run":
        report = run_live_target_service_control(
            args.service_manifest,
            args.checkpoint_manifest,
            local_files_only=args.local_files_only,
        )
    else:
        report = load_and_verify_recorded_target_service_report(
            args.service_manifest,
            args.checkpoint_manifest,
            args.report,
        )
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
