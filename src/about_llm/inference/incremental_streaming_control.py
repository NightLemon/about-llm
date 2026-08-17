"""Real-loopback incremental SSE and cooperative disconnect-cancellation control."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hmac
import importlib.metadata
import logging
import os
import platform
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

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
    ChatCompletionRequest,
    IncrementalTokenDelta,
    build_incremental_reference_app,
    decode_strict_json_object,
)
from about_llm.inference.sse import STREAM_FINISHED, parse_sse_data_line
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

INCREMENTAL_STREAMING_CONTROL_VERSION: Final = (
    "about-llm.incremental-streaming-control.v1"
)
INCREMENTAL_STREAMING_REPORT_VERSION: Final = (
    "about-llm.incremental-streaming-control-report.v1"
)
CONTROL_AUDIT_PATH: Final = "/control/incremental-audit"
CONTROL_TOKEN_ENV: Final = "ABOUT_LLM_INCREMENTAL_STREAM_TOKEN"
MODEL_ID: Final = "about-llm/scripted-incremental-backend"
COMPLETE_PROMPT: Final = "run-complete-control"
CANCEL_PROMPT: Final = "run-cancel-control"
COMPLETE_PROMPT_TOKENS: Final = 5
CANCEL_PROMPT_TOKENS: Final = 4
COMPLETE_PLAN: Final = (
    (101, "甲", None),
    (102, "🙂", None),
    (103, "终", "stop"),
)
CANCEL_FIRST: Final = (201, "首", None)
MAX_NEW_TOKENS: Final = 8
MAX_RECORDED_REPORT_BYTES: Final = 64_000
REQUEST_TIMEOUT_SECONDS: Final = 10.0
START_TIMEOUT_SECONDS: Final = 30.0

INCREMENTAL_STREAMING_EVIDENCE_BOUNDARY: Final = (
    "This control uses an authored asynchronous pseudo-token backend and a real Uvicorn "
    "subprocess over IPv4 loopback TCP/HTTP. It proves that one client observes content "
    "before that backend completes, and that explicitly closing a second response causes "
    "the ASGI streaming task and cooperative backend iterator to observe cancellation "
    "before later authored deltas are produced. It does not execute a tokenizer, model "
    "forward, Transformers generation thread, vLLM, CUDA, a GPU kernel, KV allocation, "
    "TLS, a reverse proxy, OAuth/JWT/IAM, a remote client, multiple workers, network "
    "packet capture, provider billing, or performance/quality/SLO measurement. It does "
    "not prove that a blocking thread, process, remote provider, or model runtime stops "
    "work or releases resources after disconnect, and its unkeyed report fingerprint "
    "does not authenticate the recorder."
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return "sha256:" + artifact_fingerprint(value)


SCRIPTED_BACKEND_FINGERPRINT: Final = _canonical_sha256(
    {
        "control_version": INCREMENTAL_STREAMING_CONTROL_VERSION,
        "model_id": MODEL_ID,
        "complete_plan": [list(item) for item in COMPLETE_PLAN],
        "cancel_first": list(CANCEL_FIRST),
        "backend": "authored-cooperative-async-iterator",
    }
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "report_version",
        "checked_at",
        "implementation",
        "backend_fingerprint",
        "runtime",
        "transport",
        "protocol",
        "complete_stream",
        "disconnect_stream",
        "audit",
        "server_process",
        "scope",
        "evidence_boundary",
        "report_fingerprint",
    }
)
_RUNTIME_FIELDS = frozenset({"python", "httpx", "starlette", "uvicorn"})
_TRANSPORT_FIELDS = frozenset(
    {
        "scheme",
        "address_scope",
        "real_tcp_http",
        "server_subprocess",
        "tls",
        "bearer_header_gate",
        "unauthorized_audit_status",
        "raw_bearer_or_session_published",
    }
)
_PROTOCOL_FIELDS = frozenset(
    {
        "service_version",
        "endpoint",
        "models_endpoint_executed",
        "streaming_only_profile",
        "usage_event_required_for_complete_case",
        "sse_done_required_for_complete_case",
        "raw_request_or_response_published",
    }
)
_COMPLETE_FIELDS = frozenset(
    {
        "client_content_delta_count",
        "client_content_fingerprint",
        "client_finish_reason",
        "client_usage",
        "client_sse_done_observed",
        "backend_completion_token_ids",
        "backend_completed",
    }
)
_DISCONNECT_FIELDS = frozenset(
    {
        "client_content_delta_count_before_close",
        "client_content_fingerprint_before_close",
        "preclose_service_active_streams",
        "preclose_service_cancelled_streams",
        "preclose_backend_first_delta_emitted",
        "preclose_backend_completed",
        "client_response_explicitly_closed",
        "postclose_service_active_streams",
        "postclose_service_cancelled_streams",
        "postclose_backend_asyncio_cancelled_error_observed",
        "postclose_backend_iterator_closed",
        "postclose_backend_completed",
        "postclose_backend_emitted_token_ids",
    }
)
_AUDIT_FIELDS = frozenset(
    {
        "accepted_requests",
        "stream_requests",
        "nonstream_requests",
        "completed_incremental_streams",
        "cancelled_incremental_streams",
        "failed_backend_requests",
        "single_process_admission_limit",
        "backend_stream_call_count",
    }
)
_PROCESS_FIELDS = frozenset(
    {"subprocess_used", "stdout_bytes", "stderr_bytes", "stdout_stderr_empty"}
)
_SCOPE_FIELDS = frozenset(
    {
        "authored_async_backend_executed",
        "real_ipv4_loopback_tcp_http_executed",
        "content_observed_before_backend_completion",
        "client_disconnect_cancelled_asgi_stream_task",
        "cooperative_async_backend_cancellation_observed",
        "later_authored_deltas_suppressed_after_disconnect",
        "tokenizer_or_model_forward_executed",
        "transformers_generation_thread_cancellation_proven",
        "vllm_or_cuda_executed",
        "blocking_thread_or_process_termination_proven",
        "kv_or_gpu_resource_release_proven",
        "remote_provider_cancellation_or_billing_proven",
        "tls_proxy_remote_or_multiworker_proven",
        "performance_quality_or_slo_proven",
        "full_openai_api_compatibility_proven",
        "report_fingerprint_proves_authenticity",
    }
)

REVIEWED_RUNTIME: Final = {
    "python": "3.12.10",
    "httpx": "0.28.1",
    "starlette": "0.41.3",
    "uvicorn": "0.52.1",
}


class ScriptedIncrementalBackend:
    """Authored cooperative backend with one normal and one blocked stream."""

    model_id = MODEL_ID
    backend_fingerprint = SCRIPTED_BACKEND_FINGERPRINT

    def __init__(self) -> None:
        self.stream_call_count = 0
        self.active_streams = 0
        self.complete_started = False
        self.complete_completed = False
        self.complete_emitted_token_ids: list[int] = []
        self.cancel_started = False
        self.cancel_first_delta_emitted = False
        self.cancel_asyncio_cancelled_error_observed = False
        self.cancel_iterator_closed = False
        self.cancel_completed = False
        self.cancel_emitted_token_ids: list[int] = []

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[IncrementalTokenDelta]:
        if len(request.messages) != 1 or request.messages[0].role != "user":
            raise ValueError("scripted backend requires one user message")
        prompt = request.messages[0].content
        self.stream_call_count += 1
        self.active_streams += 1
        try:
            if prompt == COMPLETE_PROMPT:
                if request.max_tokens != len(COMPLETE_PLAN):
                    raise ValueError("complete control max_tokens drift")
                self.complete_started = True
                for token_id, text_delta, finish_reason in COMPLETE_PLAN:
                    self.complete_emitted_token_ids.append(token_id)
                    yield IncrementalTokenDelta(
                        text_delta=text_delta,
                        token_id=token_id,
                        prompt_token_count=COMPLETE_PROMPT_TOKENS,
                        finish_reason=finish_reason,
                    )
                    await asyncio.sleep(0)
                self.complete_completed = True
                return
            if prompt != CANCEL_PROMPT or request.max_tokens != MAX_NEW_TOKENS:
                raise ValueError("scripted backend workload drift")
            self.cancel_started = True
            token_id, text_delta, finish_reason = CANCEL_FIRST
            self.cancel_emitted_token_ids.append(token_id)
            self.cancel_first_delta_emitted = True
            yield IncrementalTokenDelta(
                text_delta=text_delta,
                token_id=token_id,
                prompt_token_count=CANCEL_PROMPT_TOKENS,
                finish_reason=finish_reason,
            )
            await asyncio.Event().wait()
            self.cancel_completed = True  # pragma: no cover - cancellation invariant
        except asyncio.CancelledError:
            self.cancel_asyncio_cancelled_error_observed = True
            raise
        finally:
            if prompt == CANCEL_PROMPT:
                self.cancel_iterator_closed = True
            self.active_streams -= 1

    def audit_projection(self) -> Mapping[str, Any]:
        return {
            "implementation": "authored-cooperative-async-iterator",
            "stream_call_count": self.stream_call_count,
            "active_streams": self.active_streams,
            "complete_started": self.complete_started,
            "complete_completed": self.complete_completed,
            "complete_emitted_token_ids": list(self.complete_emitted_token_ids),
            "cancel_started": self.cancel_started,
            "cancel_first_delta_emitted": self.cancel_first_delta_emitted,
            "cancel_asyncio_cancelled_error_observed": (
                self.cancel_asyncio_cancelled_error_observed
            ),
            "cancel_iterator_closed": self.cancel_iterator_closed,
            "cancel_completed": self.cancel_completed,
            "cancel_emitted_token_ids": list(self.cancel_emitted_token_ids),
        }


def _authorized(request: Request, token: str) -> bool:
    headers = cast(Sequence[tuple[bytes, bytes]], request.scope["headers"])
    values = [
        value.decode("latin-1")
        for key, value in headers
        if key == b"authorization"
    ]
    if len(values) != 1:
        return False
    scheme, separator, supplied = values[0].partition(" ")
    return (
        bool(separator)
        and scheme.lower() == "bearer"
        and hmac.compare_digest(supplied, token)
    )


def build_control_app(
    backend: ScriptedIncrementalBackend,
    *,
    bearer_token: str,
) -> Starlette:
    app = build_incremental_reference_app(
        backend,
        bearer_token=bearer_token,
        maximum_new_tokens=MAX_NEW_TOKENS,
    )

    async def audit(request: Request) -> Response:
        if not _authorized(request, bearer_token):
            return Response(status_code=401)
        payload = app.state.reference_service.audit_projection()
        return Response(
            canonical_json_bytes(payload),
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )

    app.router.routes.append(Route(CONTROL_AUDIT_PATH, audit, methods=["GET"]))
    return app


def serve_control(host: str, port: int, *, bearer_token: str) -> int:
    if host != "127.0.0.1":
        raise ValueError("incremental control only permits IPv4 loopback")
    if not 0 < port < 65_536:
        raise ValueError("port must be between 1 and 65535")
    logging.disable(logging.CRITICAL)
    server = uvicorn.Server(
        uvicorn.Config(
            build_control_app(ScriptedIncrementalBackend(), bearer_token=bearer_token),
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


def _request_body(prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def _wait_ready(
    client: httpx.AsyncClient,
    base_url: str,
    process: subprocess.Popen[bytes],
    headers: Mapping[str, str],
) -> None:
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("incremental control server exited before readiness")
        try:
            response = await client.get(f"{base_url}{HEALTH_PATH}", headers=headers)
            if response.status_code == 200:
                return
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.05)
    raise TimeoutError("incremental control readiness timed out")


async def _consume_complete(response: httpx.Response) -> dict[str, Any]:
    if response.status_code != 200 or not response.headers.get(
        "content-type", ""
    ).startswith("text/event-stream"):
        raise ValueError("complete stream request failed")
    content = ""
    content_delta_count = 0
    role_seen = False
    finish_reason: Any = None
    usage: dict[str, Any] | None = None
    done = False
    fingerprints: set[str] = set()
    async for line in response.aiter_lines():
        event = parse_sse_data_line(line)
        if event is None:
            continue
        if event is STREAM_FINISHED:
            done = True
            continue
        if not isinstance(event, dict):
            raise ValueError("complete stream event is invalid")
        fingerprint = event.get("system_fingerprint")
        if isinstance(fingerprint, str):
            fingerprints.add(fingerprint)
        choices = event.get("choices") or []
        if choices:
            if not isinstance(choices, list) or not isinstance(choices[0], Mapping):
                raise ValueError("complete stream choices are invalid")
            choice = cast(Mapping[str, Any], choices[0])
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                raise ValueError("complete stream delta is invalid")
            if delta.get("role") == "assistant":
                role_seen = True
            fragment = delta.get("content")
            if fragment is not None:
                if not isinstance(fragment, str) or not fragment:
                    raise ValueError("complete stream content is invalid")
                content += fragment
                content_delta_count += 1
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
        raw_usage = event.get("usage")
        if raw_usage is not None:
            if not isinstance(raw_usage, Mapping):
                raise ValueError("complete stream usage is invalid")
            usage = dict(raw_usage)
    if (
        not done
        or not role_seen
        or not content
        or usage is None
        or len(fingerprints) != 1
    ):
        raise ValueError("complete stream did not reach the reviewed terminal state")
    return {
        "content": content,
        "content_delta_count": content_delta_count,
        "finish_reason": finish_reason,
        "usage": usage,
        "done": done,
        "system_fingerprint": next(iter(fingerprints)),
    }


async def _read_cancel_prefix(
    response: httpx.Response,
    *,
    audit_client: httpx.AsyncClient,
    base_url: str,
    headers: Mapping[str, str],
) -> tuple[str, dict[str, Any]]:
    if response.status_code != 200 or not response.headers.get(
        "content-type", ""
    ).startswith("text/event-stream"):
        raise ValueError("cancel stream request failed")
    async for line in response.aiter_lines():
        event = parse_sse_data_line(line)
        if event is None:
            continue
        if event is STREAM_FINISHED:
            raise ValueError("cancel stream completed before client close")
        if not isinstance(event, dict):
            raise ValueError("cancel stream event is invalid")
        choices = event.get("choices") or []
        if not choices:
            continue
        if not isinstance(choices, list) or not isinstance(choices[0], Mapping):
            raise ValueError("cancel stream choices are invalid")
        delta = cast(Mapping[str, Any], choices[0]).get("delta")
        if not isinstance(delta, Mapping):
            raise ValueError("cancel stream delta is invalid")
        fragment = delta.get("content")
        if fragment is None:
            continue
        if not isinstance(fragment, str) or not fragment:
            raise ValueError("cancel stream content is invalid")
        audit_response = await audit_client.get(
            f"{base_url}{CONTROL_AUDIT_PATH}", headers=headers
        )
        if audit_response.status_code != 200:
            raise ValueError("preclose audit request failed")
        return fragment, decode_strict_json_object(audit_response.content)
    raise ValueError("cancel stream ended without content")


async def _wait_cancelled(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Mapping[str, str],
) -> dict[str, Any]:
    deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = await client.get(f"{base_url}{CONTROL_AUDIT_PATH}", headers=headers)
        if response.status_code == 200:
            audit = decode_strict_json_object(response.content)
            backend = audit.get("backend")
            if (
                audit.get("cancelled_incremental_streams") == 1
                and audit.get("active_incremental_streams") == 0
                and isinstance(backend, Mapping)
                and backend.get("cancel_asyncio_cancelled_error_observed") is True
                and backend.get("cancel_iterator_closed") is True
            ):
                return audit
        await asyncio.sleep(0.02)
    raise TimeoutError("server did not observe reviewed disconnect cancellation")


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "httpx": importlib.metadata.version("httpx"),
        "starlette": importlib.metadata.version("starlette"),
        "uvicorn": importlib.metadata.version("uvicorn"),
    }


async def _run_control_async() -> dict[str, Any]:
    host = "127.0.0.1"
    port = _reserve_port()
    base_url = f"http://{host}:{port}"
    token = secrets.token_urlsafe(32)
    environment = os.environ.copy()
    environment[CONTROL_TOKEN_ENV] = token
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "about_llm.inference.incremental_streaming_control",
            "serve",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    stdout = b""
    stderr = b""
    complete: dict[str, Any] | None = None
    preclose: dict[str, Any] | None = None
    final_audit: dict[str, Any] | None = None
    cancel_fragment = ""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
        async with (
            httpx.AsyncClient(timeout=timeout, trust_env=False) as stream_client,
            httpx.AsyncClient(timeout=timeout, trust_env=False) as audit_client,
        ):
            await _wait_ready(audit_client, base_url, process, headers)
            unauthorized = await audit_client.get(f"{base_url}{CONTROL_AUDIT_PATH}")
            models = await audit_client.get(f"{base_url}{MODELS_PATH}", headers=headers)
            if unauthorized.status_code != 401 or models.status_code != 200:
                raise ValueError("incremental control auth/models check failed")
            model_payload = decode_strict_json_object(models.content)
            data = model_payload.get("data")
            if (
                not isinstance(data, list)
                or len(data) != 1
                or not isinstance(data[0], Mapping)
                or data[0].get("id") != MODEL_ID
            ):
                raise ValueError("incremental control model discovery drift")

            async with stream_client.stream(
                "POST",
                f"{base_url}{CHAT_COMPLETIONS_PATH}",
                headers=headers,
                json=_request_body(COMPLETE_PROMPT, len(COMPLETE_PLAN)),
            ) as response:
                complete = await _consume_complete(response)

            async with stream_client.stream(
                "POST",
                f"{base_url}{CHAT_COMPLETIONS_PATH}",
                headers=headers,
                json=_request_body(CANCEL_PROMPT, MAX_NEW_TOKENS),
            ) as response:
                cancel_fragment, preclose = await _read_cancel_prefix(
                    response,
                    audit_client=audit_client,
                    base_url=base_url,
                    headers=headers,
                )
                await response.aclose()
            final_audit = await _wait_cancelled(audit_client, base_url, headers)
            if process.poll() is not None:
                raise RuntimeError("incremental control server exited during run")
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=10.0)

    if complete is None or preclose is None or final_audit is None:
        raise RuntimeError("incremental control did not collect all audit states")
    preclose_backend = preclose.get("backend")
    final_backend = final_audit.get("backend")
    if not isinstance(preclose_backend, Mapping) or not isinstance(
        final_backend, Mapping
    ):
        raise ValueError("incremental control backend audit is missing")
    expected_complete_text = "".join(item[1] for item in COMPLETE_PLAN)
    if (
        complete.get("content") != expected_complete_text
        or complete.get("content_delta_count") != len(COMPLETE_PLAN)
        or complete.get("finish_reason") != "stop"
        or complete.get("usage")
        != {
            "prompt_tokens": COMPLETE_PROMPT_TOKENS,
            "completion_tokens": len(COMPLETE_PLAN),
            "total_tokens": COMPLETE_PROMPT_TOKENS + len(COMPLETE_PLAN),
        }
        or complete.get("done") is not True
        or complete.get("system_fingerprint") != SCRIPTED_BACKEND_FINGERPRINT
    ):
        raise ValueError("complete incremental stream projection drift")
    if (
        cancel_fragment != CANCEL_FIRST[1]
        or preclose.get("active_incremental_streams") != 1
        or preclose.get("cancelled_incremental_streams") != 0
        or preclose_backend.get("cancel_first_delta_emitted") is not True
        or preclose_backend.get("cancel_completed") is not False
        or final_audit.get("active_incremental_streams") != 0
        or final_audit.get("cancelled_incremental_streams") != 1
        or final_backend.get("cancel_asyncio_cancelled_error_observed") is not True
        or final_backend.get("cancel_iterator_closed") is not True
        or final_backend.get("cancel_completed") is not False
        or final_backend.get("cancel_emitted_token_ids") != [CANCEL_FIRST[0]]
    ):
        raise ValueError("disconnect cancellation audit drift")
    if (
        final_audit.get("accepted_requests") != 2
        or final_audit.get("stream_requests") != 2
        or final_audit.get("nonstream_requests") != 0
        or final_audit.get("completed_incremental_streams") != 1
        or final_audit.get("failed_backend_requests") != 0
        or final_backend.get("stream_call_count") != 2
        or final_backend.get("complete_completed") is not True
        or final_backend.get("complete_emitted_token_ids")
        != [item[0] for item in COMPLETE_PLAN]
    ):
        raise ValueError("incremental control final audit drift")

    report: dict[str, Any] = {
        "report_version": INCREMENTAL_STREAMING_REPORT_VERSION,
        "checked_at": "2026-08-13",
        "implementation": INCREMENTAL_STREAMING_CONTROL_VERSION,
        "backend_fingerprint": SCRIPTED_BACKEND_FINGERPRINT,
        "runtime": _runtime_versions(),
        "transport": {
            "scheme": "http",
            "address_scope": "IPv4 loopback",
            "real_tcp_http": True,
            "server_subprocess": True,
            "tls": False,
            "bearer_header_gate": True,
            "unauthorized_audit_status": 401,
            "raw_bearer_or_session_published": False,
        },
        "protocol": {
            "service_version": OPENAI_REFERENCE_SERVICE_VERSION,
            "endpoint": CHAT_COMPLETIONS_PATH,
            "models_endpoint_executed": True,
            "streaming_only_profile": True,
            "usage_event_required_for_complete_case": True,
            "sse_done_required_for_complete_case": True,
            "raw_request_or_response_published": False,
        },
        "complete_stream": {
            "client_content_delta_count": len(COMPLETE_PLAN),
            "client_content_fingerprint": _canonical_sha256(
                {"text": expected_complete_text}
            ),
            "client_finish_reason": "stop",
            "client_usage": copy.deepcopy(complete["usage"]),
            "client_sse_done_observed": True,
            "backend_completion_token_ids": [item[0] for item in COMPLETE_PLAN],
            "backend_completed": True,
        },
        "disconnect_stream": {
            "client_content_delta_count_before_close": 1,
            "client_content_fingerprint_before_close": _canonical_sha256(
                {"text": cancel_fragment}
            ),
            "preclose_service_active_streams": 1,
            "preclose_service_cancelled_streams": 0,
            "preclose_backend_first_delta_emitted": True,
            "preclose_backend_completed": False,
            "client_response_explicitly_closed": True,
            "postclose_service_active_streams": 0,
            "postclose_service_cancelled_streams": 1,
            "postclose_backend_asyncio_cancelled_error_observed": True,
            "postclose_backend_iterator_closed": True,
            "postclose_backend_completed": False,
            "postclose_backend_emitted_token_ids": [CANCEL_FIRST[0]],
        },
        "audit": {
            "accepted_requests": 2,
            "stream_requests": 2,
            "nonstream_requests": 0,
            "completed_incremental_streams": 1,
            "cancelled_incremental_streams": 1,
            "failed_backend_requests": 0,
            "single_process_admission_limit": 1,
            "backend_stream_call_count": 2,
        },
        "server_process": {
            "subprocess_used": True,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_stderr_empty": not stdout and not stderr,
        },
        "scope": {
            "authored_async_backend_executed": True,
            "real_ipv4_loopback_tcp_http_executed": True,
            "content_observed_before_backend_completion": True,
            "client_disconnect_cancelled_asgi_stream_task": True,
            "cooperative_async_backend_cancellation_observed": True,
            "later_authored_deltas_suppressed_after_disconnect": True,
            "tokenizer_or_model_forward_executed": False,
            "transformers_generation_thread_cancellation_proven": False,
            "vllm_or_cuda_executed": False,
            "blocking_thread_or_process_termination_proven": False,
            "kv_or_gpu_resource_release_proven": False,
            "remote_provider_cancellation_or_billing_proven": False,
            "tls_proxy_remote_or_multiworker_proven": False,
            "performance_quality_or_slo_proven": False,
            "full_openai_api_compatibility_proven": False,
            "report_fingerprint_proves_authenticity": False,
        },
        "evidence_boundary": INCREMENTAL_STREAMING_EVIDENCE_BOUNDARY,
    }
    report["report_fingerprint"] = _canonical_sha256(report)
    return report


def run_incremental_streaming_control() -> dict[str, Any]:
    """Execute the deterministic real-loopback completion and disconnect cases."""

    return asyncio.run(_run_control_async())


def _exact(value: Mapping[str, Any], fields: frozenset[str], location: str) -> None:
    if frozenset(value) != fields:
        raise ValueError(f"{location} fields are invalid")


def verify_incremental_streaming_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the reviewed recorded report without starting a server."""

    _exact(report, _TOP_LEVEL_FIELDS, "report")
    nested = (
        ("runtime", _RUNTIME_FIELDS),
        ("transport", _TRANSPORT_FIELDS),
        ("protocol", _PROTOCOL_FIELDS),
        ("complete_stream", _COMPLETE_FIELDS),
        ("disconnect_stream", _DISCONNECT_FIELDS),
        ("audit", _AUDIT_FIELDS),
        ("server_process", _PROCESS_FIELDS),
        ("scope", _SCOPE_FIELDS),
    )
    for name, fields in nested:
        value = report.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"report.{name} must be an object")
        _exact(value, fields, f"report.{name}")
    supplied = report.get("report_fingerprint")
    if not isinstance(supplied, str):
        raise ValueError("report fingerprint is invalid")
    unsigned = {key: copy.deepcopy(value) for key, value in report.items()}
    del unsigned["report_fingerprint"]
    if not hmac.compare_digest(supplied, _canonical_sha256(unsigned)):
        raise ValueError("report fingerprint mismatch")
    if (
        report.get("report_version") != INCREMENTAL_STREAMING_REPORT_VERSION
        or report.get("checked_at") != "2026-08-13"
        or report.get("implementation") != INCREMENTAL_STREAMING_CONTROL_VERSION
        or report.get("backend_fingerprint") != SCRIPTED_BACKEND_FINGERPRINT
        or report.get("runtime") != REVIEWED_RUNTIME
        or report.get("evidence_boundary") != INCREMENTAL_STREAMING_EVIDENCE_BOUNDARY
    ):
        raise ValueError("report identity/runtime/evidence boundary drift")
    if report["transport"] != {
        "scheme": "http",
        "address_scope": "IPv4 loopback",
        "real_tcp_http": True,
        "server_subprocess": True,
        "tls": False,
        "bearer_header_gate": True,
        "unauthorized_audit_status": 401,
        "raw_bearer_or_session_published": False,
    }:
        raise ValueError("report transport drift")
    if report["protocol"] != {
        "service_version": OPENAI_REFERENCE_SERVICE_VERSION,
        "endpoint": CHAT_COMPLETIONS_PATH,
        "models_endpoint_executed": True,
        "streaming_only_profile": True,
        "usage_event_required_for_complete_case": True,
        "sse_done_required_for_complete_case": True,
        "raw_request_or_response_published": False,
    }:
        raise ValueError("report protocol drift")
    complete_text = "".join(item[1] for item in COMPLETE_PLAN)
    if report["complete_stream"] != {
        "client_content_delta_count": len(COMPLETE_PLAN),
        "client_content_fingerprint": _canonical_sha256({"text": complete_text}),
        "client_finish_reason": "stop",
        "client_usage": {
            "prompt_tokens": COMPLETE_PROMPT_TOKENS,
            "completion_tokens": len(COMPLETE_PLAN),
            "total_tokens": COMPLETE_PROMPT_TOKENS + len(COMPLETE_PLAN),
        },
        "client_sse_done_observed": True,
        "backend_completion_token_ids": [item[0] for item in COMPLETE_PLAN],
        "backend_completed": True,
    }:
        raise ValueError("report complete-stream drift")
    if report["disconnect_stream"] != {
        "client_content_delta_count_before_close": 1,
        "client_content_fingerprint_before_close": _canonical_sha256(
            {"text": CANCEL_FIRST[1]}
        ),
        "preclose_service_active_streams": 1,
        "preclose_service_cancelled_streams": 0,
        "preclose_backend_first_delta_emitted": True,
        "preclose_backend_completed": False,
        "client_response_explicitly_closed": True,
        "postclose_service_active_streams": 0,
        "postclose_service_cancelled_streams": 1,
        "postclose_backend_asyncio_cancelled_error_observed": True,
        "postclose_backend_iterator_closed": True,
        "postclose_backend_completed": False,
        "postclose_backend_emitted_token_ids": [CANCEL_FIRST[0]],
    }:
        raise ValueError("report disconnect-stream drift")
    if report["audit"] != {
        "accepted_requests": 2,
        "stream_requests": 2,
        "nonstream_requests": 0,
        "completed_incremental_streams": 1,
        "cancelled_incremental_streams": 1,
        "failed_backend_requests": 0,
        "single_process_admission_limit": 1,
        "backend_stream_call_count": 2,
    }:
        raise ValueError("report audit drift")
    if report["server_process"] != {
        "subprocess_used": True,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_stderr_empty": True,
    }:
        raise ValueError("report server-process drift")
    expected_true = {
        "authored_async_backend_executed",
        "real_ipv4_loopback_tcp_http_executed",
        "content_observed_before_backend_completion",
        "client_disconnect_cancelled_asgi_stream_task",
        "cooperative_async_backend_cancellation_observed",
        "later_authored_deltas_suppressed_after_disconnect",
    }
    scope = cast(Mapping[str, Any], report["scope"])
    if any(scope[name] is not (name in expected_true) for name in _SCOPE_FIELDS):
        raise ValueError("report scope drift")
    return copy.deepcopy(dict(report))


def load_and_verify_incremental_streaming_report(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_RECORDED_REPORT_BYTES:
        raise ValueError("recorded report size is invalid")
    try:
        report = decode_strict_json_object(raw)
    except ValueError as error:
        raise ValueError("recorded report is not strict JSON") from error
    return verify_incremental_streaming_report(report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, required=True)
    subparsers.add_parser("run")
    verify_parser = subparsers.add_parser("verify-recorded")
    verify_parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        token = os.environ.get(CONTROL_TOKEN_ENV, "")
        if not token:
            raise ValueError("incremental control token is missing")
        return serve_control(args.host, args.port, bearer_token=token)
    if args.command == "run":
        report = run_incremental_streaming_control()
    else:
        report = load_and_verify_incremental_streaming_report(args.report)
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
