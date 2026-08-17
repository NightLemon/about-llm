"""Real-loopback cooperative cancellation of a tiny Transformers generation thread."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hmac
import importlib.metadata
import logging
import os
import platform
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, timezone
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
    MODELS_PATH,
    OPENAI_REFERENCE_SERVICE_VERSION,
    ChatCompletionRequest,
    IncrementalTokenDelta,
    build_incremental_reference_app,
    decode_strict_json_object,
)
from about_llm.inference.sse import STREAM_FINISHED, parse_sse_data_line
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

TRANSFORMERS_THREAD_CANCELLATION_CONTROL_VERSION: Final = (
    "about-llm.transformers-thread-cancellation-control.v1"
)
TRANSFORMERS_THREAD_CANCELLATION_REPORT_VERSION: Final = (
    "about-llm.transformers-thread-cancellation-control-report.v1"
)
CONTROL_AUDIT_PATH: Final = "/control/transformers-thread-audit"
CONTROL_TOKEN_ENV: Final = "ABOUT_LLM_TRANSFORMERS_THREAD_CANCEL_TOKEN"
MODEL_ID: Final = "about-llm/random-tiny-gpt2-thread-cancel"
PROMPT: Final = "cancel-reviewed-transformers-thread"
INPUT_TOKEN_IDS: Final = (1, 2, 3)
FORCED_TOKEN_ID: Final = 7
AUTHORED_TEXT_DELTA: Final = "首"
RANDOM_SEED: Final = 20_260_813
EXPECTED_PARAMETER_COUNT: Final = 1_272
MAX_NEW_TOKENS: Final = 8
MAX_RECORDED_REPORT_BYTES: Final = 64_000
REQUEST_TIMEOUT_SECONDS: Final = 30.0
START_TIMEOUT_SECONDS: Final = 60.0

TRANSFORMERS_THREAD_CANCELLATION_EVIDENCE_BOUNDARY: Final = (
    "This control constructs a random 1,272-parameter GPT2LMHeadModel on CPU and "
    "executes the real Transformers GenerationMixin.generate loop in one Python "
    "thread. An authored logits processor forces one integer token, and an authored "
    "streamer deliberately pauses that generation thread after publishing the first "
    "token until the ASGI backend observes client disconnect. The backend then sets a "
    "threading.Event; an authored StoppingCriteria observes it on the reviewed next "
    "termination check; generate returns; and the thread is joined. This proves only "
    "that this explicitly cooperative tiny CPU path can propagate cancellation and "
    "stop before a second generated token. It does not execute a tokenizer, chat "
    "template, public checkpoint, target-model logits, vLLM, CUDA, a GPU kernel, TLS, "
    "a proxy, IAM, a remote client, multiple workers, provider billing, or performance, "
    "quality, capacity, or SLO measurement. The deterministic streamer pause is a "
    "control synchronization mechanism, not production scheduling. The result does "
    "not prove that an unmodified or already-blocked Transformers call, arbitrary "
    "thread/process, model runtime, CUDA kernel, or remote provider will stop, nor does "
    "it prove KV/CPU/GPU memory release. Its unkeyed fingerprint does not authenticate "
    "the recorder."
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return "sha256:" + artifact_fingerprint(value)


BACKEND_FINGERPRINT: Final = _canonical_sha256(
    {
        "control_version": TRANSFORMERS_THREAD_CANCELLATION_CONTROL_VERSION,
        "model_id": MODEL_ID,
        "architecture": "GPT2LMHeadModel",
        "config": {
            "vocab_size": 32,
            "n_positions": 16,
            "n_embd": 8,
            "n_layer": 1,
            "n_head": 1,
        },
        "random_seed": RANDOM_SEED,
        "input_token_ids": list(INPUT_TOKEN_IDS),
        "forced_token_id": FORCED_TOKEN_ID,
        "authored_text_delta": AUTHORED_TEXT_DELTA,
        "cooperation": "streamer-pause+threading-event+stopping-criteria",
    }
)

REVIEWED_RUNTIME: Final = {
    "python": "3.12.10",
    "torch": "2.13.0+cpu",
    "transformers": "4.57.6",
    "httpx": "0.28.1",
    "starlette": "0.41.3",
    "uvicorn": "0.52.1",
}
REVIEWED_CHECKED_AT: Final = "2026-08-13"

_TOP_FIELDS = frozenset(
    {
        "report_version",
        "checked_at",
        "implementation",
        "backend_fingerprint",
        "runtime",
        "model",
        "transport",
        "protocol",
        "preclose",
        "postclose",
        "audit",
        "server_process",
        "scope",
        "evidence_boundary",
        "report_fingerprint",
    }
)
_RUNTIME_FIELDS = frozenset(REVIEWED_RUNTIME)
_MODEL_FIELDS = frozenset(
    {
        "architecture",
        "parameter_count",
        "device",
        "dtype",
        "random_seed",
        "input_token_ids",
        "forced_token_id",
        "tokenizer_or_chat_template_executed",
        "public_checkpoint_loaded",
    }
)
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
        "client_content_delta_count_before_close",
        "client_content_fingerprint_before_close",
        "raw_request_or_response_published",
    }
)
_PRECLOSE_FIELDS = frozenset(
    {
        "service_active_streams",
        "service_cancelled_streams",
        "backend_stream_call_count",
        "generation_thread_alive",
        "generation_returned",
        "streamer_waiting_for_cancel",
        "stopping_criteria_observed_cancel",
        "generated_token_ids",
        "forward_call_count",
    }
)
_POSTCLOSE_FIELDS = frozenset(
    {
        "client_response_explicitly_closed",
        "service_active_streams",
        "service_cancelled_streams",
        "backend_asyncio_cancelled_error_observed",
        "cancellation_event_set",
        "streamer_wait_released_by_cancel",
        "stopping_criteria_observed_cancel",
        "stopping_criteria_call_count",
        "generation_returned",
        "generation_thread_exited",
        "generation_thread_joined",
        "generation_thread_alive",
        "generation_error_type",
        "generated_token_ids",
        "generate_output_token_ids",
        "forward_call_count",
        "logits_processor_call_count",
        "streamer_end_called",
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
    }
)
_PROCESS_FIELDS = frozenset(
    {"subprocess_used", "stdout_bytes", "stderr_bytes", "stdout_stderr_empty"}
)
_SCOPE_FIELDS = frozenset(
    {
        "random_tiny_gpt2_cpu_model_constructed",
        "transformers_generation_mixin_generate_executed",
        "real_model_forward_executed",
        "blocking_python_generation_thread_executed",
        "real_ipv4_loopback_tcp_http_executed",
        "content_observed_before_generate_returned",
        "client_disconnect_cancelled_asgi_stream_task",
        "backend_asyncio_cancelled_error_observed",
        "threading_event_observed_by_stopping_criteria",
        "generation_thread_joined_before_postclose_audit",
        "second_generated_token_suppressed",
        "unmodified_transformers_cancellation_proven",
        "tokenizer_or_chat_template_executed",
        "public_checkpoint_or_target_logits_executed",
        "vllm_or_cuda_executed",
        "kv_cpu_or_gpu_memory_release_proven",
        "arbitrary_thread_process_or_kernel_termination_proven",
        "remote_provider_cancellation_or_billing_proven",
        "tls_proxy_remote_or_multiworker_proven",
        "performance_quality_capacity_or_slo_proven",
        "report_fingerprint_proves_authenticity",
    }
)


class _ForcedTokenProcessor:
    def __init__(self, token_id: int, lock: threading.Lock) -> None:
        self.token_id = token_id
        self.lock = lock
        self.call_count = 0

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if scores.ndim != 2 or not 0 <= self.token_id < scores.shape[1]:
            raise RuntimeError("forced-token processor received invalid scores")
        with self.lock:
            self.call_count += 1
        scores.fill_(float("-inf"))
        scores[:, self.token_id] = 0.0
        return scores


class _CancellationStoppingCriteria:
    def __init__(
        self,
        cancellation_event: threading.Event,
        lock: threading.Lock,
    ) -> None:
        self.cancellation_event = cancellation_event
        self.lock = lock
        self.call_count = 0
        self.observed_cancel = False

    def __call__(self, input_ids: Any, scores: Any, **_: Any) -> Any:
        import torch

        observed = self.cancellation_event.is_set()
        with self.lock:
            self.call_count += 1
            self.observed_cancel = self.observed_cancel or observed
        return torch.full(
            (input_ids.shape[0],),
            observed,
            dtype=torch.bool,
            device=input_ids.device,
        )


class _BlockingFirstTokenStreamer:
    def __init__(
        self,
        cancellation_event: threading.Event,
        token_queue: queue.Queue[int],
        lock: threading.Lock,
    ) -> None:
        self.cancellation_event = cancellation_event
        self.token_queue = token_queue
        self.lock = lock
        self.prompt_seen = False
        self.waiting_for_cancel = False
        self.wait_released_by_cancel = False
        self.end_called = False
        self.generated_token_ids: list[int] = []

    def put(self, value: Any) -> None:
        values = value.detach().to(device="cpu").reshape(-1).tolist()
        token_ids = [int(item) for item in values]
        with self.lock:
            if not self.prompt_seen:
                if token_ids != list(INPUT_TOKEN_IDS):
                    raise RuntimeError("generation streamer prompt token drift")
                self.prompt_seen = True
                return
            if len(token_ids) != 1:
                raise RuntimeError("generation streamer token shape drift")
            token_id = token_ids[0]
            self.generated_token_ids.append(token_id)
            if len(self.generated_token_ids) > 1:
                raise RuntimeError("generation produced a second token after cancellation")
            self.waiting_for_cancel = True
        self.token_queue.put(token_id)
        if not self.cancellation_event.wait(timeout=REQUEST_TIMEOUT_SECONDS):
            raise TimeoutError("generation streamer did not receive cancellation")
        with self.lock:
            self.waiting_for_cancel = False
            self.wait_released_by_cancel = True

    def end(self) -> None:
        with self.lock:
            self.end_called = True


class TinyTransformersThreadBackend:
    """Random tiny GPT-2 backend with explicit cooperative thread cancellation."""

    model_id = MODEL_ID
    backend_fingerprint = BACKEND_FINGERPRINT

    def __init__(self) -> None:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(RANDOM_SEED)
        config = GPT2Config(  # type: ignore[no-untyped-call]
            vocab_size=32,
            n_positions=16,
            n_ctx=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=0,
        )
        self.model = GPT2LMHeadModel(config).to(  # type: ignore[no-untyped-call,call-arg]
            device="cpu", dtype=torch.float32
        )
        self.model.requires_grad_(False)
        self.model.eval()
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError("tiny GPT-2 parameter-count drift")

        self._lock = threading.Lock()
        self._cancellation_event = threading.Event()
        self._token_queue: queue.Queue[int] = queue.Queue(maxsize=2)
        self._processor = _ForcedTokenProcessor(FORCED_TOKEN_ID, self._lock)
        self._stopping = _CancellationStoppingCriteria(
            self._cancellation_event, self._lock
        )
        self._streamer = _BlockingFirstTokenStreamer(
            self._cancellation_event, self._token_queue, self._lock
        )
        self._thread: threading.Thread | None = None
        self._thread_exited_event = threading.Event()
        self.stream_call_count = 0
        self.backend_asyncio_cancelled_error_observed = False
        self.generation_started = False
        self.generation_returned = False
        self.generation_thread_exited = False
        self.generation_thread_joined = False
        self.generation_error_type: str | None = None
        self.generate_output_token_ids: list[int] = []
        self.forward_call_count = 0
        self._forward_hook = self.model.register_forward_pre_hook(
            self._record_forward_call
        )

    def _record_forward_call(self, _module: Any, _inputs: Any) -> None:
        with self._lock:
            self.forward_call_count += 1

    def _run_generate(self) -> None:
        import torch

        with self._lock:
            self.generation_started = True
        try:
            input_ids = torch.tensor([INPUT_TOKEN_IDS], dtype=torch.long, device="cpu")
            attention_mask = torch.ones_like(input_ids)
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=0,
                eos_token_id=None,
                logits_processor=[self._processor],
                stopping_criteria=[self._stopping],
                streamer=self._streamer,
                return_dict_in_generate=True,
            )
            continuation = generated.sequences[0, len(INPUT_TOKEN_IDS) :]
            output_ids = [int(item) for item in continuation.tolist()]
            with self._lock:
                self.generate_output_token_ids = output_ids
                self.generation_returned = True
        except BaseException as error:
            with self._lock:
                self.generation_error_type = type(error).__name__
        finally:
            with self._lock:
                self.generation_thread_exited = True
            self._thread_exited_event.set()

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[IncrementalTokenDelta]:
        if (
            self.stream_call_count != 0
            or len(request.messages) != 1
            or request.messages[0].role != "user"
            or request.messages[0].content != PROMPT
            or request.max_tokens != MAX_NEW_TOKENS
        ):
            raise ValueError("tiny Transformers cancellation workload drift")
        self.stream_call_count += 1
        thread = threading.Thread(
            target=self._run_generate,
            name="about-llm-tiny-transformers-generate",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        try:
            while True:
                try:
                    token_id = self._token_queue.get_nowait()
                except queue.Empty:
                    if self._thread_exited_event.is_set():
                        raise RuntimeError(
                            "generation thread exited before disconnect cancellation"
                        ) from None
                    await asyncio.sleep(0.005)
                    continue
                if token_id != FORCED_TOKEN_ID:
                    raise RuntimeError("generation thread emitted unexpected token")
                yield IncrementalTokenDelta(
                    text_delta=AUTHORED_TEXT_DELTA,
                    token_id=token_id,
                    prompt_token_count=len(INPUT_TOKEN_IDS),
                )
        except asyncio.CancelledError:
            self.backend_asyncio_cancelled_error_observed = True
            self._cancellation_event.set()
            raise
        finally:
            self._cancellation_event.set()
            thread.join(timeout=REQUEST_TIMEOUT_SECONDS)
            with self._lock:
                self.generation_thread_joined = not thread.is_alive()
            self._forward_hook.remove()

    def audit_projection(self) -> Mapping[str, Any]:
        with self._lock:
            thread = self._thread
            return {
                "implementation": "tiny-gpt2-generate-thread+cooperative-stop",
                "architecture": self.model.__class__.__name__,
                "parameter_count": sum(
                    parameter.numel() for parameter in self.model.parameters()
                ),
                "device": "cpu",
                "dtype": "float32",
                "random_seed": RANDOM_SEED,
                "input_token_ids": list(INPUT_TOKEN_IDS),
                "forced_token_id": FORCED_TOKEN_ID,
                "stream_call_count": self.stream_call_count,
                "generation_started": self.generation_started,
                "generation_returned": self.generation_returned,
                "generation_thread_exited": self.generation_thread_exited,
                "generation_thread_joined": self.generation_thread_joined,
                "generation_thread_alive": bool(thread and thread.is_alive()),
                "generation_error_type": self.generation_error_type,
                "backend_asyncio_cancelled_error_observed": (
                    self.backend_asyncio_cancelled_error_observed
                ),
                "cancellation_event_set": self._cancellation_event.is_set(),
                "streamer_waiting_for_cancel": self._streamer.waiting_for_cancel,
                "streamer_wait_released_by_cancel": (
                    self._streamer.wait_released_by_cancel
                ),
                "streamer_end_called": self._streamer.end_called,
                "generated_token_ids": list(self._streamer.generated_token_ids),
                "generate_output_token_ids": list(self.generate_output_token_ids),
                "stopping_criteria_call_count": self._stopping.call_count,
                "stopping_criteria_observed_cancel": self._stopping.observed_cancel,
                "forward_call_count": self.forward_call_count,
                "logits_processor_call_count": self._processor.call_count,
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


def build_threaded_control_app(
    backend: TinyTransformersThreadBackend,
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
        raise ValueError("thread-cancellation control only permits IPv4 loopback")
    if not 0 < port < 65_536:
        raise ValueError("port must be between 1 and 65535")
    logging.disable(logging.CRITICAL)
    server = uvicorn.Server(
        uvicorn.Config(
            build_threaded_control_app(
                TinyTransformersThreadBackend(), bearer_token=bearer_token
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


def _request_body() -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_NEW_TOKENS,
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
            raise RuntimeError("thread-cancellation server exited before readiness")
        try:
            response = await client.get(
                f"{base_url}{CONTROL_AUDIT_PATH}", headers=headers
            )
            if response.status_code == 200:
                return
        except httpx.TransportError:
            pass
        await asyncio.sleep(0.05)
    raise TimeoutError("thread-cancellation server did not become ready")


async def _read_first_content_and_audit(
    response: httpx.Response,
    *,
    audit_client: httpx.AsyncClient,
    base_url: str,
    headers: Mapping[str, str],
) -> tuple[str, dict[str, Any]]:
    if response.status_code != 200:
        raise ValueError("thread-cancellation stream request failed")
    async for line in response.aiter_lines():
        parsed = parse_sse_data_line(line)
        if parsed is None:
            continue
        if parsed is STREAM_FINISHED:
            raise ValueError("thread-cancellation stream completed before disconnect")
        if not isinstance(parsed, dict):
            raise ValueError("thread-cancellation stream event is invalid")
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ValueError("thread-cancellation stream choices are invalid")
        delta = first.get("delta")
        if not isinstance(delta, Mapping):
            raise ValueError("thread-cancellation stream delta is invalid")
        fragment = delta.get("content")
        if fragment is None:
            continue
        if not isinstance(fragment, str) or not fragment:
            raise ValueError("thread-cancellation stream content is invalid")
        audit_response = await audit_client.get(
            f"{base_url}{CONTROL_AUDIT_PATH}", headers=headers
        )
        if audit_response.status_code != 200:
            raise ValueError("thread-cancellation preclose audit failed")
        return fragment, decode_strict_json_object(audit_response.content)
    raise ValueError("thread-cancellation stream ended before first content")


async def _wait_postclose(
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
                and backend.get("stopping_criteria_observed_cancel") is True
                and backend.get("generation_thread_joined") is True
                and backend.get("generation_thread_alive") is False
            ):
                return audit
        await asyncio.sleep(0.02)
    raise TimeoutError("thread-cancellation server did not reach postclose state")


def get_transformers_thread_cancellation_runtime() -> dict[str, str]:
    """Return the runtime identity used by a live control execution."""

    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
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
            "about_llm.inference.transformers_thread_cancellation_control",
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
    fragment = ""
    preclose: dict[str, Any] | None = None
    postclose: dict[str, Any] | None = None
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
                raise ValueError("thread-cancellation auth/models check failed")
            model_payload = decode_strict_json_object(models.content)
            data = model_payload.get("data")
            if (
                not isinstance(data, list)
                or len(data) != 1
                or not isinstance(data[0], Mapping)
                or data[0].get("id") != MODEL_ID
            ):
                raise ValueError("thread-cancellation model discovery drift")

            async with stream_client.stream(
                "POST",
                f"{base_url}{CHAT_COMPLETIONS_PATH}",
                headers=headers,
                json=_request_body(),
            ) as response:
                fragment, preclose = await _read_first_content_and_audit(
                    response,
                    audit_client=audit_client,
                    base_url=base_url,
                    headers=headers,
                )
                await response.aclose()
            postclose = await _wait_postclose(audit_client, base_url, headers)
            if process.poll() is not None:
                raise RuntimeError("thread-cancellation server exited during control")
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=10.0)

    if preclose is None or postclose is None:
        raise RuntimeError("thread-cancellation control did not collect audit states")
    pre_backend = preclose.get("backend")
    post_backend = postclose.get("backend")
    if not isinstance(pre_backend, Mapping) or not isinstance(post_backend, Mapping):
        raise ValueError("thread-cancellation backend audit is missing")
    if (
        fragment != AUTHORED_TEXT_DELTA
        or preclose.get("active_incremental_streams") != 1
        or preclose.get("cancelled_incremental_streams") != 0
        or pre_backend.get("stream_call_count") != 1
        or pre_backend.get("generation_thread_alive") is not True
        or pre_backend.get("generation_returned") is not False
        or pre_backend.get("streamer_waiting_for_cancel") is not True
        or pre_backend.get("stopping_criteria_observed_cancel") is not False
        or pre_backend.get("generated_token_ids") != [FORCED_TOKEN_ID]
        or pre_backend.get("forward_call_count") != 1
    ):
        raise ValueError("thread-cancellation preclose audit drift")
    if (
        postclose.get("active_incremental_streams") != 0
        or postclose.get("cancelled_incremental_streams") != 1
        or postclose.get("completed_incremental_streams") != 0
        or postclose.get("failed_backend_requests") != 0
        or post_backend.get("backend_asyncio_cancelled_error_observed") is not True
        or post_backend.get("cancellation_event_set") is not True
        or post_backend.get("streamer_wait_released_by_cancel") is not True
        or post_backend.get("stopping_criteria_observed_cancel") is not True
        or post_backend.get("stopping_criteria_call_count") != 1
        or post_backend.get("generation_returned") is not True
        or post_backend.get("generation_thread_exited") is not True
        or post_backend.get("generation_thread_joined") is not True
        or post_backend.get("generation_thread_alive") is not False
        or post_backend.get("generation_error_type") is not None
        or post_backend.get("generated_token_ids") != [FORCED_TOKEN_ID]
        or post_backend.get("generate_output_token_ids") != [FORCED_TOKEN_ID]
        or post_backend.get("forward_call_count") != 1
        or post_backend.get("logits_processor_call_count") != 1
        or post_backend.get("streamer_end_called") is not True
    ):
        raise ValueError("thread-cancellation postclose audit drift")

    runtime = get_transformers_thread_cancellation_runtime()
    checked_at = datetime.now(timezone.utc).date().isoformat()
    report: dict[str, Any] = {
        "report_version": TRANSFORMERS_THREAD_CANCELLATION_REPORT_VERSION,
        "checked_at": checked_at,
        "implementation": TRANSFORMERS_THREAD_CANCELLATION_CONTROL_VERSION,
        "backend_fingerprint": BACKEND_FINGERPRINT,
        "runtime": runtime,
        "model": {
            "architecture": "GPT2LMHeadModel",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "device": "cpu",
            "dtype": "float32",
            "random_seed": RANDOM_SEED,
            "input_token_ids": list(INPUT_TOKEN_IDS),
            "forced_token_id": FORCED_TOKEN_ID,
            "tokenizer_or_chat_template_executed": False,
            "public_checkpoint_loaded": False,
        },
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
            "client_content_delta_count_before_close": 1,
            "client_content_fingerprint_before_close": _canonical_sha256(
                {"text": fragment}
            ),
            "raw_request_or_response_published": False,
        },
        "preclose": {
            "service_active_streams": 1,
            "service_cancelled_streams": 0,
            "backend_stream_call_count": 1,
            "generation_thread_alive": True,
            "generation_returned": False,
            "streamer_waiting_for_cancel": True,
            "stopping_criteria_observed_cancel": False,
            "generated_token_ids": [FORCED_TOKEN_ID],
            "forward_call_count": 1,
        },
        "postclose": {
            "client_response_explicitly_closed": True,
            "service_active_streams": 0,
            "service_cancelled_streams": 1,
            "backend_asyncio_cancelled_error_observed": True,
            "cancellation_event_set": True,
            "streamer_wait_released_by_cancel": True,
            "stopping_criteria_observed_cancel": True,
            "stopping_criteria_call_count": 1,
            "generation_returned": True,
            "generation_thread_exited": True,
            "generation_thread_joined": True,
            "generation_thread_alive": False,
            "generation_error_type": None,
            "generated_token_ids": [FORCED_TOKEN_ID],
            "generate_output_token_ids": [FORCED_TOKEN_ID],
            "forward_call_count": 1,
            "logits_processor_call_count": 1,
            "streamer_end_called": True,
        },
        "audit": {
            "accepted_requests": 1,
            "stream_requests": 1,
            "nonstream_requests": 0,
            "completed_incremental_streams": 0,
            "cancelled_incremental_streams": 1,
            "failed_backend_requests": 0,
            "single_process_admission_limit": 1,
        },
        "server_process": {
            "subprocess_used": True,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_stderr_empty": not stdout and not stderr,
        },
        "scope": {
            "random_tiny_gpt2_cpu_model_constructed": True,
            "transformers_generation_mixin_generate_executed": True,
            "real_model_forward_executed": True,
            "blocking_python_generation_thread_executed": True,
            "real_ipv4_loopback_tcp_http_executed": True,
            "content_observed_before_generate_returned": True,
            "client_disconnect_cancelled_asgi_stream_task": True,
            "backend_asyncio_cancelled_error_observed": True,
            "threading_event_observed_by_stopping_criteria": True,
            "generation_thread_joined_before_postclose_audit": True,
            "second_generated_token_suppressed": True,
            "unmodified_transformers_cancellation_proven": False,
            "tokenizer_or_chat_template_executed": False,
            "public_checkpoint_or_target_logits_executed": False,
            "vllm_or_cuda_executed": False,
            "kv_cpu_or_gpu_memory_release_proven": False,
            "arbitrary_thread_process_or_kernel_termination_proven": False,
            "remote_provider_cancellation_or_billing_proven": False,
            "tls_proxy_remote_or_multiworker_proven": False,
            "performance_quality_capacity_or_slo_proven": False,
            "report_fingerprint_proves_authenticity": False,
        },
        "evidence_boundary": TRANSFORMERS_THREAD_CANCELLATION_EVIDENCE_BOUNDARY,
    }
    report["report_fingerprint"] = _canonical_sha256(report)
    return validate_transformers_thread_cancellation_observation(
        report,
        expected_runtime=runtime,
        expected_checked_at=checked_at,
    )


def run_transformers_thread_cancellation_control() -> dict[str, Any]:
    """Run the deterministic tiny-Transformers real-loopback disconnect control."""

    return asyncio.run(_run_control_async())


def _exact(value: Mapping[str, Any], fields: frozenset[str], location: str) -> None:
    if frozenset(value) != fields:
        raise ValueError(f"{location} fields are invalid")


def validate_transformers_thread_cancellation_observation(
    report: Mapping[str, Any],
    *,
    expected_runtime: Mapping[str, str],
    expected_checked_at: str,
) -> dict[str, Any]:
    """Validate one live observation against its explicit runtime and date identity."""

    if (
        frozenset(expected_runtime) != _RUNTIME_FIELDS
        or any(
            not isinstance(value, str) or not value
            for value in expected_runtime.values()
        )
    ):
        raise ValueError("expected runtime identity is invalid")
    try:
        parsed_checked_at = datetime.strptime(expected_checked_at, "%Y-%m-%d").date()
    except (TypeError, ValueError) as error:
        raise ValueError("expected checked-at date is invalid") from error
    if parsed_checked_at.isoformat() != expected_checked_at:
        raise ValueError("expected checked-at date is invalid")

    _exact(report, _TOP_FIELDS, "report")
    nested = (
        ("runtime", _RUNTIME_FIELDS),
        ("model", _MODEL_FIELDS),
        ("transport", _TRANSPORT_FIELDS),
        ("protocol", _PROTOCOL_FIELDS),
        ("preclose", _PRECLOSE_FIELDS),
        ("postclose", _POSTCLOSE_FIELDS),
        ("audit", _AUDIT_FIELDS),
        ("server_process", _PROCESS_FIELDS),
        ("scope", _SCOPE_FIELDS),
    )
    for name, fields in nested:
        value = report.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"report.{name} must be an object")
        _exact(value, fields, f"report.{name}")
    fingerprint = report.get("report_fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("report fingerprint is invalid")
    unsigned = copy.deepcopy(dict(report))
    del unsigned["report_fingerprint"]
    if not hmac.compare_digest(fingerprint, _canonical_sha256(unsigned)):
        raise ValueError("report fingerprint mismatch")
    if (
        report.get("report_version")
        != TRANSFORMERS_THREAD_CANCELLATION_REPORT_VERSION
        or report.get("checked_at") != expected_checked_at
        or report.get("implementation")
        != TRANSFORMERS_THREAD_CANCELLATION_CONTROL_VERSION
        or report.get("backend_fingerprint") != BACKEND_FINGERPRINT
        or report.get("runtime") != dict(expected_runtime)
        or report.get("evidence_boundary")
        != TRANSFORMERS_THREAD_CANCELLATION_EVIDENCE_BOUNDARY
    ):
        raise ValueError("report identity/runtime/evidence boundary drift")
    expected = _expected_report_body()
    for name in (
        "model",
        "transport",
        "protocol",
        "preclose",
        "postclose",
        "audit",
        "server_process",
        "scope",
    ):
        if report[name] != expected[name]:
            raise ValueError(f"report {name} drift")
    return copy.deepcopy(dict(report))


def verify_transformers_thread_cancellation_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the fixed reviewed artifact without running its model or server."""

    return validate_transformers_thread_cancellation_observation(
        report,
        expected_runtime=REVIEWED_RUNTIME,
        expected_checked_at=REVIEWED_CHECKED_AT,
    )


def _expected_report_body() -> dict[str, Any]:
    expected_true_scope = {
        "random_tiny_gpt2_cpu_model_constructed",
        "transformers_generation_mixin_generate_executed",
        "real_model_forward_executed",
        "blocking_python_generation_thread_executed",
        "real_ipv4_loopback_tcp_http_executed",
        "content_observed_before_generate_returned",
        "client_disconnect_cancelled_asgi_stream_task",
        "backend_asyncio_cancelled_error_observed",
        "threading_event_observed_by_stopping_criteria",
        "generation_thread_joined_before_postclose_audit",
        "second_generated_token_suppressed",
    }
    return {
        "model": {
            "architecture": "GPT2LMHeadModel",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "device": "cpu",
            "dtype": "float32",
            "random_seed": RANDOM_SEED,
            "input_token_ids": list(INPUT_TOKEN_IDS),
            "forced_token_id": FORCED_TOKEN_ID,
            "tokenizer_or_chat_template_executed": False,
            "public_checkpoint_loaded": False,
        },
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
            "client_content_delta_count_before_close": 1,
            "client_content_fingerprint_before_close": _canonical_sha256(
                {"text": AUTHORED_TEXT_DELTA}
            ),
            "raw_request_or_response_published": False,
        },
        "preclose": {
            "service_active_streams": 1,
            "service_cancelled_streams": 0,
            "backend_stream_call_count": 1,
            "generation_thread_alive": True,
            "generation_returned": False,
            "streamer_waiting_for_cancel": True,
            "stopping_criteria_observed_cancel": False,
            "generated_token_ids": [FORCED_TOKEN_ID],
            "forward_call_count": 1,
        },
        "postclose": {
            "client_response_explicitly_closed": True,
            "service_active_streams": 0,
            "service_cancelled_streams": 1,
            "backend_asyncio_cancelled_error_observed": True,
            "cancellation_event_set": True,
            "streamer_wait_released_by_cancel": True,
            "stopping_criteria_observed_cancel": True,
            "stopping_criteria_call_count": 1,
            "generation_returned": True,
            "generation_thread_exited": True,
            "generation_thread_joined": True,
            "generation_thread_alive": False,
            "generation_error_type": None,
            "generated_token_ids": [FORCED_TOKEN_ID],
            "generate_output_token_ids": [FORCED_TOKEN_ID],
            "forward_call_count": 1,
            "logits_processor_call_count": 1,
            "streamer_end_called": True,
        },
        "audit": {
            "accepted_requests": 1,
            "stream_requests": 1,
            "nonstream_requests": 0,
            "completed_incremental_streams": 0,
            "cancelled_incremental_streams": 1,
            "failed_backend_requests": 0,
            "single_process_admission_limit": 1,
        },
        "server_process": {
            "subprocess_used": True,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "stdout_stderr_empty": True,
        },
        "scope": {
            name: name in expected_true_scope for name in _SCOPE_FIELDS
        },
    }


def load_and_verify_transformers_thread_cancellation_report(
    path: Path,
) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_RECORDED_REPORT_BYTES:
        raise ValueError("recorded report size is invalid")
    try:
        report = decode_strict_json_object(raw)
    except ValueError as error:
        raise ValueError("recorded report is not strict JSON") from error
    return verify_transformers_thread_cancellation_report(report)


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
            raise ValueError("thread-cancellation control token is missing")
        return serve_control(args.host, args.port, bearer_token=token)
    if args.command == "run":
        report = run_transformers_thread_cancellation_control()
    else:
        report = load_and_verify_transformers_thread_cancellation_report(args.report)
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
