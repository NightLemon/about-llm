"""Strict OpenAI-compatible chat subset for local reference serving.

The implementation is intentionally small and explicit.  It is suitable for
protocol and packaging controls, not a claim of complete OpenAI API, vLLM, or
production-server compatibility.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import math
import secrets
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NoReturn, Protocol, cast

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

OPENAI_REFERENCE_SERVICE_VERSION: Final = "about-llm.openai-reference-service.v1"
CHAT_COMPLETIONS_PATH: Final = "/v1/chat/completions"
MODELS_PATH: Final = "/v1/models"
HEALTH_PATH: Final = "/healthz"
MAX_REQUEST_BODY_BYTES: Final = 32_768
MAX_MESSAGE_CHARACTERS: Final = 4_096
MAX_TOTAL_MESSAGE_CHARACTERS: Final = 8_192
MAX_MESSAGE_COUNT: Final = 8


class StrictJSONBodyError(ValueError):
    """The HTTP body is not one bounded strict JSON object."""


class OpenAIRequestError(ValueError):
    """A stable client-facing rejection without echoed request content."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.safe_message = message
        super().__init__(message)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatCompletionRequest:
    model: str
    messages: tuple[ChatMessage, ...]
    max_tokens: int
    stream: bool
    include_usage: bool


@dataclass(frozen=True)
class GeneratedCompletion:
    """One deterministic backend result before HTTP response projection."""

    text: str
    text_deltas: tuple[str, ...]
    prompt_token_count: int
    completion_token_ids: tuple[int, ...]
    finish_reason: str

    def __post_init__(self) -> None:
        if not self.text or "".join(self.text_deltas) != self.text:
            raise ValueError("completion deltas must concatenate to non-empty text")
        if self.prompt_token_count <= 0 or not self.completion_token_ids:
            raise ValueError("completion token counts must be positive")
        if len(self.text_deltas) > len(self.completion_token_ids):
            raise ValueError("completion cannot expose more deltas than token ids")
        if self.finish_reason not in {"stop", "length"}:
            raise ValueError("unsupported finish reason")

    @property
    def completion_token_count(self) -> int:
        return len(self.completion_token_ids)


class CompletionBackend(Protocol):
    """Model runtime boundary consumed by the HTTP reference service."""

    model_id: str
    backend_fingerprint: str

    async def generate(self, request: ChatCompletionRequest) -> GeneratedCompletion: ...

    def audit_projection(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class IncrementalTokenDelta:
    """One backend-produced token projection for true incremental SSE."""

    text_delta: str
    token_id: int
    prompt_token_count: int
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text_delta, str) or not self.text_delta:
            raise ValueError("incremental text delta must be non-empty")
        if (
            isinstance(self.token_id, bool)
            or not isinstance(self.token_id, int)
            or self.token_id < 0
        ):
            raise ValueError("incremental token id must be a non-negative integer")
        if (
            isinstance(self.prompt_token_count, bool)
            or not isinstance(self.prompt_token_count, int)
            or self.prompt_token_count <= 0
        ):
            raise ValueError("incremental prompt token count must be positive")
        if self.finish_reason not in {None, "stop", "length"}:
            raise ValueError("incremental finish reason is unsupported")


class IncrementalCompletionBackend(Protocol):
    """Cooperative async-token backend consumed by the incremental service."""

    model_id: str
    backend_fingerprint: str

    def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[IncrementalTokenDelta]: ...

    def audit_projection(self) -> Mapping[str, Any]: ...


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON float is forbidden")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def decode_strict_json_object(raw: bytes) -> dict[str, Any]:
    """Decode finite UTF-8 JSON while rejecting ambiguity and oversized input."""

    if not raw or len(raw) > MAX_REQUEST_BODY_BYTES:
        raise StrictJSONBodyError("invalid request body size")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise StrictJSONBodyError("request body is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise StrictJSONBodyError("request body must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> bool:
    keys = frozenset(value)
    return required.issubset(keys) and keys.issubset(required | optional)


def parse_chat_completion_request(
    value: Mapping[str, Any],
    *,
    model_id: str,
    maximum_new_tokens: int,
) -> ChatCompletionRequest:
    """Parse the service's closed greedy chat-completion request profile."""

    if not _exact_keys(
        value,
        required=frozenset(
            {"model", "messages", "max_tokens", "temperature", "stream"}
        ),
        optional=frozenset({"stream_options"}),
    ):
        raise OpenAIRequestError(422, "invalid_request_schema", "Invalid request schema")
    if value.get("model") != model_id:
        raise OpenAIRequestError(404, "model_not_found", "Requested model is unavailable")
    temperature = value.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or float(temperature) != 0.0
    ):
        raise OpenAIRequestError(
            422,
            "unsupported_sampling",
            "Reference service requires greedy temperature=0",
        )
    max_tokens = value.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= maximum_new_tokens
    ):
        raise OpenAIRequestError(422, "invalid_max_tokens", "Invalid max_tokens")
    stream = value.get("stream")
    if not isinstance(stream, bool):
        raise OpenAIRequestError(422, "invalid_stream", "stream must be boolean")

    stream_options = value.get("stream_options")
    include_usage = False
    if stream:
        if stream_options is not None:
            if not isinstance(stream_options, Mapping) or not _exact_keys(
                stream_options, required=frozenset({"include_usage"})
            ):
                raise OpenAIRequestError(
                    422, "invalid_stream_options", "Invalid stream_options"
                )
            if stream_options.get("include_usage") is not True:
                raise OpenAIRequestError(
                    422,
                    "unsupported_stream_options",
                    "Streaming control requires include_usage=true",
                )
            include_usage = True
    elif stream_options is not None:
        raise OpenAIRequestError(
            422,
            "invalid_stream_options",
            "stream_options require stream=true",
        )

    raw_messages = value.get("messages")
    if (
        not isinstance(raw_messages, Sequence)
        or isinstance(raw_messages, str | bytes)
        or not 1 <= len(raw_messages) <= MAX_MESSAGE_COUNT
    ):
        raise OpenAIRequestError(422, "invalid_messages", "Invalid messages")
    messages: list[ChatMessage] = []
    total_characters = 0
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping) or not _exact_keys(
            raw_message, required=frozenset({"role", "content"})
        ):
            raise OpenAIRequestError(422, "invalid_messages", "Invalid messages")
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(
            content, str
        ):
            raise OpenAIRequestError(422, "invalid_messages", "Invalid messages")
        if not content or len(content) > MAX_MESSAGE_CHARACTERS:
            raise OpenAIRequestError(422, "invalid_messages", "Invalid messages")
        total_characters += len(content)
        messages.append(ChatMessage(cast(str, role), content))
    if total_characters > MAX_TOTAL_MESSAGE_CHARACTERS:
        raise OpenAIRequestError(422, "invalid_messages", "Invalid messages")
    if messages[-1].role != "user":
        raise OpenAIRequestError(
            422, "invalid_messages", "Final message must have role user"
        )
    return ChatCompletionRequest(
        model=model_id,
        messages=tuple(messages),
        max_tokens=max_tokens,
        stream=stream,
        include_usage=include_usage,
    )


def _header_values(request: Request, name: str) -> list[str]:
    expected = name.lower().encode("ascii")
    headers = cast(Sequence[tuple[bytes, bytes]], request.scope["headers"])
    return [value.decode("latin-1") for key, value in headers if key == expected]


async def _bounded_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BODY_BYTES:
            raise OpenAIRequestError(413, "request_too_large", "Request body too large")
    return bytes(body)


def _json_response(value: Mapping[str, Any], status_code: int = 200) -> Response:
    return Response(
        canonical_json_bytes(value),
        status_code=status_code,
        media_type="application/json",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


def _error_response(error: OpenAIRequestError) -> Response:
    return _json_response(
        {
            "error": {
                "message": error.safe_message,
                "type": "invalid_request_error",
                "param": None,
                "code": error.code,
            }
        },
        status_code=error.status_code,
    )


def _usage(completion: GeneratedCompletion) -> dict[str, int]:
    return {
        "prompt_tokens": completion.prompt_token_count,
        "completion_tokens": completion.completion_token_count,
        "total_tokens": (
            completion.prompt_token_count + completion.completion_token_count
        ),
    }


class OpenAIReferenceService:
    """Authenticated, single-worker HTTP projection around a completion backend."""

    def __init__(
        self,
        backend: CompletionBackend,
        *,
        bearer_token: str,
        maximum_new_tokens: int,
    ) -> None:
        if not backend.model_id or not backend.backend_fingerprint.startswith("sha256:"):
            raise ValueError("backend identity is invalid")
        if len(bearer_token) < 32 or any(
            ord(character) < 0x21 or ord(character) > 0x7E
            for character in bearer_token
        ):
            raise ValueError("bearer token must be at least 32 visible ASCII characters")
        if maximum_new_tokens <= 0:
            raise ValueError("maximum_new_tokens must be positive")
        self.backend = backend
        self._bearer_token = bearer_token
        self._maximum_new_tokens = maximum_new_tokens
        self._admission = asyncio.Semaphore(1)
        self.accepted_requests = 0
        self.stream_requests = 0
        self.nonstream_requests = 0
        self.failed_backend_requests = 0

    def _authorize(self, request: Request) -> None:
        values = _header_values(request, "authorization")
        valid = False
        if len(values) == 1:
            scheme, separator, token = values[0].partition(" ")
            valid = (
                bool(separator)
                and scheme.lower() == "bearer"
                and hmac.compare_digest(token, self._bearer_token)
            )
        if not valid:
            raise OpenAIRequestError(401, "invalid_api_key", "Invalid API key")

    async def _parse_http_request(self, request: Request) -> ChatCompletionRequest:
        self._authorize(request)
        content_types = _header_values(request, "content-type")
        if len(content_types) != 1 or (
            content_types[0].partition(";")[0].strip().lower()
            != "application/json"
        ):
            raise OpenAIRequestError(
                415, "unsupported_media_type", "Content-Type must be application/json"
            )
        raw = await _bounded_body(request)
        try:
            body = decode_strict_json_object(raw)
        except StrictJSONBodyError as error:
            raise OpenAIRequestError(
                400, "invalid_json", "Invalid JSON request body"
            ) from error
        return parse_chat_completion_request(
            body,
            model_id=self.backend.model_id,
            maximum_new_tokens=self._maximum_new_tokens,
        )

    async def health(self, request: Request) -> Response:
        try:
            self._authorize(request)
        except OpenAIRequestError as error:
            return _error_response(error)
        return _json_response(
            {
                "status": "ready",
                "service_version": OPENAI_REFERENCE_SERVICE_VERSION,
                "model": self.backend.model_id,
                "backend_fingerprint": self.backend.backend_fingerprint,
            }
        )

    async def models(self, request: Request) -> Response:
        try:
            self._authorize(request)
        except OpenAIRequestError as error:
            return _error_response(error)
        return _json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.backend.model_id,
                        "object": "model",
                        "owned_by": "about-llm-local-control",
                    }
                ],
            }
        )

    async def chat_completions(self, request: Request) -> Response:
        try:
            parsed = await self._parse_http_request(request)
            async with self._admission:
                try:
                    completion = await self.backend.generate(parsed)
                except OpenAIRequestError:
                    raise
                except Exception as error:
                    self.failed_backend_requests += 1
                    raise OpenAIRequestError(
                        500, "backend_error", "Model backend failed"
                    ) from error
            self.accepted_requests += 1
            if parsed.stream:
                self.stream_requests += 1
                return StreamingResponse(
                    self._stream_response(parsed, completion),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-store",
                        "X-Content-Type-Options": "nosniff",
                    },
                )
            self.nonstream_requests += 1
            return self._nonstream_response(completion)
        except OpenAIRequestError as error:
            return _error_response(error)

    def _nonstream_response(self, completion: GeneratedCompletion) -> Response:
        return _json_response(
            {
                "id": "chatcmpl-" + secrets.token_hex(12),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.backend.model_id,
                "system_fingerprint": self.backend.backend_fingerprint,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": completion.text},
                        "finish_reason": completion.finish_reason,
                    }
                ],
                "usage": _usage(completion),
            }
        )

    async def _stream_response(
        self,
        request: ChatCompletionRequest,
        completion: GeneratedCompletion,
    ) -> AsyncIterator[bytes]:
        stream_id = "chatcmpl-" + secrets.token_hex(12)
        created = int(time.time())

        def chunk(choices: Sequence[Mapping[str, Any]], usage: Any = None) -> bytes:
            value: dict[str, Any] = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.backend.model_id,
                "system_fingerprint": self.backend.backend_fingerprint,
                "choices": list(choices),
            }
            if usage is not None:
                value["usage"] = usage
            return b"data: " + canonical_json_bytes(value) + b"\n\n"

        yield chunk(
            [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        )
        for delta in completion.text_deltas:
            yield chunk(
                [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]
            )
        yield chunk(
            [{"index": 0, "delta": {}, "finish_reason": completion.finish_reason}]
        )
        if request.include_usage:
            yield chunk([], _usage(completion))
        yield b"data: [DONE]\n\n"

    def audit_projection(self) -> dict[str, Any]:
        return {
            "service_version": OPENAI_REFERENCE_SERVICE_VERSION,
            "accepted_requests": self.accepted_requests,
            "stream_requests": self.stream_requests,
            "nonstream_requests": self.nonstream_requests,
            "failed_backend_requests": self.failed_backend_requests,
            "single_process_admission_limit": 1,
            "backend": dict(self.backend.audit_projection()),
        }


class IncrementalOpenAIReferenceService(OpenAIReferenceService):
    """Streaming-only service that holds admission until completion or disconnect."""

    def __init__(
        self,
        backend: IncrementalCompletionBackend,
        *,
        bearer_token: str,
        maximum_new_tokens: int,
    ) -> None:
        super().__init__(
            cast(CompletionBackend, backend),
            bearer_token=bearer_token,
            maximum_new_tokens=maximum_new_tokens,
        )
        self.completed_incremental_streams = 0
        self.cancelled_incremental_streams = 0
        self.active_incremental_streams = 0

    async def chat_completions(self, request: Request) -> Response:
        try:
            parsed = await self._parse_http_request(request)
            if not parsed.stream:
                raise OpenAIRequestError(
                    422,
                    "stream_required",
                    "Incremental reference service requires stream=true",
                )
            await self._admission.acquire()
            self.accepted_requests += 1
            self.stream_requests += 1
            self.active_incremental_streams += 1
            return StreamingResponse(
                self._incremental_response(parsed),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except OpenAIRequestError as error:
            return _error_response(error)

    async def _incremental_response(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[bytes]:
        stream_id = "chatcmpl-" + secrets.token_hex(12)
        created = int(time.time())
        completion_ids: list[int] = []
        prompt_token_count: int | None = None
        finish_reason: str | None = None
        terminal_seen = False

        def chunk(choices: Sequence[Mapping[str, Any]], usage: Any = None) -> bytes:
            value: dict[str, Any] = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.backend.model_id,
                "system_fingerprint": self.backend.backend_fingerprint,
                "choices": list(choices),
            }
            if usage is not None:
                value["usage"] = usage
            return b"data: " + canonical_json_bytes(value) + b"\n\n"

        try:
            yield chunk(
                [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
            )
            backend = cast(IncrementalCompletionBackend, self.backend)
            async for event in backend.stream(request):
                if terminal_seen:
                    raise RuntimeError("incremental backend emitted after terminal token")
                if prompt_token_count is None:
                    prompt_token_count = event.prompt_token_count
                elif prompt_token_count != event.prompt_token_count:
                    raise RuntimeError("incremental prompt token count drift")
                completion_ids.append(event.token_id)
                if len(completion_ids) > request.max_tokens:
                    raise RuntimeError("incremental backend exceeded max_tokens")
                yield chunk(
                    [
                        {
                            "index": 0,
                            "delta": {"content": event.text_delta},
                            "finish_reason": None,
                        }
                    ]
                )
                if event.finish_reason is not None:
                    terminal_seen = True
                    finish_reason = event.finish_reason
            if (
                not terminal_seen
                or prompt_token_count is None
                or not completion_ids
                or finish_reason is None
            ):
                raise RuntimeError("incremental backend ended without a terminal token")
            yield chunk(
                [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
            )
            if request.include_usage:
                usage = {
                    "prompt_tokens": prompt_token_count,
                    "completion_tokens": len(completion_ids),
                    "total_tokens": prompt_token_count + len(completion_ids),
                }
                yield chunk([], usage)
            self.completed_incremental_streams += 1
            yield b"data: [DONE]\n\n"
        except asyncio.CancelledError:
            self.cancelled_incremental_streams += 1
            raise
        except Exception:
            self.failed_backend_requests += 1
            raise
        finally:
            self.active_incremental_streams -= 1
            self._admission.release()

    def audit_projection(self) -> dict[str, Any]:
        value = super().audit_projection()
        value.update(
            {
                "completed_incremental_streams": self.completed_incremental_streams,
                "cancelled_incremental_streams": self.cancelled_incremental_streams,
                "active_incremental_streams": self.active_incremental_streams,
            }
        )
        return value


class TransformersCPUBackend:
    """Greedy ``GenerationMixin.generate`` backend for a preloaded CPU model."""

    def __init__(
        self,
        *,
        model_id: str,
        backend_fingerprint: str,
        model: Any,
        tokenizer: Any,
        maximum_prompt_tokens: int,
    ) -> None:
        if maximum_prompt_tokens <= 0:
            raise ValueError("maximum_prompt_tokens must be positive")
        self.model_id = model_id
        self.backend_fingerprint = backend_fingerprint
        self.model = model
        self.tokenizer = tokenizer
        self.maximum_prompt_tokens = maximum_prompt_tokens
        self._lock = asyncio.Lock()
        self._generation_call_count = 0
        self._last_execution: dict[str, Any] | None = None
        self.model.to("cpu")
        self.model.requires_grad_(False)
        self.model.eval()

    async def generate(self, request: ChatCompletionRequest) -> GeneratedCompletion:
        async with self._lock:
            result = await asyncio.to_thread(self._generate_sync, request)
            self._generation_call_count += 1
            self._last_execution = {
                "prompt_token_count": result.prompt_token_count,
                "completion_token_ids": list(result.completion_token_ids),
                "completion_text_fingerprint": "sha256:"
                + artifact_fingerprint({"text": result.text}),
                "finish_reason": result.finish_reason,
            }
            return result

    def _generate_sync(self, request: ChatCompletionRequest) -> GeneratedCompletion:
        try:
            import torch
            from transformers import GenerationConfig
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise RuntimeError("torch and transformers are required") from error
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]
        raw_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(raw_ids, torch.Tensor):
            raise RuntimeError("chat template did not return a tensor")
        input_ids = raw_ids.to(device="cpu", dtype=torch.long)
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or not 1 <= input_ids.shape[1] <= self.maximum_prompt_tokens
        ):
            raise OpenAIRequestError(
                422, "prompt_too_long", "Prompt exceeds the reviewed token limit"
            )
        if torch.any(input_ids < 0) or torch.any(input_ids >= len(self.tokenizer)):
            raise RuntimeError("chat template returned an out-of-vocabulary token id")
        attention_mask = torch.ones_like(input_ids)
        generation_config = GenerationConfig(  # type: ignore[no-untyped-call]
            do_sample=False,
            max_new_tokens=request.max_tokens,
            repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            bos_token_id=None,
        )
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
        sequences = generated.sequences
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
            raise RuntimeError("generate returned invalid sequences")
        continuation = sequences[0, input_ids.shape[1] :].to(dtype=torch.long)
        completion_ids = tuple(int(value) for value in continuation.tolist())
        if not completion_ids or len(completion_ids) > request.max_tokens:
            raise RuntimeError("generate returned an invalid completion length")
        text = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(text, str) or not text:
            raise RuntimeError("tokenizer returned an empty completion")
        deltas: list[str] = []
        previous = ""
        for index in range(1, len(completion_ids) + 1):
            current = self.tokenizer.decode(
                completion_ids[:index],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(current, str) or not current.startswith(previous):
                raise RuntimeError("tokenizer prefix decode is not append-only")
            delta = current[len(previous) :]
            if delta:
                deltas.append(delta)
            previous = current
        if previous != text or not deltas:
            raise RuntimeError("stream deltas do not reconstruct the completion")
        eos_value = getattr(self.tokenizer, "eos_token_id", None)
        finish_reason = "stop" if completion_ids[-1] == eos_value else "length"
        return GeneratedCompletion(
            text=text,
            text_deltas=tuple(deltas),
            prompt_token_count=int(input_ids.shape[1]),
            completion_token_ids=completion_ids,
            finish_reason=finish_reason,
        )

    def audit_projection(self) -> Mapping[str, Any]:
        return {
            "implementation": "transformers.GenerationMixin.generate",
            "device": "cpu",
            "generation_call_count": self._generation_call_count,
            "last_execution": (
                None if self._last_execution is None else dict(self._last_execution)
            ),
        }


def build_reference_app(
    backend: CompletionBackend,
    *,
    bearer_token: str,
    maximum_new_tokens: int,
) -> Starlette:
    """Build the Starlette application and expose its service via ``app.state``."""

    service = OpenAIReferenceService(
        backend,
        bearer_token=bearer_token,
        maximum_new_tokens=maximum_new_tokens,
    )
    app = Starlette(
        routes=[
            Route(HEALTH_PATH, service.health, methods=["GET"]),
            Route(MODELS_PATH, service.models, methods=["GET"]),
            Route(
                CHAT_COMPLETIONS_PATH,
                service.chat_completions,
                methods=["POST"],
            ),
        ]
    )
    app.state.reference_service = service
    return app


def build_incremental_reference_app(
    backend: IncrementalCompletionBackend,
    *,
    bearer_token: str,
    maximum_new_tokens: int,
) -> Starlette:
    """Build a streaming-only app whose backend iterator receives disconnect cancel."""

    service = IncrementalOpenAIReferenceService(
        backend,
        bearer_token=bearer_token,
        maximum_new_tokens=maximum_new_tokens,
    )
    app = Starlette(
        routes=[
            Route(HEALTH_PATH, service.health, methods=["GET"]),
            Route(MODELS_PATH, service.models, methods=["GET"]),
            Route(
                CHAT_COMPLETIONS_PATH,
                service.chat_completions,
                methods=["POST"],
            ),
        ]
    )
    app.state.reference_service = service
    return app
