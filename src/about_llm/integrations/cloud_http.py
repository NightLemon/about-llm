"""Async HTTP execution for strict cloud JSON contracts.

The executor owns retry orchestration but not the ``httpx.AsyncClient``. It
never follows redirects and requires an exact origin allowlist before any
request is built or sent.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import httpx

from about_llm.inference.sse import SSEDecoder, SSELimitError, SSEProtocolError
from about_llm.integrations.cloud_api import RequestSpec
from about_llm.integrations.cloud_stream import (
    StreamProtocolError,
    StreamUpdate,
    TextStreamState,
)
from about_llm.integrations.retry import (
    ErrorCategory,
    RetryDecision,
    RetryPolicy,
    decide_retry,
)

FailureKind = Literal[
    "http_status",
    "timeout",
    "transport",
    "invalid_response",
    "response_too_large",
    "deadline_exhausted",
]


@dataclass(frozen=True)
class HttpExecutorConfig:
    """Local transport bounds; both timeouts include waiting for response bytes."""

    allowed_origins: frozenset[str]
    deadline_seconds: float = 30.0
    request_timeout_seconds: float = 20.0
    max_response_bytes: int = 4 * 1024 * 1024
    require_json_content_type: bool = True
    require_https: bool = True
    allow_query: bool = False
    request_id_header_names: tuple[str, ...] = ("request-id", "x-request-id")

    def __post_init__(self) -> None:
        if not self.allowed_origins:
            raise ValueError("allowed_origins must not be empty")
        origins = frozenset(
            _canonical_origin(value, origin_only=True) for value in self.allowed_origins
        )
        for name, value in (
            ("deadline_seconds", self.deadline_seconds),
            ("request_timeout_seconds", self.request_timeout_seconds),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if (
            not isinstance(self.max_response_bytes, int)
            or isinstance(self.max_response_bytes, bool)
            or self.max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        for name, value in (
            ("require_json_content_type", self.require_json_content_type),
            ("require_https", self.require_https),
            ("allow_query", self.allow_query),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if self.require_https and any(not origin.startswith("https://") for origin in origins):
            raise ValueError("allowed_origins must use HTTPS when require_https is true")
        normalized_names: list[str] = []
        for name in self.request_id_header_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("request id header names must be non-empty strings")
            normalized = name.lower()
            if normalized in normalized_names:
                raise ValueError("request id header names must be unique case-insensitively")
            normalized_names.append(normalized)
        object.__setattr__(self, "allowed_origins", origins)
        object.__setattr__(self, "request_id_header_names", tuple(normalized_names))


@dataclass(frozen=True)
class AttemptTrace:
    attempt: int
    started_after_seconds: float
    duration_seconds: float
    status_code: int | None
    failure_kind: FailureKind | None
    error_category: ErrorCategory | None
    outcome_uncertain: bool
    request_id: str | None
    retry_decision: RetryDecision | None


@dataclass(frozen=True)
class CloudHttpResult:
    payload: Mapping[str, Any]
    status_code: int
    attempts: tuple[AttemptTrace, ...]


@dataclass(frozen=True)
class CloudStreamResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    status_code: int
    bytes_received: int
    event_count: int
    update_count: int
    attempts: tuple[AttemptTrace, ...]


class CloudCallError(RuntimeError):
    """Sanitized terminal failure with structured attempt evidence."""

    def __init__(self, reason: str, attempts: Sequence[AttemptTrace]) -> None:
        self.reason = reason
        self.attempts = tuple(attempts)
        super().__init__(f"cloud call failed: {reason}; attempts={len(self.attempts)}")


def validate_request_target(
    request: RequestSpec, config: HttpExecutorConfig
) -> None:
    """Validate the complete outbound target policy without sending a request."""

    if not isinstance(request, RequestSpec):
        raise TypeError("request must be RequestSpec")
    if not isinstance(config, HttpExecutorConfig):
        raise TypeError("config must be HttpExecutorConfig")
    _validate_request_target(request, config)


async def execute_json_request(
    *,
    client: httpx.AsyncClient,
    request: RequestSpec,
    policy: RetryPolicy,
    config: HttpExecutorConfig,
    replay_safe: bool,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], datetime] | None = None,
    jitter: Callable[[], float] = random.random,
) -> CloudHttpResult:
    """Execute a bounded JSON POST and return strict object JSON plus attempt trace.

    Connect/pool failures are classified as outcome-known and may be retried.
    Write/read/protocol/overall-attempt timeouts are outcome-uncertain and stop
    automatically. Cancellation is never converted into a retry.
    """
    if not isinstance(replay_safe, bool):
        raise ValueError("replay_safe must be a boolean")
    _validate_request_target(request, config)
    wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
    started = _finite_monotonic(monotonic())
    deadline = started + config.deadline_seconds
    attempts: list[AttemptTrace] = []
    attempt = 1

    while True:
        attempt_started = _finite_monotonic(monotonic())
        _require_non_decreasing(attempt_started, started)
        remaining = deadline - attempt_started
        if remaining <= 0:
            raise CloudCallError("deadline_exhausted", attempts)
        timeout = min(config.request_timeout_seconds, remaining)
        outbound = client.build_request(
            "POST",
            request.url,
            headers=dict(request.headers),
            json=dict(request.body),
        )
        try:
            response = await asyncio.wait_for(
                client.send(outbound, follow_redirects=False), timeout=timeout
            )
        except asyncio.CancelledError:
            raise
        except (TimeoutError, asyncio.TimeoutError, httpx.HTTPError) as error:
            ended = _finite_monotonic(monotonic())
            duration = _elapsed(ended, attempt_started)
            error_category, failure_kind, outcome_uncertain = _classify_transport_error(
                error
            )
            decision = decide_retry(
                policy=policy,
                attempt=attempt,
                replay_safe=replay_safe,
                outcome_uncertain=outcome_uncertain,
                error_category=error_category,
                remaining_seconds=max(0.0, deadline - ended),
                jitter_fraction=jitter(),
            )
            attempts.append(
                AttemptTrace(
                    attempt=attempt,
                    started_after_seconds=attempt_started - started,
                    duration_seconds=duration,
                    status_code=None,
                    failure_kind=failure_kind,
                    error_category=error_category,
                    outcome_uncertain=outcome_uncertain,
                    request_id=None,
                    retry_decision=decision,
                )
            )
            if not decision.retry:
                raise CloudCallError(decision.reason, attempts) from None
            await sleep(cast(float, decision.delay_seconds))
            attempt += 1
            continue

        ended = _finite_monotonic(monotonic())
        duration = _elapsed(ended, attempt_started)
        request_id = _request_id(response, config.request_id_header_names)
        if not 200 <= response.status_code < 300:
            decision = decide_retry(
                policy=policy,
                attempt=attempt,
                replay_safe=replay_safe,
                outcome_uncertain=False,
                status_code=response.status_code,
                response_headers=_retry_after_headers(response),
                now=wall_clock(),
                remaining_seconds=max(0.0, deadline - ended),
                jitter_fraction=jitter(),
            )
            attempts.append(
                AttemptTrace(
                    attempt=attempt,
                    started_after_seconds=attempt_started - started,
                    duration_seconds=duration,
                    status_code=response.status_code,
                    failure_kind="http_status",
                    error_category=None,
                    outcome_uncertain=False,
                    request_id=request_id,
                    retry_decision=decision,
                )
            )
            if not decision.retry:
                raise CloudCallError(decision.reason, attempts)
            await sleep(cast(float, decision.delay_seconds))
            attempt += 1
            continue

        terminal = AttemptTrace(
            attempt=attempt,
            started_after_seconds=attempt_started - started,
            duration_seconds=duration,
            status_code=response.status_code,
            failure_kind=None,
            error_category=None,
            outcome_uncertain=False,
            request_id=request_id,
            retry_decision=None,
        )
        attempts.append(terminal)
        try:
            payload = _strict_response_object(response, config=config)
        except _ResponseValidationError as error:
            attempts[-1] = AttemptTrace(
                attempt=terminal.attempt,
                started_after_seconds=terminal.started_after_seconds,
                duration_seconds=terminal.duration_seconds,
                status_code=terminal.status_code,
                failure_kind=error.kind,
                error_category=None,
                outcome_uncertain=False,
                request_id=terminal.request_id,
                retry_decision=None,
            )
            raise CloudCallError(error.kind, attempts) from error
        return CloudHttpResult(
            payload=MappingProxyType(payload),
            status_code=response.status_code,
            attempts=tuple(attempts),
        )


async def execute_sse_request(
    *,
    client: httpx.AsyncClient,
    request: RequestSpec,
    state: TextStreamState,
    policy: RetryPolicy,
    config: HttpExecutorConfig,
    replay_safe: bool,
    on_update: Callable[[StreamUpdate], Awaitable[None]] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], datetime] | None = None,
    jitter: Callable[[], float] = random.random,
    max_line_bytes: int = 64 * 1024,
    max_event_bytes: int = 1024 * 1024,
    max_total_bytes: int = 16 * 1024 * 1024,
) -> CloudStreamResult:
    """Consume one provider text stream with bounded byte framing.

    Retries are possible only before a successful response begins. Any timeout,
    transport failure, size failure, or protocol failure after a 2xx response is
    terminal; already delivered updates therefore remain partial output.
    """
    if not isinstance(replay_safe, bool):
        raise ValueError("replay_safe must be a boolean")
    _validate_request_target(request, config)
    decoder = SSEDecoder(
        max_line_bytes=max_line_bytes,
        max_event_bytes=max_event_bytes,
        max_total_bytes=max_total_bytes,
    )
    callback = on_update or _discard_stream_update
    wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
    started = _finite_monotonic(monotonic())
    deadline = started + config.deadline_seconds
    attempts: list[AttemptTrace] = []
    attempt = 1

    while True:
        attempt_started = _finite_monotonic(monotonic())
        _require_non_decreasing(attempt_started, started)
        remaining = deadline - attempt_started
        if remaining <= 0:
            raise CloudCallError("deadline_exhausted", attempts)
        outbound = client.build_request(
            "POST",
            request.url,
            headers=dict(request.headers),
            json=dict(request.body),
        )
        try:
            response = await asyncio.wait_for(
                client.send(outbound, stream=True, follow_redirects=False),
                timeout=min(config.request_timeout_seconds, remaining),
            )
        except asyncio.CancelledError:
            raise
        except (TimeoutError, asyncio.TimeoutError, httpx.HTTPError) as error:
            ended = _finite_monotonic(monotonic())
            category, kind, uncertain = _classify_transport_error(error)
            decision = decide_retry(
                policy=policy,
                attempt=attempt,
                replay_safe=replay_safe,
                outcome_uncertain=uncertain,
                error_category=category,
                remaining_seconds=max(0.0, deadline - ended),
                jitter_fraction=jitter(),
            )
            attempts.append(
                AttemptTrace(
                    attempt,
                    attempt_started - started,
                    _elapsed(ended, attempt_started),
                    None,
                    kind,
                    category,
                    uncertain,
                    None,
                    decision,
                )
            )
            if not decision.retry:
                raise CloudCallError(decision.reason, attempts) from None
            await sleep(cast(float, decision.delay_seconds))
            attempt += 1
            continue

        request_id = _request_id(response, config.request_id_header_names)
        if not 200 <= response.status_code < 300:
            await response.aclose()
            ended = _finite_monotonic(monotonic())
            decision = decide_retry(
                policy=policy,
                attempt=attempt,
                replay_safe=replay_safe,
                outcome_uncertain=False,
                status_code=response.status_code,
                response_headers=_retry_after_headers(response),
                now=wall_clock(),
                remaining_seconds=max(0.0, deadline - ended),
                jitter_fraction=jitter(),
            )
            attempts.append(
                AttemptTrace(
                    attempt,
                    attempt_started - started,
                    _elapsed(ended, attempt_started),
                    response.status_code,
                    "http_status",
                    None,
                    False,
                    request_id,
                    decision,
                )
            )
            if not decision.retry:
                raise CloudCallError(decision.reason, attempts)
            await sleep(cast(float, decision.delay_seconds))
            attempt += 1
            continue
        if not _is_sse_content_type(response.headers.get("content-type", "")):
            await response.aclose()
            ended = _finite_monotonic(monotonic())
            attempts.append(
                _stream_failure_trace(
                    attempt,
                    started,
                    attempt_started,
                    ended,
                    response.status_code,
                    "invalid_response",
                    request_id,
                )
            )
            raise CloudCallError("invalid_response", attempts)
        break

    text_parts: list[str] = []
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    bytes_received = 0
    event_count = 0
    update_count = 0

    async def publish(updates: tuple[StreamUpdate, ...]) -> None:
        nonlocal input_tokens, output_tokens, finish_reason, update_count
        for update in updates:
            remaining = deadline - _finite_monotonic(monotonic())
            if remaining <= 0:
                raise TimeoutError
            await asyncio.wait_for(callback(update), timeout=remaining)
            update_count += 1
            if update.text is not None:
                text_parts.append(update.text)
            if update.input_tokens is not None:
                input_tokens = update.input_tokens
            if update.output_tokens is not None:
                output_tokens = update.output_tokens
            if update.finish_reason is not None:
                finish_reason = update.finish_reason

    try:
        iterator = response.aiter_bytes().__aiter__()
        while True:
            now = _finite_monotonic(monotonic())
            remaining = deadline - now
            if remaining <= 0:
                raise TimeoutError
            try:
                chunk = await asyncio.wait_for(
                    anext(iterator),
                    timeout=min(config.request_timeout_seconds, remaining),
                )
            except StopAsyncIteration:
                break
            bytes_received += len(chunk)
            events = decoder.feed(chunk)
            event_count += len(events)
            for event in events:
                await publish(state.consume(event))
        final_events = decoder.finish()
        event_count += len(final_events)
        for event in final_events:
            await publish(state.consume(event))
        await publish(state.finish())
    except asyncio.CancelledError:
        raise
    except SSELimitError as error:
        ended = _finite_monotonic(monotonic())
        attempts.append(
            _stream_failure_trace(
                attempt, started, attempt_started, ended, response.status_code,
                "response_too_large", request_id,
            )
        )
        raise CloudCallError("response_too_large", attempts) from error
    except (SSEProtocolError, StreamProtocolError) as error:
        ended = _finite_monotonic(monotonic())
        attempts.append(
            _stream_failure_trace(
                attempt, started, attempt_started, ended, response.status_code,
                "invalid_response", request_id,
            )
        )
        raise CloudCallError("invalid_response", attempts) from error
    except (TimeoutError, asyncio.TimeoutError, httpx.HTTPError) as error:
        ended = _finite_monotonic(monotonic())
        category, kind, _ = _classify_transport_error(error)
        decision = decide_retry(
            policy=policy,
            attempt=attempt,
            replay_safe=replay_safe,
            outcome_uncertain=True,
            error_category=category,
            remaining_seconds=max(0.0, deadline - ended),
            jitter_fraction=jitter(),
        )
        attempts.append(
            AttemptTrace(
                attempt,
                attempt_started - started,
                _elapsed(ended, attempt_started),
                response.status_code,
                kind,
                category,
                True,
                request_id,
                decision,
            )
        )
        raise CloudCallError(decision.reason, attempts) from None
    finally:
        await response.aclose()

    ended = _finite_monotonic(monotonic())
    attempts.append(
        AttemptTrace(
            attempt,
            attempt_started - started,
            _elapsed(ended, attempt_started),
            response.status_code,
            None,
            None,
            False,
            request_id,
            None,
        )
    )
    return CloudStreamResult(
        text="".join(text_parts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        status_code=response.status_code,
        bytes_received=bytes_received,
        event_count=event_count,
        update_count=update_count,
        attempts=tuple(attempts),
    )


class _ResponseValidationError(ValueError):
    def __init__(self, kind: Literal["invalid_response", "response_too_large"]) -> None:
        self.kind = kind
        super().__init__(kind)


async def _discard_stream_update(_: StreamUpdate) -> None:
    return None


def _stream_failure_trace(
    attempt: int,
    started: float,
    attempt_started: float,
    ended: float,
    status_code: int,
    kind: Literal["invalid_response", "response_too_large"],
    request_id: str | None,
) -> AttemptTrace:
    return AttemptTrace(
        attempt=attempt,
        started_after_seconds=attempt_started - started,
        duration_seconds=_elapsed(ended, attempt_started),
        status_code=status_code,
        failure_kind=kind,
        error_category=None,
        outcome_uncertain=True,
        request_id=request_id,
        retry_decision=None,
    )


def _validate_request_target(request: RequestSpec, config: HttpExecutorConfig) -> None:
    request_origin = _canonical_origin(request.url, origin_only=False)
    if request_origin not in config.allowed_origins:
        raise ValueError("request origin is not in allowed_origins")
    if urlsplit(request.url).query and not config.allow_query:
        raise ValueError("request URL query is disabled; keep credentials out of URLs")


def _strict_response_object(
    response: httpx.Response, *, config: HttpExecutorConfig
) -> dict[str, Any]:
    if len(response.content) > config.max_response_bytes:
        raise _ResponseValidationError("response_too_large")
    content_type = response.headers.get("content-type", "")
    if config.require_json_content_type and not _is_json_content_type(content_type):
        raise _ResponseValidationError("invalid_response")

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON constant")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON float")
        return parsed

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate JSON object key")
            result[name] = value
        return result

    try:
        payload = json.loads(
            response.content,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _ResponseValidationError("invalid_response") from error
    if not isinstance(payload, dict):
        raise _ResponseValidationError("invalid_response")
    return cast(dict[str, Any], payload)


def _classify_transport_error(
    error: BaseException,
) -> tuple[ErrorCategory, Literal["timeout", "transport"], bool]:
    if isinstance(error, httpx.PoolTimeout):
        return "timeout", "timeout", False
    if isinstance(error, httpx.ConnectTimeout):
        return "timeout", "timeout", False
    if isinstance(error, httpx.ConnectError):
        return "transport", "transport", False
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
        return "timeout", "timeout", True
    return "transport", "transport", True


def _retry_after_headers(response: httpx.Response) -> Mapping[str, str] | None:
    values = response.headers.get_list("retry-after")
    if not values:
        return None
    return {"Retry-After": values[0] if len(values) == 1 else ",".join(values)}


def _request_id(response: httpx.Response, names: Sequence[str]) -> str | None:
    for name in names:
        value = response.headers.get(name)
        if value and len(value) <= 256:
            return cast(str, value)
    return None


def _is_json_content_type(value: str) -> bool:
    media_type = value.split(";", maxsplit=1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _is_sse_content_type(value: str) -> bool:
    return value.split(";", maxsplit=1)[0].strip().lower() == "text/event-stream"


def _finite_monotonic(value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise RuntimeError("monotonic clock returned a non-finite value")
    return float(value)


def _require_non_decreasing(current: float, previous: float) -> None:
    if current < previous:
        raise RuntimeError("monotonic clock moved backwards")


def _elapsed(ended: float, started: float) -> float:
    _require_non_decreasing(ended, started)
    return ended - started


def _canonical_origin(value: str, *, origin_only: bool) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("origin/URL must be a non-empty string")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as error:
        raise ValueError("origin/URL is invalid") from error
    if parts.scheme.lower() not in {"http", "https"} or parts.hostname is None:
        raise ValueError("origin/URL must be absolute HTTP(S)")
    if parts.username is not None or parts.password is not None or parts.fragment:
        raise ValueError("origin/URL must not contain userinfo or fragment")
    if origin_only and (parts.path not in {"", "/"} or parts.query):
        raise ValueError("allowed origin must not contain a path or query")
    host = parts.hostname
    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            normalized_host = host.encode("idna").decode("ascii").lower()
        except UnicodeError as error:
            raise ValueError("origin/URL hostname is invalid") from error
    else:
        normalized_host = f"[{parsed_ip.compressed}]" if parsed_ip.version == 6 else str(parsed_ip)
    scheme = parts.scheme.lower()
    default_port = 443 if scheme == "https" else 80
    port_suffix = "" if port is None or port == default_port else f":{port}"
    return f"{scheme}://{normalized_host}{port_suffix}"
