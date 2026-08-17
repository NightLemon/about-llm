"""Strict MCP Streamable HTTP loopback control for interoperability study.

This module implements a deliberately small projection of MCP 2025-11-25 over
real IPv4 loopback TCP/HTTP.  It covers transport and lifecycle invariants, but
it is not the official MCP SDK, an OAuth implementation, or a conformance suite.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import math
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn, cast

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from about_llm.agents.mcp_stdio import (
    ADD_INPUT_CONTRACT,
    MCP_PROTOCOL_VERSION,
    MCPStdioSession,
    SessionPhase,
)
from about_llm.agents.schema import DRAFT_2020_12_URI
from about_llm.inference.sse import SSEDecoder, SSEEvent, parse_sse_json_object
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

MCP_STREAMABLE_HTTP_CONTROL_VERSION: Final = (
    "about-llm.mcp-streamable-http-control.v1"
)
MCP_ENDPOINT_PATH: Final = "/mcp"
CONTROL_TOKEN_ENV: Final = "ABOUT_LLM_MCP_HTTP_CONTROL_TOKEN"
CONTROL_ORIGIN_ENV: Final = "ABOUT_LLM_MCP_HTTP_CONTROL_ORIGIN"
CONTROL_ALLOWED_ORIGIN: Final = "https://client.example.invalid"
MAX_HTTP_BODY_BYTES: Final = 64_000
MAX_SSE_BYTES: Final = 128_000
REQUEST_TIMEOUT_SECONDS: Final = 5.0
SERVER_START_TIMEOUT_SECONDS: Final = 10.0
WAIT_SERVER_TIMEOUT_SECONDS: Final = 10.0

INVALID_REQUEST = -32600
INVALID_PARAMS = -32602


class MCPHTTPBodyError(ValueError):
    """An HTTP request body is not one bounded strict JSON object."""


class MCPHTTPControlError(ValueError):
    """A stable HTTP-layer rejection used by the fixture endpoint."""

    def __init__(
        self,
        status_code: int,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = dict(headers or {})
        super().__init__(f"MCP HTTP request rejected with status {status_code}")


def _reject_nonfinite(value: str) -> NoReturn:
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


def decode_http_json(raw: bytes) -> dict[str, Any]:
    """Decode one strict, finite UTF-8 JSON object within the body limit."""

    if len(raw) > MAX_HTTP_BODY_BYTES:
        raise MCPHTTPBodyError("MCP HTTP body exceeds the byte limit")
    if not raw:
        raise MCPHTTPBodyError("MCP HTTP body cannot be empty")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise MCPHTTPBodyError("MCP HTTP body is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise MCPHTTPBodyError("MCP HTTP body must be a JSON object")
    return cast(dict[str, Any], value)


def _valid_request_id(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
        value, str
    )


def _visible_ascii(value: str) -> bool:
    return bool(value) and all(0x21 <= ord(character) <= 0x7E for character in value)


def _header_values(request: Request, name: str) -> list[str]:
    expected = name.lower().encode("ascii")
    raw_headers = cast(Sequence[tuple[bytes, bytes]], request.scope["headers"])
    return [value.decode("latin-1") for key, value in raw_headers if key == expected]


def _single_header(request: Request, name: str) -> str | None:
    values = _header_values(request, name)
    if len(values) > 1:
        raise MCPHTTPControlError(400)
    return values[0] if values else None


def _listed_media_types(value: str) -> frozenset[str]:
    accepted: set[str] = set()
    for item in value.split(","):
        pieces = [piece.strip() for piece in item.split(";")]
        media_type = pieces[0].lower()
        if not media_type:
            continue
        quality = 1.0
        for parameter in pieces[1:]:
            name, separator, raw_value = parameter.partition("=")
            if separator and name.strip().lower() == "q":
                try:
                    quality = float(raw_value.strip())
                except ValueError:
                    quality = 0.0
        if quality > 0:
            accepted.add(media_type)
    return frozenset(accepted)


async def _bounded_request_body(request: Request) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_HTTP_BODY_BYTES:
            raise MCPHTTPControlError(413)
    return bytes(body)


def _json_response(message: Mapping[str, Any], status_code: int = 200) -> Response:
    return Response(
        canonical_json_bytes(message),
        status_code=status_code,
        media_type="application/json",
    )


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _is_client_response(message: Mapping[str, Any]) -> bool:
    keys = frozenset(message)
    if message.get("jsonrpc") != "2.0" or not _valid_request_id(message.get("id")):
        return False
    if "result" in message:
        return keys == frozenset({"jsonrpc", "id", "result"})
    if "error" in message:
        error = message.get("error")
        return keys == frozenset({"jsonrpc", "id", "error"}) and isinstance(
            error, Mapping
        )
    return False


WAIT_TOOL = {
    "name": "fixture.wait",
    "title": "Cancellable fixture wait",
    "description": "Wait until an explicit MCP cancellation notification is received.",
    "inputSchema": {
        "$schema": DRAFT_2020_12_URI,
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    "execution": {"taskSupport": "forbidden"},
}


@dataclass
class PendingWait:
    """One in-flight fixture request that may be explicitly cancelled."""

    request_id: int | str
    cancelled: asyncio.Event


class MCPHTTPSession:
    """One stateful MCP lifecycle plus HTTP-only cancellation bookkeeping."""

    def __init__(self) -> None:
        self.protocol = MCPStdioSession()
        self.pending: dict[int | str, PendingWait] = {}
        self._event_counter = 0

    @property
    def phase(self) -> SessionPhase:
        return self.protocol.phase

    def next_event_id(self, stream: str) -> str:
        self._event_counter += 1
        return f"{stream}.{self._event_counter}.{secrets.token_hex(8)}"

    def process_message(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        if message.get("method") == "notifications/cancelled" and "id" not in message:
            self._cancel(message)
            return None
        response = self.protocol.process_message(message)
        if (
            response is not None
            and message.get("method") == "tools/list"
            and isinstance(response.get("result"), dict)
        ):
            result = cast(dict[str, Any], response["result"])
            tools = result.get("tools")
            if isinstance(tools, list):
                tools.append(json.loads(canonical_json_bytes(WAIT_TOOL)))
        return response

    def begin_wait(self, message: Mapping[str, Any]) -> PendingWait | dict[str, Any]:
        request_id = message.get("id")
        if self.phase is not SessionPhase.READY:
            return _error_response(request_id, INVALID_REQUEST, "Session not initialized")
        if not _valid_request_id(request_id):
            return _error_response(None, INVALID_REQUEST, "Invalid Request")
        params = message.get("params")
        if (
            frozenset(message) != frozenset({"jsonrpc", "id", "method", "params"})
            or message.get("jsonrpc") != "2.0"
            or message.get("method") != "tools/call"
            or not isinstance(params, Mapping)
            or frozenset(params) != frozenset({"name", "arguments"})
            or params.get("name") != WAIT_TOOL["name"]
            or params.get("arguments") != {}
        ):
            return _error_response(request_id, INVALID_PARAMS, "Invalid wait params")
        typed_id = cast(int | str, request_id)
        if typed_id in self.pending:
            return _error_response(request_id, INVALID_REQUEST, "Request already pending")
        pending = PendingWait(typed_id, asyncio.Event())
        self.pending[typed_id] = pending
        return pending

    def finish_wait(self, request_id: int | str) -> None:
        self.pending.pop(request_id, None)

    def terminate(self) -> None:
        for pending in self.pending.values():
            pending.cancelled.set()
        self.pending.clear()

    def _cancel(self, message: Mapping[str, Any]) -> None:
        # Cancellation is fire-and-forget.  The protocol says malformed, unknown,
        # completed, or otherwise un-cancellable references should be ignored.
        if self.phase is not SessionPhase.READY:
            return
        if frozenset(message) - frozenset({"jsonrpc", "method", "params"}):
            return
        if message.get("jsonrpc") != "2.0":
            return
        params = message.get("params")
        if not isinstance(params, Mapping):
            return
        if not frozenset(params).issubset(frozenset({"requestId", "reason"})):
            return
        request_id = params.get("requestId")
        reason = params.get("reason")
        if not _valid_request_id(request_id) or (
            reason is not None and not isinstance(reason, str)
        ):
            return
        pending = self.pending.get(cast(int | str, request_id))
        if pending is not None:
            pending.cancelled.set()


class MCPStreamableHTTPServer:
    """Strict single-endpoint Streamable HTTP fixture server."""

    def __init__(self, *, bearer_token: str, allowed_origin: str) -> None:
        if len(bearer_token) < 32 or not _visible_ascii(bearer_token):
            raise ValueError("fixture bearer token must be at least 32 visible ASCII bytes")
        if not allowed_origin.startswith("https://"):
            raise ValueError("fixture allowed origin must use HTTPS")
        self._bearer_token = bearer_token
        self._allowed_origin = allowed_origin
        self._sessions: dict[str, MCPHTTPSession] = {}

    async def endpoint(self, request: Request) -> Response:
        try:
            self._validate_origin(request)
            self._validate_authentication(request)
            if request.method == "POST":
                return await self._post(request)
            if request.method == "GET":
                return self._get(request)
            if request.method == "DELETE":
                return self._delete(request)
            return Response(status_code=405, headers={"Allow": "POST, GET, DELETE"})
        except MCPHTTPControlError as error:
            return Response(status_code=error.status_code, headers=error.headers)

    def _validate_origin(self, request: Request) -> None:
        origins = _header_values(request, "origin")
        if len(origins) > 1 or (origins and origins[0] != self._allowed_origin):
            raise MCPHTTPControlError(403)

    def _validate_authentication(self, request: Request) -> None:
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
            raise MCPHTTPControlError(
                401,
                headers={"WWW-Authenticate": 'Bearer realm="about-llm-mcp-control"'},
            )

    def _validate_accept(self, request: Request, required: frozenset[str]) -> None:
        value = _single_header(request, "accept")
        if value is None or not required.issubset(_listed_media_types(value)):
            raise MCPHTTPControlError(406)

    def _require_session(self, request: Request) -> MCPHTTPSession:
        session_id = _single_header(request, "mcp-session-id")
        if session_id is None:
            raise MCPHTTPControlError(400)
        if not _visible_ascii(session_id):
            raise MCPHTTPControlError(400)
        session = self._sessions.get(session_id)
        if session is None:
            raise MCPHTTPControlError(404)
        version = _single_header(request, "mcp-protocol-version")
        if version != MCP_PROTOCOL_VERSION:
            raise MCPHTTPControlError(400)
        return session

    async def _post(self, request: Request) -> Response:
        self._validate_accept(
            request, frozenset({"application/json", "text/event-stream"})
        )
        content_type = _single_header(request, "content-type")
        if content_type is None or content_type.partition(";")[0].strip().lower() != (
            "application/json"
        ):
            raise MCPHTTPControlError(415)
        raw = await _bounded_request_body(request)
        try:
            message = decode_http_json(raw)
        except MCPHTTPBodyError as error:
            raise MCPHTTPControlError(400) from error

        session_id = _single_header(request, "mcp-session-id")
        if session_id is None:
            return self._initialize(message)
        session = self._require_session(request)

        if _is_client_response(message):
            return Response(status_code=202)

        if self._is_wait_call(message):
            wait = session.begin_wait(message)
            if isinstance(wait, dict):
                return _json_response(wait)
            return StreamingResponse(
                self._cancelled_wait_stream(session, wait),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store"},
            )

        response = session.process_message(message)
        if response is None:
            return Response(status_code=202)
        if self._use_sse_response(message):
            return StreamingResponse(
                self._single_response_stream(session, response),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store"},
            )
        return _json_response(response)

    def _initialize(self, message: Mapping[str, Any]) -> Response:
        if message.get("method") != "initialize" or "id" not in message:
            raise MCPHTTPControlError(400)
        session = MCPHTTPSession()
        response = session.process_message(message)
        if response is None:
            raise MCPHTTPControlError(400)
        if "result" not in response:
            return _json_response(response)
        session_id = secrets.token_urlsafe(32)
        if not _visible_ascii(session_id):  # pragma: no cover - stdlib invariant
            raise RuntimeError("generated MCP session id is not visible ASCII")
        while session_id in self._sessions:  # pragma: no cover - collision defense
            session_id = secrets.token_urlsafe(32)
        self._sessions[session_id] = session
        response_object = _json_response(response)
        response_object.headers["MCP-Session-Id"] = session_id
        return response_object

    def _get(self, request: Request) -> Response:
        self._validate_accept(request, frozenset({"text/event-stream"}))
        session = self._require_session(request)
        if _single_header(request, "last-event-id") is not None:
            # This fixture does not retain an event store and therefore cannot
            # safely claim resumability or replay a guessed stream.
            raise MCPHTTPControlError(400)
        return StreamingResponse(
            self._listen_stream(session),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )

    def _delete(self, request: Request) -> Response:
        session_id = _single_header(request, "mcp-session-id")
        session = self._require_session(request)
        if session_id is None:  # pragma: no cover - enforced by _require_session
            raise MCPHTTPControlError(400)
        session.terminate()
        del self._sessions[session_id]
        return Response(status_code=204)

    @staticmethod
    def _is_wait_call(message: Mapping[str, Any]) -> bool:
        params = message.get("params")
        return (
            message.get("method") == "tools/call"
            and isinstance(params, Mapping)
            and params.get("name") == WAIT_TOOL["name"]
        )

    @staticmethod
    def _use_sse_response(message: Mapping[str, Any]) -> bool:
        params = message.get("params")
        return (
            message.get("method") == "tools/call"
            and isinstance(params, Mapping)
            and params.get("name") == ADD_INPUT_CONTRACT.name
        )

    @staticmethod
    async def _single_response_stream(
        session: MCPHTTPSession, response: Mapping[str, Any]
    ) -> AsyncIterator[bytes]:
        yield _sse_event(session.next_event_id("post"), "")
        yield _sse_event(
            session.next_event_id("post"),
            canonical_json_bytes(response).decode("utf-8"),
        )

    @staticmethod
    async def _listen_stream(session: MCPHTTPSession) -> AsyncIterator[bytes]:
        yield _sse_event(session.next_event_id("get"), "")

    @staticmethod
    async def _cancelled_wait_stream(
        session: MCPHTTPSession, pending: PendingWait
    ) -> AsyncIterator[bytes]:
        yield _sse_event(session.next_event_id("wait"), "")
        try:
            await asyncio.wait_for(
                pending.cancelled.wait(), timeout=WAIT_SERVER_TIMEOUT_SECONDS
            )
            # MCP cancellation asks the receiver not to send a JSON-RPC response.
            return
        except asyncio.TimeoutError:
            timeout_response = _error_response(
                pending.request_id, -32000, "Fixture wait timed out"
            )
            yield _sse_event(
                session.next_event_id("wait"),
                canonical_json_bytes(timeout_response).decode("utf-8"),
            )
        finally:
            session.finish_wait(pending.request_id)


def _sse_event(event_id: str, data: str) -> bytes:
    if not _visible_ascii(event_id) or "\r" in event_id or "\n" in event_id:
        raise ValueError("invalid SSE event id")
    if "\r" in data or "\n" in data:
        raise ValueError("fixture SSE data must fit on one line")
    return f"id: {event_id}\ndata: {data}\n\n".encode()


def build_server_app(*, bearer_token: str, allowed_origin: str) -> Starlette:
    """Build the strict single-endpoint fixture ASGI application."""

    server = MCPStreamableHTTPServer(
        bearer_token=bearer_token,
        allowed_origin=allowed_origin,
    )
    return Starlette(
        routes=[
            Route(
                MCP_ENDPOINT_PATH,
                server.endpoint,
                methods=["POST", "GET", "DELETE"],
            )
        ]
    )


def serve(host: str, port: int, *, bearer_token: str, allowed_origin: str) -> int:
    """Run the fixture server, restricted to IPv4 loopback."""

    if host != "127.0.0.1":
        raise ValueError("the MCP HTTP control only permits IPv4 loopback")
    if not 0 < port < 65_536:
        raise ValueError("port must be between 1 and 65535")
    logging.disable(logging.CRITICAL)
    config = uvicorn.Config(
        build_server_app(
            bearer_token=bearer_token,
            allowed_origin=allowed_origin,
        ),
        host=host,
        port=port,
        access_log=False,
        log_level="critical",
        lifespan="off",
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
    return 0


def _reserve_candidate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        address = cast(tuple[str, int], candidate.getsockname())
        return address[1]


async def _wait_until_ready(
    host: str,
    port: int,
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("MCP HTTP server exited before readiness")
        try:
            reader, writer = await asyncio.open_connection(host, port)
            del reader
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise TimeoutError("MCP HTTP server readiness timed out")


def _base_headers(token: str, *, include_origin: bool = True) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if include_origin:
        headers["Origin"] = CONTROL_ALLOWED_ORIGIN
    return headers


def _subsequent_headers(token: str, session_id: str) -> dict[str, str]:
    return {
        **_base_headers(token),
        "MCP-Session-Id": session_id,
        "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
    }


def _decode_json_response(response: httpx.Response) -> dict[str, Any]:
    if response.headers.get("content-type", "").partition(";")[0] != "application/json":
        raise ValueError("MCP response is not application/json")
    return decode_http_json(response.content)


def _expect_result(response: Mapping[str, Any], request_id: int) -> Mapping[str, Any]:
    if (
        response.get("jsonrpc") != "2.0"
        or response.get("id") != request_id
        or "error" in response
        or not isinstance(response.get("result"), Mapping)
    ):
        raise ValueError(f"MCP request {request_id} did not return a valid result")
    return cast(Mapping[str, Any], response["result"])


async def _read_sse_response(response: httpx.Response) -> tuple[SSEEvent, ...]:
    content_type = response.headers.get("content-type", "").partition(";")[0]
    if content_type != "text/event-stream":
        raise ValueError("MCP response is not text/event-stream")
    decoder = SSEDecoder(
        max_line_bytes=MAX_SSE_BYTES,
        max_event_bytes=MAX_SSE_BYTES,
        max_total_bytes=MAX_SSE_BYTES,
    )
    events: list[SSEEvent] = []
    async for chunk in response.aiter_bytes():
        events.extend(decoder.feed(chunk))
    events.extend(decoder.finish())
    return tuple(events)


def _validate_prime(event: SSEEvent) -> None:
    if not event.last_event_id or event.data != "" or event.event != "message":
        raise ValueError("MCP SSE stream did not start with an id-bearing empty event")


async def _run_control_async(
    *,
    cwd: Path | None,
    request_timeout_seconds: float,
    server_start_timeout_seconds: float,
) -> dict[str, Any]:
    host = "127.0.0.1"
    port = _reserve_candidate_port()
    endpoint = f"http://{host}:{port}{MCP_ENDPOINT_PATH}"
    bearer_token = secrets.token_urlsafe(32)
    rejected_token = "REJECTED-MCP-HTTP-TOKEN-DO-NOT-PUBLISH"
    environment = os.environ.copy()
    environment[CONTROL_TOKEN_ENV] = bearer_token
    environment[CONTROL_ORIGIN_ENV] = CONTROL_ALLOWED_ORIGIN
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "about_llm.agents.mcp_streamable_http",
            "serve",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=None if cwd is None else str(cwd),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = b""
    stderr = b""
    session_id = ""
    event_ids: list[str] = []
    try:
        await _wait_until_ready(
            host,
            port,
            process,
            timeout_seconds=server_start_timeout_seconds,
        )
        timeout = httpx.Timeout(request_timeout_seconds)
        limits = httpx.Limits(max_connections=5, max_keepalive_connections=2)
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            trust_env=False,
        ) as client:
            invalid_origin = await client.post(
                endpoint,
                headers={
                    **_base_headers(bearer_token),
                    "Origin": "https://attacker.example.invalid",
                },
                json={"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
            )
            missing_auth_headers = _base_headers(bearer_token)
            del missing_auth_headers["Authorization"]
            missing_auth = await client.post(
                endpoint,
                headers=missing_auth_headers,
                json={"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
            )
            wrong_auth = await client.post(
                endpoint,
                headers={
                    **_base_headers(bearer_token),
                    "Authorization": f"Bearer {rejected_token}",
                },
                json={"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
            )
            if (invalid_origin.status_code, missing_auth.status_code, wrong_auth.status_code) != (
                403,
                401,
                401,
            ):
                raise ValueError("MCP HTTP connection security controls failed")

            initialize = await client.post(
                endpoint,
                headers=_base_headers(bearer_token),
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {
                            "name": "about-llm-mcp-http-client",
                            "version": "1.0.0",
                        },
                    },
                },
            )
            if initialize.status_code != 200:
                raise ValueError("MCP HTTP initialize failed")
            initialize_result = _expect_result(_decode_json_response(initialize), 1)
            if initialize_result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
                raise ValueError("MCP HTTP version negotiation failed")
            session_id = initialize.headers.get("mcp-session-id", "")
            if not _visible_ascii(session_id) or len(session_id) < 32:
                raise ValueError("MCP HTTP server did not assign a secure-shaped session id")
            headers = _subsequent_headers(bearer_token, session_id)

            missing_session = await client.post(
                endpoint,
                headers={
                    **_base_headers(bearer_token),
                    "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
                },
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            missing_version_headers = dict(headers)
            del missing_version_headers["MCP-Protocol-Version"]
            missing_version = await client.post(
                endpoint,
                headers=missing_version_headers,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            wrong_version = await client.post(
                endpoint,
                headers={**headers, "MCP-Protocol-Version": "2099-01-01"},
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            if (
                missing_session.status_code,
                missing_version.status_code,
                wrong_version.status_code,
            ) != (400, 400, 400):
                raise ValueError("MCP HTTP session/version negative controls failed")

            initialized = await client.post(
                endpoint,
                headers=headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            if initialized.status_code != 202 or initialized.content:
                raise ValueError("MCP initialized notification did not return empty 202")

            listed = await client.post(
                endpoint,
                headers=headers,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            list_result = _expect_result(_decode_json_response(listed), 2)
            tools = list_result.get("tools")
            if not isinstance(tools, list) or [tool.get("name") for tool in tools] != [
                ADD_INPUT_CONTRACT.name,
                WAIT_TOOL["name"],
            ]:
                raise ValueError("MCP HTTP tool discovery drifted")

            async with client.stream(
                "POST",
                endpoint,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": ADD_INPUT_CONTRACT.name,
                        "arguments": {"a": 7, "b": 5},
                    },
                },
            ) as tool_response:
                tool_events = await _read_sse_response(tool_response)
            if len(tool_events) != 2:
                raise ValueError("MCP POST SSE response did not contain two events")
            _validate_prime(tool_events[0])
            event_ids.extend(event.last_event_id for event in tool_events)
            tool_result = _expect_result(parse_sse_json_object(tool_events[1].data), 3)
            if tool_result.get("structuredContent") != {"sum": 12}:
                raise ValueError("MCP HTTP local tool verifier rejected the result")

            get_headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Origin": CONTROL_ALLOWED_ORIGIN,
                "Accept": "text/event-stream",
                "MCP-Session-Id": session_id,
                "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            }
            async with client.stream("GET", endpoint, headers=get_headers) as get_response:
                get_events = await _read_sse_response(get_response)
            if len(get_events) != 1:
                raise ValueError("MCP GET SSE control expected one priming event")
            _validate_prime(get_events[0])
            event_ids.append(get_events[0].last_event_id)

            prime_seen = asyncio.Event()

            async def run_wait() -> tuple[int, tuple[SSEEvent, ...]]:
                async with client.stream(
                    "POST",
                    endpoint,
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {"name": WAIT_TOOL["name"], "arguments": {}},
                    },
                ) as wait_response:
                    decoder = SSEDecoder(
                        max_line_bytes=MAX_SSE_BYTES,
                        max_event_bytes=MAX_SSE_BYTES,
                        max_total_bytes=MAX_SSE_BYTES,
                    )
                    events: list[SSEEvent] = []
                    async for chunk in wait_response.aiter_bytes():
                        framed = decoder.feed(chunk)
                        events.extend(framed)
                        if framed:
                            _validate_prime(framed[0])
                            prime_seen.set()
                    events.extend(decoder.finish())
                    return wait_response.status_code, tuple(events)

            wait_task = asyncio.create_task(run_wait())
            await asyncio.wait_for(prime_seen.wait(), timeout=request_timeout_seconds)
            cancellation = await client.post(
                endpoint,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": 4, "reason": "local control"},
                },
            )
            wait_status, wait_events = await asyncio.wait_for(
                wait_task, timeout=request_timeout_seconds
            )
            if cancellation.status_code != 202 or cancellation.content:
                raise ValueError("MCP cancellation notification was not accepted")
            if wait_status != 200 or len(wait_events) != 1:
                raise ValueError("cancelled MCP request emitted an unexpected response")
            event_ids.append(wait_events[0].last_event_id)

            terminated = await client.delete(endpoint, headers=headers)
            after_delete = await client.post(
                endpoint,
                headers=headers,
                json={"jsonrpc": "2.0", "id": 5, "method": "tools/list", "params": {}},
            )
            if terminated.status_code != 204 or terminated.content:
                raise ValueError("MCP session DELETE failed")
            if after_delete.status_code != 404:
                raise ValueError("terminated MCP session was not rejected with 404")

        if len(event_ids) != len(set(event_ids)):
            raise ValueError("MCP SSE event ids were not unique within the session")
        if process.poll() is not None:
            raise RuntimeError("MCP HTTP server exited during control")
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5.0)

    projection: dict[str, Any] = {
        "implementation": MCP_STREAMABLE_HTTP_CONTROL_VERSION,
        "protocol_version": MCP_PROTOCOL_VERSION,
        "binding": "Streamable HTTP",
        "network": {
            "scheme": "http",
            "address_scope": "IPv4 loopback",
            "real_tcp_http": True,
            "tls": False,
        },
        "transport": {
            "single_endpoint_path": MCP_ENDPOINT_PATH,
            "post_json_response_executed": True,
            "post_sse_response_executed": True,
            "get_sse_executed": True,
            "delete_session_executed": True,
            "notification_empty_202_verified": True,
            "sse_priming_event_with_id_verified": True,
            "event_ids_unique_within_session": True,
        },
        "security_controls": {
            "origin_allowlist_executed": True,
            "invalid_origin_status": 403,
            "bearer_header_gate_executed": True,
            "missing_or_wrong_bearer_status": 401,
            "oauth_flow_executed": False,
        },
        "session": {
            "server_assigned_on_initialize": True,
            "visible_ascii_and_minimum_length_verified": True,
            "included_on_subsequent_requests": True,
            "missing_session_status": 400,
            "missing_or_unsupported_protocol_version_status": 400,
            "terminated_session_status": 404,
        },
        "lifecycle_methods": [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
            "notifications/cancelled",
        ],
        "tool_result": {
            "structured_output_local_verifier_passed": True,
            "raw_arguments_or_result_published": False,
        },
        "cancellation": {
            "concurrent_request_and_notification_executed": True,
            "notification_status": 202,
            "jsonrpc_response_after_cancellation_count": 0,
            "stream_closed_after_cancellation": True,
        },
    }
    projection_fingerprint = "sha256:" + artifact_fingerprint(projection)
    return {
        **projection,
        "projection_fingerprint": projection_fingerprint,
        "public_projection_fields": sorted(projection),
        "raw_http_messages_published": False,
        "secret_or_session_identifiers_published": False,
        "server_process": {
            "subprocess_used": True,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_stderr_empty": not stdout and not stderr,
        },
        "evidence_limits": {
            "official_mcp_sdk_used": False,
            "full_mcp_schema_or_conformance_suite_executed": False,
            "oauth_or_protected_resource_metadata_proven": False,
            "business_authorization_or_human_approval_proven": False,
            "tls_proven": False,
            "sse_resumption_or_redelivery_proven": False,
            "server_to_client_jsonrpc_request_proven": False,
            "multi_stream_non_broadcast_proven": False,
            "remote_or_cross_vendor_interoperability_proven": False,
            "production_safety_proven": False,
            "projection_fingerprint_proves_authenticity": False,
        },
    }


def run_streamable_http_control(
    *,
    cwd: Path | None = None,
    request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    server_start_timeout_seconds: float = SERVER_START_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the real loopback transport control and return a content-free report."""

    if request_timeout_seconds <= 0 or server_start_timeout_seconds <= 0:
        raise ValueError("timeouts must be positive")
    return asyncio.run(
        _run_control_async(
            cwd=cwd,
            request_timeout_seconds=request_timeout_seconds,
            server_start_timeout_seconds=server_start_timeout_seconds,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="run the local fixture server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, required=True)
    subparsers.add_parser("control", help="run the real loopback control")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        token = os.environ.get(CONTROL_TOKEN_ENV, "")
        origin = os.environ.get(CONTROL_ORIGIN_ENV, "")
        if not token or not origin:
            raise ValueError("fixture control environment is missing")
        return serve(args.host, args.port, bearer_token=token, allowed_origin=origin)
    report = run_streamable_http_control()
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
