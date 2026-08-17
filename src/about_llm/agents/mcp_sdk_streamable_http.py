"""Official MCP SDK client/server control over real Streamable HTTP.

This control keeps the reviewed SDK client, low-level server, generated types,
and Streamable HTTP session manager in one execution.  The server runs in a
distinct Python subprocess on IPv4 loopback.  A private, random-token control
endpoint is used only for readiness and graceful Uvicorn shutdown; it is not
part of MCP and is not evidence of MCP authentication or authorization.
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
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Final, cast

import anyio
import httpx
import mcp.types as mcp_types
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from about_llm.agents.mcp_sdk_memory import (
    MCP_SDK_PROTOCOL_VERSION,
    MCP_SDK_REVIEWED_VERSION,
    build_mcp_sdk_fixture_server,
)
from about_llm.agents.mcp_stdio import ADD_INPUT_CONTRACT, ADD_OUTPUT_CONTRACT
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

MCP_SDK_HTTP_CONTROL_VERSION: Final = (
    "about-llm.mcp-sdk-streamable-http-control.v1"
)
MCP_SDK_HTTP_RECEIPT_VERSION: Final = (
    "about-llm.mcp-sdk-streamable-http-receipt.v1"
)
MCP_SDK_HTTP_CHECKED_AT: Final = "2026-08-14"
MCP_SDK_HTTP_ENDPOINT: Final = "/mcp"
MCP_SDK_HTTP_READY_PATH: Final = "/__control__/ready"
MCP_SDK_HTTP_SHUTDOWN_PATH: Final = "/__control__/shutdown"
MCP_SDK_HTTP_TOKEN_ENV: Final = "ABOUT_LLM_MCP_SDK_HTTP_CONTROL_TOKEN"
MCP_SDK_HTTP_MAX_RECEIPT_BYTES: Final = 4_096
MCP_SDK_HTTP_REQUEST_TIMEOUT_SECONDS: Final = 5.0
MCP_SDK_HTTP_SERVER_START_TIMEOUT_SECONDS: Final = 10.0
MCP_SDK_HTTP_SERVER_EXIT_TIMEOUT_SECONDS: Final = 10.0
MCP_SDK_HTTP_EVIDENCE_BOUNDARY: Final = (
    "This control uses mcp 1.29.0 ClientSession, low-level Server, generated "
    "types, streamable_http_client, StreamableHTTPSessionManager, and the SDK "
    "ASGI adapter for MCP 2025-11-25. A distinct Python server subprocess and "
    "real IPv4-loopback TCP/HTTP execute a stateful session, POST SSE responses, "
    "an opened GET SSE stream, and client-close DELETE termination. A separate "
    "random-token readiness/shutdown endpoint is control plumbing, not MCP "
    "authentication or authorization. The run executes discovery, one successful "
    "structured call, SDK schema rejection, and an application unknown-tool gate. "
    "It does not publish raw HTTP/protocol payloads, headers, the session id, token, "
    "PID, or SDK error content. It does not independently inject malformed JSON, "
    "duplicate keys, invalid UTF-8, oversized bodies, Host/Origin failures, network "
    "failure, reconnect/resumption, cancellation, or deadline cases. It does not "
    "execute TLS, OAuth, remote or cross-vendor interoperability, the official "
    "conformance suite, business authorization/approval, side effects, multi-worker "
    "operation, or production supervision. Loopback port reservation has a bind "
    "race. The minimized local receipt and unkeyed fingerprints do not authenticate "
    "the process, execution, or source."
)

_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "control_version",
        "checked_at",
        "runtime",
        "transport",
        "initialization",
        "discovery",
        "calls",
        "http_observations",
        "server_receipt",
        "scope",
        "evidence_boundary",
        "report_fingerprint",
    }
)
_RUNTIME_FIELDS: Final = frozenset(
    {"sdk_distribution", "sdk_version", "latest_protocol", "supported_protocols"}
)
_TRANSPORT_FIELDS: Final = frozenset(
    {
        "kind",
        "client_transport",
        "server_transport",
        "control_launched_server_subprocess",
        "real_ipv4_loopback_tcp_http",
        "stateful_session",
        "post_response_mode",
        "server_process_distinct",
        "client_session_id_observed",
        "mcp_session_termination_delete_observed",
        "server_shutdown_via_separate_control_endpoint",
        "private_control_unauthorized_status",
        "server_process_graceful_shutdown_observed",
        "server_stderr_empty",
        "raw_http_or_protocol_payload_published",
    }
)
_INITIALIZATION_FIELDS: Final = frozenset(
    {"protocol_version", "server_name", "server_version", "tools_capability"}
)
_DISCOVERY_FIELDS: Final = frozenset(
    {
        "tool_count",
        "tool_name",
        "input_schema_fingerprint",
        "output_schema_fingerprint",
        "closed_input_schema",
        "closed_output_schema",
    }
)
_CALL_FIELDS: Final = frozenset(
    {
        "successful_sum",
        "success_is_error",
        "invalid_schema_is_error",
        "invalid_schema_handler_delta",
        "unknown_tool_is_error",
        "unknown_tool_handler_delta",
        "raw_error_content_published",
    }
)
_HTTP_OBSERVATION_FIELDS: Final = frozenset(
    {
        "mcp_response_count",
        "post_count",
        "get_count",
        "delete_count",
        "status_200_count",
        "status_202_count",
        "sse_response_count",
        "json_response_count",
        "unexpected_method_status_or_media_type_count",
        "raw_headers_bodies_or_session_id_published",
    }
)
_SERVER_RECEIPT_FIELDS: Final = frozenset(
    {
        "handler_events",
        "recognized_handler_calls",
        "total_handler_calls",
        "session_manager_run_completed",
        "shutdown_control_received",
        "raw_arguments_or_results_published",
        "receipt_fingerprint",
    }
)
_SCOPE_FIELDS: Final = frozenset(
    {
        "official_sdk_client_executed",
        "official_sdk_server_executed",
        "official_sdk_streamable_http_client_executed",
        "official_sdk_streamable_http_session_manager_executed",
        "real_loopback_tcp_http_executed",
        "stateful_session_and_delete_executed",
        "post_sse_responses_executed",
        "get_sse_stream_opened",
        "mcp_2025_11_25_negotiated",
        "official_generated_types_executed",
        "sdk_json_schema_validation_executed",
        "application_unknown_tool_gate_executed",
        "private_control_token_gate_executed",
        "malformed_http_controls_executed",
        "session_resumption_executed",
        "tls_or_oauth_executed",
        "remote_or_cross_vendor_interop_proven",
        "official_conformance_suite_executed",
        "authentication_or_authorization_proven",
        "production_readiness_proven",
    }
)
_RECEIPT_FIELDS: Final = frozenset(
    {
        "receipt_version",
        "sdk_version",
        "protocol_version",
        "server_pid",
        "handler_events",
        "recognized_handler_calls",
        "total_handler_calls",
        "session_manager_run_completed",
        "shutdown_control_received",
        "raw_arguments_or_results_published",
    }
)


def _snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(canonical_json_bytes(value)))


def _exact(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{label} fields mismatch")


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_server_receipt(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw or len(raw) > MCP_SDK_HTTP_MAX_RECEIPT_BYTES:
        raise ValueError("official MCP SDK HTTP receipt size is invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("official MCP SDK HTTP receipt is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError("official MCP SDK HTTP receipt must be an object")
    if canonical_json_bytes(value) != raw:
        raise ValueError("official MCP SDK HTTP receipt is not canonical JSON")
    _exact(value, _RECEIPT_FIELDS, "server receipt")
    return value


def _write_server_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(receipt)
    if len(raw) > MCP_SDK_HTTP_MAX_RECEIPT_BYTES:
        raise ValueError("official MCP SDK HTTP receipt exceeds its byte cap")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _assert_reviewed_sdk() -> str:
    sdk_version = importlib.metadata.version("mcp")
    if sdk_version != MCP_SDK_REVIEWED_VERSION:
        raise ValueError(
            "mcp SDK version drift: "
            f"expected {MCP_SDK_REVIEWED_VERSION}, got {sdk_version}"
        )
    if (
        mcp_types.LATEST_PROTOCOL_VERSION != MCP_SDK_PROTOCOL_VERSION
        or MCP_SDK_PROTOCOL_VERSION not in SUPPORTED_PROTOCOL_VERSIONS
    ):
        raise ValueError("mcp SDK protocol-version drift")
    return sdk_version


def _valid_control_token(value: str) -> bool:
    return 32 <= len(value) <= 256 and all(0x21 <= ord(item) <= 0x7E for item in value)


def _authorized(request: Request, token: str) -> bool:
    supplied = request.headers.get("authorization", "")
    expected = f"Bearer {token}"
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _control_json(value: Mapping[str, Any], *, status_code: int = 200) -> Response:
    return Response(
        canonical_json_bytes(value),
        status_code=status_code,
        media_type="application/json",
    )


def _build_server_app(
    host: str,
    port: int,
    receipt_path: Path,
    token: str,
    request_shutdown: Callable[[], None],
) -> Starlette:
    handler_events: list[str] = []
    server, counters = build_mcp_sdk_fixture_server(
        handler_events=handler_events,
        server_name="about-llm-mcp-sdk-streamable-http",
    )
    manager = StreamableHTTPSessionManager(
        app=server,
        json_response=False,
        stateless=False,
        security_settings=TransportSecuritySettings(
            allowed_hosts=[f"{host}:{port}"],
        ),
    )
    lifecycle = {
        "manager_completed": False,
        "shutdown_received": False,
    }

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            async with manager.run():
                yield
            lifecycle["manager_completed"] = True
        finally:
            _write_server_receipt(
                receipt_path,
                {
                    "receipt_version": MCP_SDK_HTTP_RECEIPT_VERSION,
                    "sdk_version": MCP_SDK_REVIEWED_VERSION,
                    "protocol_version": MCP_SDK_PROTOCOL_VERSION,
                    "server_pid": os.getpid(),
                    "handler_events": list(handler_events),
                    "recognized_handler_calls": counters["recognized"],
                    "total_handler_calls": counters["total"],
                    "session_manager_run_completed": lifecycle[
                        "manager_completed"
                    ],
                    "shutdown_control_received": lifecycle["shutdown_received"],
                    "raw_arguments_or_results_published": False,
                },
            )

    async def ready(request: Request) -> Response:
        if not _authorized(request, token):
            return _control_json({"error": "unauthorized"}, status_code=401)
        return _control_json({"ready": True})

    async def shutdown(request: Request) -> Response:
        if not _authorized(request, token):
            return _control_json({"error": "unauthorized"}, status_code=401)
        lifecycle["shutdown_received"] = True
        request_shutdown()
        return Response(status_code=204)

    return Starlette(
        routes=[
            Route(MCP_SDK_HTTP_ENDPOINT, endpoint=StreamableHTTPASGIApp(manager)),
            Route(MCP_SDK_HTTP_READY_PATH, endpoint=ready, methods=["GET"]),
            Route(
                MCP_SDK_HTTP_SHUTDOWN_PATH,
                endpoint=shutdown,
                methods=["POST"],
            ),
        ],
        lifespan=lifespan,
    )


def serve_mcp_sdk_http(host: str, port: int, receipt_path: Path) -> int:
    """Serve the official SDK fixture until the private shutdown control fires."""

    _assert_reviewed_sdk()
    if host != "127.0.0.1":
        raise ValueError("the official MCP SDK HTTP control only permits IPv4 loopback")
    if not 0 < port < 65_536:
        raise ValueError("port must be between 1 and 65535")
    if not receipt_path.is_absolute() or not receipt_path.parent.is_dir():
        raise ValueError("server receipt path must be absolute with an existing parent")
    if receipt_path.exists():
        raise FileExistsError("server receipt path already exists")
    token = os.environ.get(MCP_SDK_HTTP_TOKEN_ENV, "")
    if not _valid_control_token(token):
        raise ValueError("private server control token is missing or invalid")

    holder: dict[str, uvicorn.Server] = {}

    def request_shutdown() -> None:
        holder["server"].should_exit = True

    app = _build_server_app(host, port, receipt_path, token, request_shutdown)
    logging.disable(logging.CRITICAL)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=host,
            port=port,
            access_log=False,
            log_level="critical",
            ws="none",
        )
    )
    holder["server"] = server
    asyncio.run(server.serve())
    return 0


def _reserve_candidate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind(("127.0.0.1", 0))
        return cast(tuple[str, int], candidate.getsockname())[1]


async def _wait_until_ready(
    client: httpx.AsyncClient,
    base_url: str,
    process: subprocess.Popen[bytes],
    token: str,
) -> None:
    deadline = time.monotonic() + MCP_SDK_HTTP_SERVER_START_TIMEOUT_SECONDS
    headers = {"Authorization": f"Bearer {token}"}
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("official MCP SDK HTTP server exited before readiness")
        try:
            response = await client.get(
                f"{base_url}{MCP_SDK_HTTP_READY_PATH}", headers=headers
            )
            if response.status_code == 200 and response.content == canonical_json_bytes(
                {"ready": True}
            ):
                return
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.05)
    raise TimeoutError("official MCP SDK HTTP server readiness timed out")


def _stop_failed_process(process: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    if process.poll() is None:
        process.terminate()
    try:
        return process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.communicate(timeout=5.0)


def _summarize_http_observations(
    observations: Sequence[tuple[str, int, str]],
) -> dict[str, Any]:
    normalized = [
        (method, status, content_type.partition(";")[0].strip().lower())
        for method, status, content_type in observations
    ]
    expected = {
        ("POST", 200, "text/event-stream"),
        ("POST", 202, "application/json"),
        ("GET", 200, "text/event-stream"),
        ("DELETE", 200, "application/json"),
    }
    return {
        "mcp_response_count": len(normalized),
        "post_count": sum(method == "POST" for method, _, _ in normalized),
        "get_count": sum(method == "GET" for method, _, _ in normalized),
        "delete_count": sum(method == "DELETE" for method, _, _ in normalized),
        "status_200_count": sum(status == 200 for _, status, _ in normalized),
        "status_202_count": sum(status == 202 for _, status, _ in normalized),
        "sse_response_count": sum(
            content_type == "text/event-stream" for _, _, content_type in normalized
        ),
        "json_response_count": sum(
            content_type == "application/json" for _, _, content_type in normalized
        ),
        "unexpected_method_status_or_media_type_count": sum(
            observation not in expected for observation in normalized
        ),
        "raw_headers_bodies_or_session_id_published": False,
    }


async def _run_mcp_sdk_http_control() -> dict[str, Any]:
    sdk_version = _assert_reviewed_sdk()
    host = "127.0.0.1"
    port = _reserve_candidate_port()
    base_url = f"http://{host}:{port}"
    token = secrets.token_urlsafe(32)
    observations: list[tuple[str, int, str]] = []

    async def observe_response(response: httpx.Response) -> None:
        if response.request.url.path == MCP_SDK_HTTP_ENDPOINT:
            observations.append(
                (
                    response.request.method,
                    response.status_code,
                    response.headers.get("content-type", ""),
                )
            )

    sdk_loggers = ("mcp", "FastMCP")
    prior_levels = {name: logging.getLogger(name).level for name in sdk_loggers}
    for name in sdk_loggers:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    process: subprocess.Popen[bytes] | None = None
    stdout = b""
    stderr = b""
    try:
        with tempfile.TemporaryDirectory(
            prefix="about-llm-mcp-sdk-http-"
        ) as temp_directory:
            receipt_path = (Path(temp_directory) / "server-receipt.json").resolve()
            environment = os.environ.copy()
            environment[MCP_SDK_HTTP_TOKEN_ENV] = token
            environment["PYTHONUTF8"] = "1"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "about_llm.agents.mcp_sdk_streamable_http",
                    "serve",
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--receipt",
                    str(receipt_path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            timeout = httpx.Timeout(
                MCP_SDK_HTTP_REQUEST_TIMEOUT_SECONDS,
                read=MCP_SDK_HTTP_REQUEST_TIMEOUT_SECONDS * 2,
            )
            async with httpx.AsyncClient(
                timeout=timeout,
                trust_env=False,
                event_hooks={"response": [observe_response]},
            ) as http_client:
                await _wait_until_ready(http_client, base_url, process, token)
                unauthorized_control = await http_client.get(
                    f"{base_url}{MCP_SDK_HTTP_READY_PATH}"
                )
                if (
                    unauthorized_control.status_code != 401
                    or unauthorized_control.content
                    != canonical_json_bytes({"error": "unauthorized"})
                ):
                    raise ValueError("private HTTP control token gate failed")
                async with (
                    streamable_http_client(
                        f"{base_url}{MCP_SDK_HTTP_ENDPOINT}",
                        http_client=http_client,
                        terminate_on_close=True,
                    ) as (read_stream, write_stream, get_session_id),
                    ClientSession(
                        read_stream=read_stream,
                        write_stream=write_stream,
                        read_timeout_seconds=timedelta(seconds=5),
                        client_info=mcp_types.Implementation(
                            name="about-llm-sdk-http-client", version="1.0.0"
                        ),
                    ) as session,
                ):
                    initialized = await session.initialize()
                    session_id = get_session_id()
                    if (
                        not isinstance(session_id, str)
                        or not session_id
                        or len(session_id) > 256
                        or not session_id.isascii()
                    ):
                        raise ValueError("official SDK HTTP session id was not observed")
                    await session.send_ping()
                    listed = await session.list_tools()
                    if len(listed.tools) != 1:
                        raise ValueError("official SDK HTTP tool discovery count drift")
                    tool = listed.tools[0]
                    if tool.name != ADD_INPUT_CONTRACT.name:
                        raise ValueError("official SDK HTTP tool identity drift")
                    success = await session.call_tool(
                        ADD_INPUT_CONTRACT.name, {"a": 2, "b": 3}
                    )
                    invalid = await session.call_tool(
                        ADD_INPUT_CONTRACT.name,
                        {"a": 2, "b": 3, "extra": 1},
                    )
                    unknown = await session.call_tool("fixture.missing", {})

                shutdown = await http_client.post(
                    f"{base_url}{MCP_SDK_HTTP_SHUTDOWN_PATH}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if shutdown.status_code != 204:
                    raise ValueError("private HTTP shutdown control failed")

            try:
                stdout, stderr = process.communicate(
                    timeout=MCP_SDK_HTTP_SERVER_EXIT_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired as error:
                _stop_failed_process(process)
                raise TimeoutError(
                    "official MCP SDK HTTP server graceful shutdown timed out"
                ) from error
            if process.returncode != 0:
                raise RuntimeError("official MCP SDK HTTP server exited unsuccessfully")
            if stdout or stderr:
                raise ValueError("official MCP SDK HTTP server wrote to stdout/stderr")
            receipt = _load_server_receipt(receipt_path)
    finally:
        if process is not None and process.poll() is None:
            _stop_failed_process(process)
        for name, level in prior_levels.items():
            logging.getLogger(name).setLevel(level)

    if success.isError or success.structuredContent != {"sum": 5}:
        raise ValueError("official SDK HTTP successful result drift")
    if invalid.isError is not True or unknown.isError is not True:
        raise ValueError("official SDK HTTP negative control did not fail closed")
    if receipt != {
        "receipt_version": MCP_SDK_HTTP_RECEIPT_VERSION,
        "sdk_version": MCP_SDK_REVIEWED_VERSION,
        "protocol_version": MCP_SDK_PROTOCOL_VERSION,
        "server_pid": receipt.get("server_pid"),
        "handler_events": [ADD_INPUT_CONTRACT.name, "fixture.missing"],
        "recognized_handler_calls": 1,
        "total_handler_calls": 2,
        "session_manager_run_completed": True,
        "shutdown_control_received": True,
        "raw_arguments_or_results_published": False,
    }:
        raise ValueError("official SDK HTTP server receipt semantic drift")
    server_pid = receipt.get("server_pid")
    if isinstance(server_pid, bool) or not isinstance(server_pid, int):
        raise ValueError("official SDK HTTP server pid is invalid")
    if server_pid <= 0 or server_pid == os.getpid():
        raise ValueError("official SDK HTTP server was not a distinct process")
    if initialized.protocolVersion != MCP_SDK_PROTOCOL_VERSION:
        raise ValueError("official SDK HTTP negotiated protocol drift")
    if initialized.capabilities.tools is None:
        raise ValueError("official SDK HTTP did not negotiate tools capability")

    http_observations = _summarize_http_observations(observations)
    expected_http_observations = {
        "mcp_response_count": 9,
        "post_count": 7,
        "get_count": 1,
        "delete_count": 1,
        "status_200_count": 8,
        "status_202_count": 1,
        "sse_response_count": 7,
        "json_response_count": 2,
        "unexpected_method_status_or_media_type_count": 0,
        "raw_headers_bodies_or_session_id_published": False,
    }
    if http_observations != expected_http_observations:
        raise ValueError("official SDK HTTP method/status/media-type profile drift")

    input_schema = tool.inputSchema
    output_schema = cast(dict[str, Any], tool.outputSchema)
    receipt_projection = {
        key: value for key, value in receipt.items() if key != "server_pid"
    }
    report: dict[str, Any] = {
        "control_version": MCP_SDK_HTTP_CONTROL_VERSION,
        "checked_at": MCP_SDK_HTTP_CHECKED_AT,
        "runtime": {
            "sdk_distribution": "mcp",
            "sdk_version": sdk_version,
            "latest_protocol": mcp_types.LATEST_PROTOCOL_VERSION,
            "supported_protocols": list(SUPPORTED_PROTOCOL_VERSIONS),
        },
        "transport": {
            "kind": "official_sdk_streamable_http_subprocess",
            "client_transport": (
                "mcp.client.streamable_http.streamable_http_client"
            ),
            "server_transport": (
                "mcp.server.streamable_http_manager.StreamableHTTPSessionManager"
            ),
            "control_launched_server_subprocess": True,
            "real_ipv4_loopback_tcp_http": True,
            "stateful_session": True,
            "post_response_mode": "sse",
            "server_process_distinct": True,
            "client_session_id_observed": True,
            "mcp_session_termination_delete_observed": True,
            "server_shutdown_via_separate_control_endpoint": True,
            "private_control_unauthorized_status": 401,
            "server_process_graceful_shutdown_observed": True,
            "server_stderr_empty": True,
            "raw_http_or_protocol_payload_published": False,
        },
        "initialization": {
            "protocol_version": initialized.protocolVersion,
            "server_name": initialized.serverInfo.name,
            "server_version": initialized.serverInfo.version,
            "tools_capability": True,
        },
        "discovery": {
            "tool_count": len(listed.tools),
            "tool_name": tool.name,
            "input_schema_fingerprint": artifact_fingerprint(input_schema),
            "output_schema_fingerprint": artifact_fingerprint(output_schema),
            "closed_input_schema": input_schema.get("additionalProperties") is False,
            "closed_output_schema": output_schema.get("additionalProperties") is False,
        },
        "calls": {
            "successful_sum": cast(Mapping[str, Any], success.structuredContent)[
                "sum"
            ],
            "success_is_error": bool(success.isError),
            "invalid_schema_is_error": bool(invalid.isError),
            "invalid_schema_handler_delta": 0,
            "unknown_tool_is_error": bool(unknown.isError),
            "unknown_tool_handler_delta": 1,
            "raw_error_content_published": False,
        },
        "http_observations": http_observations,
        "server_receipt": {
            "handler_events": receipt["handler_events"],
            "recognized_handler_calls": receipt["recognized_handler_calls"],
            "total_handler_calls": receipt["total_handler_calls"],
            "session_manager_run_completed": receipt[
                "session_manager_run_completed"
            ],
            "shutdown_control_received": receipt["shutdown_control_received"],
            "raw_arguments_or_results_published": receipt[
                "raw_arguments_or_results_published"
            ],
            "receipt_fingerprint": artifact_fingerprint(receipt_projection),
        },
        "scope": {
            "official_sdk_client_executed": True,
            "official_sdk_server_executed": True,
            "official_sdk_streamable_http_client_executed": True,
            "official_sdk_streamable_http_session_manager_executed": True,
            "real_loopback_tcp_http_executed": True,
            "stateful_session_and_delete_executed": True,
            "post_sse_responses_executed": True,
            "get_sse_stream_opened": True,
            "mcp_2025_11_25_negotiated": True,
            "official_generated_types_executed": True,
            "sdk_json_schema_validation_executed": True,
            "application_unknown_tool_gate_executed": True,
            "private_control_token_gate_executed": True,
            "malformed_http_controls_executed": False,
            "session_resumption_executed": False,
            "tls_or_oauth_executed": False,
            "remote_or_cross_vendor_interop_proven": False,
            "official_conformance_suite_executed": False,
            "authentication_or_authorization_proven": False,
            "production_readiness_proven": False,
        },
        "evidence_boundary": MCP_SDK_HTTP_EVIDENCE_BOUNDARY,
    }
    report["report_fingerprint"] = artifact_fingerprint(report)
    return report


def run_mcp_sdk_http_control() -> dict[str, Any]:
    """Run the official-SDK Streamable HTTP control and verify its report."""

    return verify_mcp_sdk_http_report(anyio.run(_run_mcp_sdk_http_control))


def verify_mcp_sdk_http_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the closed report and its deliberately narrow evidence scope."""

    _exact(report, _TOP_LEVEL_FIELDS, "report")
    nested_fields = (
        ("runtime", _RUNTIME_FIELDS),
        ("transport", _TRANSPORT_FIELDS),
        ("initialization", _INITIALIZATION_FIELDS),
        ("discovery", _DISCOVERY_FIELDS),
        ("calls", _CALL_FIELDS),
        ("http_observations", _HTTP_OBSERVATION_FIELDS),
        ("server_receipt", _SERVER_RECEIPT_FIELDS),
        ("scope", _SCOPE_FIELDS),
    )
    nested: dict[str, Mapping[str, Any]] = {}
    for name, fields in nested_fields:
        value = report.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"report.{name} must be an object")
        _exact(value, fields, f"report.{name}")
        nested[name] = value

    fingerprint = report.get("report_fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("report fingerprint is invalid")
    unsigned = copy.deepcopy(dict(report))
    del unsigned["report_fingerprint"]
    if fingerprint != artifact_fingerprint(unsigned):
        raise ValueError("report fingerprint mismatch")

    receipt_projection = {
        "receipt_version": MCP_SDK_HTTP_RECEIPT_VERSION,
        "sdk_version": MCP_SDK_REVIEWED_VERSION,
        "protocol_version": MCP_SDK_PROTOCOL_VERSION,
        "handler_events": [ADD_INPUT_CONTRACT.name, "fixture.missing"],
        "recognized_handler_calls": 1,
        "total_handler_calls": 2,
        "session_manager_run_completed": True,
        "shutdown_control_received": True,
        "raw_arguments_or_results_published": False,
    }
    if (
        report.get("control_version") != MCP_SDK_HTTP_CONTROL_VERSION
        or report.get("checked_at") != MCP_SDK_HTTP_CHECKED_AT
        or report.get("evidence_boundary") != MCP_SDK_HTTP_EVIDENCE_BOUNDARY
        or nested["runtime"]
        != {
            "sdk_distribution": "mcp",
            "sdk_version": MCP_SDK_REVIEWED_VERSION,
            "latest_protocol": MCP_SDK_PROTOCOL_VERSION,
            "supported_protocols": list(SUPPORTED_PROTOCOL_VERSIONS),
        }
        or nested["transport"]
        != {
            "kind": "official_sdk_streamable_http_subprocess",
            "client_transport": (
                "mcp.client.streamable_http.streamable_http_client"
            ),
            "server_transport": (
                "mcp.server.streamable_http_manager.StreamableHTTPSessionManager"
            ),
            "control_launched_server_subprocess": True,
            "real_ipv4_loopback_tcp_http": True,
            "stateful_session": True,
            "post_response_mode": "sse",
            "server_process_distinct": True,
            "client_session_id_observed": True,
            "mcp_session_termination_delete_observed": True,
            "server_shutdown_via_separate_control_endpoint": True,
            "private_control_unauthorized_status": 401,
            "server_process_graceful_shutdown_observed": True,
            "server_stderr_empty": True,
            "raw_http_or_protocol_payload_published": False,
        }
        or nested["initialization"]
        != {
            "protocol_version": MCP_SDK_PROTOCOL_VERSION,
            "server_name": "about-llm-mcp-sdk-streamable-http",
            "server_version": "1.0.0",
            "tools_capability": True,
        }
        or nested["discovery"].get("tool_count") != 1
        or nested["discovery"].get("tool_name") != ADD_INPUT_CONTRACT.name
        or nested["discovery"].get("input_schema_fingerprint")
        != artifact_fingerprint(_snapshot(ADD_INPUT_CONTRACT.arguments_schema))
        or nested["discovery"].get("output_schema_fingerprint")
        != artifact_fingerprint(_snapshot(ADD_OUTPUT_CONTRACT.arguments_schema))
        or nested["discovery"].get("closed_input_schema") is not True
        or nested["discovery"].get("closed_output_schema") is not True
        or nested["calls"]
        != {
            "successful_sum": 5,
            "success_is_error": False,
            "invalid_schema_is_error": True,
            "invalid_schema_handler_delta": 0,
            "unknown_tool_is_error": True,
            "unknown_tool_handler_delta": 1,
            "raw_error_content_published": False,
        }
        or nested["http_observations"]
        != {
            "mcp_response_count": 9,
            "post_count": 7,
            "get_count": 1,
            "delete_count": 1,
            "status_200_count": 8,
            "status_202_count": 1,
            "sse_response_count": 7,
            "json_response_count": 2,
            "unexpected_method_status_or_media_type_count": 0,
            "raw_headers_bodies_or_session_id_published": False,
        }
        or nested["server_receipt"]
        != {
            "handler_events": [ADD_INPUT_CONTRACT.name, "fixture.missing"],
            "recognized_handler_calls": 1,
            "total_handler_calls": 2,
            "session_manager_run_completed": True,
            "shutdown_control_received": True,
            "raw_arguments_or_results_published": False,
            "receipt_fingerprint": artifact_fingerprint(receipt_projection),
        }
        or nested["scope"]
        != {
            "official_sdk_client_executed": True,
            "official_sdk_server_executed": True,
            "official_sdk_streamable_http_client_executed": True,
            "official_sdk_streamable_http_session_manager_executed": True,
            "real_loopback_tcp_http_executed": True,
            "stateful_session_and_delete_executed": True,
            "post_sse_responses_executed": True,
            "get_sse_stream_opened": True,
            "mcp_2025_11_25_negotiated": True,
            "official_generated_types_executed": True,
            "sdk_json_schema_validation_executed": True,
            "application_unknown_tool_gate_executed": True,
            "private_control_token_gate_executed": True,
            "malformed_http_controls_executed": False,
            "session_resumption_executed": False,
            "tls_or_oauth_executed": False,
            "remote_or_cross_vendor_interop_proven": False,
            "official_conformance_suite_executed": False,
            "authentication_or_authorization_proven": False,
            "production_readiness_proven": False,
        }
    ):
        raise ValueError("official MCP SDK HTTP report semantic drift")
    return copy.deepcopy(dict(report))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or serve the official MCP SDK Streamable HTTP control."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("control", help="run the client/subprocess control")
    serve_parser = subparsers.add_parser("serve", help="serve one SDK HTTP control")
    serve_parser.add_argument("--host", required=True)
    serve_parser.add_argument("--port", required=True, type=int)
    serve_parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "serve":
        return serve_mcp_sdk_http(args.host, args.port, args.receipt)
    if args.command not in {None, "control"}:
        raise ValueError("unknown command")  # pragma: no cover - argparse owns this
    print(
        json.dumps(
            run_mcp_sdk_http_control(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
