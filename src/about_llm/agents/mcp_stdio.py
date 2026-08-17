"""Minimal MCP stdio lifecycle and tool control for local interoperability study.

This module implements a deliberately small projection of MCP 2025-11-25. It
exercises a real client-launched subprocess and newline-delimited UTF-8 JSON-RPC
messages, but it is not a general MCP SDK or a protocol conformance suite.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, NoReturn, cast

from about_llm.agents.schema import (
    DRAFT_2020_12_URI,
    JSONSchemaToolContract,
    ToolArgumentValidationError,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_STDIO_CONTROL_VERSION = "about-llm.mcp-stdio-control.v1"
MAX_STDIO_MESSAGE_BYTES = 64_000

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


class MCPFrameError(ValueError):
    """A value is not one complete strict MCP stdio JSON message."""


class MCPProtocolError(ValueError):
    """A stable JSON-RPC protocol error safe to return to the fixture client."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class SessionPhase(str, Enum):
    NEW = "new"
    AWAITING_INITIALIZED = "awaiting_initialized"
    READY = "ready"


ADD_INPUT_CONTRACT = JSONSchemaToolContract(
    "fixture.add",
    "Add two bounded integers in a local read-only fixture.",
    "mcp-fixture-add-input@v1",
    {
        "$schema": DRAFT_2020_12_URI,
        "type": "object",
        "properties": {
            "a": {"type": "integer", "minimum": -1_000_000, "maximum": 1_000_000},
            "b": {"type": "integer", "minimum": -1_000_000, "maximum": 1_000_000},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    },
)

ADD_OUTPUT_CONTRACT = JSONSchemaToolContract(
    "fixture.add.output",
    "Validate the structured result of the local addition fixture.",
    "mcp-fixture-add-output@v1",
    {
        "$schema": DRAFT_2020_12_URI,
        "type": "object",
        "properties": {
            "sum": {
                "type": "integer",
                "minimum": -2_000_000,
                "maximum": 2_000_000,
            }
        },
        "required": ["sum"],
        "additionalProperties": False,
    },
)


def _reject_nonfinite(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def decode_stdio_message(
    line: bytes,
    *,
    max_message_bytes: int = MAX_STDIO_MESSAGE_BYTES,
) -> dict[str, Any]:
    """Decode one LF-delimited, strict UTF-8 JSON object."""
    if isinstance(max_message_bytes, bool) or max_message_bytes <= 0:
        raise ValueError("max_message_bytes must be a positive integer")
    if not line.endswith(b"\n"):
        raise MCPFrameError("MCP stdio message is missing its LF delimiter")
    payload = line[:-1]
    if len(payload) > max_message_bytes:
        raise MCPFrameError("MCP stdio message exceeds the byte limit")
    if not payload:
        raise MCPFrameError("MCP stdio message cannot be empty")
    if b"\n" in payload or b"\r" in payload:
        raise MCPFrameError("MCP stdio message contains an embedded newline")
    try:
        text = payload.decode("utf-8", errors="strict")
        decoded = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise MCPFrameError("MCP stdio message is not strict UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise MCPFrameError("MCP stdio message must be a JSON object")
    return cast(dict[str, Any], decoded)


def encode_stdio_message(message: Mapping[str, Any]) -> bytes:
    """Encode one canonical JSON object followed by exactly one LF delimiter."""
    encoded = canonical_json_bytes(message)
    if len(encoded) > MAX_STDIO_MESSAGE_BYTES:
        raise MCPFrameError("MCP stdio message exceeds the byte limit")
    if b"\n" in encoded or b"\r" in encoded:
        raise MCPFrameError("canonical MCP message unexpectedly contains a newline")
    return encoded + b"\n"


def _snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(canonical_json_bytes(value)))


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _valid_request_id(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
        value, str
    )


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> None:
    keys = frozenset(value)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise MCPProtocolError(INVALID_PARAMS, f"Invalid {label}")


def _tool_error(error: ToolArgumentValidationError) -> dict[str, Any]:
    fields = [f"code={error.code}"]
    if error.keyword is not None:
        fields.append(f"keyword={error.keyword}")
    if error.instance_pointer:
        fields.append(f"instance_path={error.instance_pointer}")
    return {
        "content": [
            {
                "type": "text",
                "text": "Tool arguments rejected (" + ", ".join(fields) + ")",
            }
        ],
        "isError": True,
    }


class MCPStdioSession:
    """Stateful minimal server projection for initialize, tools/list and tools/call."""

    def __init__(self) -> None:
        self.phase = SessionPhase.NEW
        self.client_requested_version: str | None = None

    def process_message(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, Mapping):
            return _error_response(None, INVALID_REQUEST, "Invalid Request")

        request_id = message.get("id")
        id_present = "id" in message
        response_id = request_id if id_present and _valid_request_id(request_id) else None
        allowed_keys = frozenset({"jsonrpc", "id", "method", "params"})
        if (
            not frozenset(message).issubset(allowed_keys)
            or message.get("jsonrpc") != "2.0"
            or not isinstance(message.get("method"), str)
            or not cast(str, message.get("method"))
            or (id_present and not _valid_request_id(request_id))
        ):
            return _error_response(response_id, INVALID_REQUEST, "Invalid Request")

        params_value = message.get("params", {})
        if not isinstance(params_value, Mapping):
            if id_present:
                return _error_response(response_id, INVALID_PARAMS, "Invalid params")
            return None

        method = cast(str, message["method"])
        is_notification = not id_present
        try:
            result = self._dispatch(method, params_value, is_notification)
        except MCPProtocolError as error:
            if is_notification:
                return None
            return _error_response(response_id, error.code, error.message)

        if is_notification:
            return None
        if result is None:
            return _error_response(response_id, INVALID_REQUEST, "Invalid Request")
        return {"jsonrpc": "2.0", "id": response_id, "result": result}

    def _dispatch(
        self,
        method: str,
        params: Mapping[str, Any],
        is_notification: bool,
    ) -> Mapping[str, Any] | None:
        if method == "initialize":
            if is_notification:
                raise MCPProtocolError(INVALID_REQUEST, "initialize must be a request")
            return self._initialize(params)
        if method == "notifications/initialized":
            if not is_notification:
                raise MCPProtocolError(
                    INVALID_REQUEST, "notifications/initialized must be a notification"
                )
            self._initialized(params)
            return None
        if method == "ping":
            if is_notification:
                return None
            return {}
        if method == "tools/list":
            if is_notification:
                raise MCPProtocolError(INVALID_REQUEST, "tools/list must be a request")
            return self._list_tools(params)
        if method == "tools/call":
            if is_notification:
                raise MCPProtocolError(INVALID_REQUEST, "tools/call must be a request")
            return self._call_tool(params)
        raise MCPProtocolError(METHOD_NOT_FOUND, "Method not found")

    def _initialize(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.phase is not SessionPhase.NEW:
            raise MCPProtocolError(INVALID_REQUEST, "Session already initialized")
        _require_exact_keys(
            params,
            required=frozenset({"protocolVersion", "capabilities", "clientInfo"}),
            label="initialize params",
        )
        requested = params["protocolVersion"]
        capabilities = params["capabilities"]
        client_info = params["clientInfo"]
        if (
            not isinstance(requested, str)
            or not requested
            or not isinstance(capabilities, Mapping)
            or not isinstance(client_info, Mapping)
            or not isinstance(client_info.get("name"), str)
            or not isinstance(client_info.get("version"), str)
        ):
            raise MCPProtocolError(INVALID_PARAMS, "Invalid initialize params")

        self.client_requested_version = requested
        self.phase = SessionPhase.AWAITING_INITIALIZED
        negotiated = (
            requested if requested == MCP_PROTOCOL_VERSION else MCP_PROTOCOL_VERSION
        )
        return {
            "protocolVersion": negotiated,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": "about-llm-mcp-stdio-fixture",
                "title": "About LLM MCP stdio fixture",
                "version": "1.0.0",
                "description": "Local read-only protocol lifecycle control.",
            },
            "instructions": (
                "Fixture only: discovery is not authorization and results are untrusted."
            ),
        }

    def _initialized(self, params: Mapping[str, Any]) -> None:
        if params:
            raise MCPProtocolError(INVALID_PARAMS, "Invalid initialized params")
        if self.phase is not SessionPhase.AWAITING_INITIALIZED:
            raise MCPProtocolError(INVALID_REQUEST, "Unexpected initialized notification")
        self.phase = SessionPhase.READY

    def _require_ready(self) -> None:
        if self.phase is not SessionPhase.READY:
            raise MCPProtocolError(INVALID_REQUEST, "Session not initialized")

    def _list_tools(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        self._require_ready()
        _require_exact_keys(
            params,
            required=frozenset(),
            optional=frozenset({"cursor"}),
            label="tools/list params",
        )
        if params.get("cursor") is not None:
            raise MCPProtocolError(INVALID_PARAMS, "Unknown tools/list cursor")
        return {
            "tools": [
                {
                    "name": ADD_INPUT_CONTRACT.name,
                    "title": "Fixture integer addition",
                    "description": ADD_INPUT_CONTRACT.description,
                    "inputSchema": _snapshot(ADD_INPUT_CONTRACT.arguments_schema),
                    "outputSchema": _snapshot(ADD_OUTPUT_CONTRACT.arguments_schema),
                    "execution": {"taskSupport": "forbidden"},
                }
            ]
        }

    def _call_tool(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        self._require_ready()
        _require_exact_keys(
            params,
            required=frozenset({"name", "arguments"}),
            label="tools/call params",
        )
        name = params["name"]
        arguments = params["arguments"]
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            raise MCPProtocolError(INVALID_PARAMS, "Invalid tools/call params")
        if name != ADD_INPUT_CONTRACT.name:
            raise MCPProtocolError(INVALID_PARAMS, "Unknown tool")

        try:
            ADD_INPUT_CONTRACT.validate(arguments)
        except ToolArgumentValidationError as error:
            return _tool_error(error)

        result = {"sum": cast(int, arguments["a"]) + cast(int, arguments["b"])}
        ADD_OUTPUT_CONTRACT.validate(result)
        serialized = canonical_json_bytes(result).decode("utf-8")
        return {
            "content": [{"type": "text", "text": serialized}],
            "structuredContent": result,
            "isError": False,
        }


def _write_server_message(stream: BinaryIO, message: Mapping[str, Any]) -> None:
    stream.write(encode_stdio_message(message))
    stream.flush()


def serve_stdio(
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
) -> int:
    """Serve the minimal projection until the client closes stdin."""
    source = sys.stdin.buffer if input_stream is None else input_stream
    sink = sys.stdout.buffer if output_stream is None else output_stream
    session = MCPStdioSession()

    while True:
        line = source.readline(MAX_STDIO_MESSAGE_BYTES + 2)
        if not line:
            return 0
        if not line.endswith(b"\n"):
            while line and not line.endswith(b"\n"):
                line = source.readline(MAX_STDIO_MESSAGE_BYTES + 2)
            _write_server_message(
                sink, _error_response(None, PARSE_ERROR, "Parse error")
            )
            if not line:
                return 1
            continue
        try:
            message = decode_stdio_message(line)
        except MCPFrameError:
            _write_server_message(
                sink, _error_response(None, PARSE_ERROR, "Parse error")
            )
            continue
        response = session.process_message(message)
        if response is not None:
            _write_server_message(sink, response)


def _read_line_with_timeout(stream: BinaryIO, timeout_seconds: float) -> bytes:
    results: queue.Queue[bytes | BaseException] = queue.Queue(maxsize=1)

    def read_one() -> None:
        try:
            results.put(stream.readline())
        except BaseException as error:  # pragma: no cover - defensive OS boundary
            results.put(error)

    worker = threading.Thread(target=read_one, daemon=True)
    worker.start()
    try:
        result = results.get(timeout=timeout_seconds)
    except queue.Empty as error:
        raise TimeoutError("MCP stdio response timed out") from error
    if isinstance(result, BaseException):
        raise RuntimeError("MCP stdio response read failed") from result
    if not result:
        raise RuntimeError("MCP stdio server closed stdout before responding")
    return result


def _send_client_message(stream: BinaryIO, message: Mapping[str, Any]) -> None:
    stream.write(encode_stdio_message(message))
    stream.flush()


def _expect_result(response: Mapping[str, Any], request_id: int) -> Mapping[str, Any]:
    if (
        response.get("jsonrpc") != "2.0"
        or response.get("id") != request_id
        or "error" in response
        or not isinstance(response.get("result"), Mapping)
    ):
        raise RuntimeError(f"MCP request {request_id} did not return a valid result")
    return cast(Mapping[str, Any], response["result"])


def _expect_error(response: Mapping[str, Any], request_id: int) -> Mapping[str, Any]:
    if (
        response.get("jsonrpc") != "2.0"
        or response.get("id") != request_id
        or "result" in response
        or not isinstance(response.get("error"), Mapping)
    ):
        raise RuntimeError(f"MCP request {request_id} did not return a protocol error")
    return cast(Mapping[str, Any], response["error"])


def run_stdio_control(
    *,
    cwd: Path | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run one real local client/server lifecycle and return a deterministic report."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    command = [sys.executable, "-m", "about_llm.agents.mcp_stdio", "serve"]
    process: subprocess.Popen[bytes] = subprocess.Popen(
        command,
        cwd=None if cwd is None else str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("failed to open MCP stdio subprocess pipes")

    input_stream = cast(BinaryIO, process.stdin)
    output_stream = cast(BinaryIO, process.stdout)
    error_stream = cast(BinaryIO, process.stderr)
    transcript: list[dict[str, Any]] = []

    def send(message: Mapping[str, Any]) -> None:
        detached = _snapshot(message)
        transcript.append({"direction": "client_to_server", "message": detached})
        _send_client_message(input_stream, detached)

    def exchange(message: Mapping[str, Any]) -> dict[str, Any]:
        send(message)
        response = decode_stdio_message(
            _read_line_with_timeout(output_stream, timeout_seconds)
        )
        transcript.append({"direction": "server_to_client", "message": response})
        return response

    secret = "TOP-SECRET-MCP-VALUE"
    try:
        initialize_response = exchange(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "about-llm-mcp-stdio-client",
                        "version": "1.0.0",
                    },
                },
            }
        )
        initialize_result = _expect_result(initialize_response, 1)
        if initialize_result.get("protocolVersion") != MCP_PROTOCOL_VERSION:
            raise RuntimeError("MCP protocol version negotiation failed")
        capabilities = initialize_result.get("capabilities")
        if not isinstance(capabilities, Mapping) or "tools" not in capabilities:
            raise RuntimeError("MCP server did not negotiate the tools capability")

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        list_result = _expect_result(
            exchange(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                }
            ),
            2,
        )
        tools = list_result.get("tools")
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
            raise RuntimeError("MCP tools/list result is invalid")
        if len(tools) != 1 or not isinstance(tools[0], Mapping):
            raise RuntimeError("MCP fixture expected exactly one tool")
        advertised_tool = cast(Mapping[str, Any], tools[0])
        if advertised_tool.get("name") != ADD_INPUT_CONTRACT.name:
            raise RuntimeError("MCP tool identity drifted")

        valid_result = _expect_result(
            exchange(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": ADD_INPUT_CONTRACT.name,
                        "arguments": {"a": 7, "b": 5},
                    },
                }
            ),
            3,
        )
        invalid_result = _expect_result(
            exchange(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": ADD_INPUT_CONTRACT.name,
                        "arguments": {"a": 7, "b": secret},
                    },
                }
            ),
            4,
        )
        unknown_error = _expect_error(
            exchange(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "fixture.unknown", "arguments": {}},
                }
            ),
            5,
        )

        input_stream.close()
        return_code = process.wait(timeout=timeout_seconds)
        trailing_stdout = output_stream.read()
        stderr_bytes = error_stream.read()
        if return_code != 0:
            raise RuntimeError("MCP stdio server exited unsuccessfully")
        if trailing_stdout:
            raise RuntimeError("MCP stdio server wrote unexpected stdout data")
        stderr_text = stderr_bytes.decode("utf-8", errors="strict")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        raise
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    invalid_serialized = canonical_json_bytes(invalid_result).decode("utf-8")
    transcript_projection: list[dict[str, Any]] = []
    for item in transcript:
        message = cast(Mapping[str, Any], item["message"])
        projected: dict[str, Any] = {
            "direction": item["direction"],
            "jsonrpc": message.get("jsonrpc"),
        }
        if "id" in message:
            projected["id"] = message["id"]
        if "method" in message:
            projected["method"] = message["method"]
        elif "result" in message:
            projected["response_kind"] = "result"
            result_value = message["result"]
            if isinstance(result_value, Mapping) and "isError" in result_value:
                projected["tool_is_error"] = result_value["isError"]
        elif "error" in message:
            projected["response_kind"] = "error"
            error_value = message["error"]
            if isinstance(error_value, Mapping):
                projected["error_code"] = error_value.get("code")
        transcript_projection.append(projected)
    projection_fingerprint = "sha256:" + artifact_fingerprint(
        {"messages": transcript_projection}
    )
    return {
        "implementation": MCP_STDIO_CONTROL_VERSION,
        "protocol_version": MCP_PROTOCOL_VERSION,
        "transport": {
            "client_launched_server_subprocess": True,
            "utf8_jsonrpc_lf_framing_executed": True,
            "server_stdout_protocol_only": True,
            "server_stderr": stderr_text,
        },
        "handshake": {
            "negotiated_protocol_version": initialize_result["protocolVersion"],
            "server_info": initialize_result["serverInfo"],
            "tools_capability_negotiated": True,
            "initialized_notification_sent_before_tools": True,
        },
        "tool_contract": {
            "name": advertised_tool["name"],
            "input_schema_revision": ADD_INPUT_CONTRACT.schema_revision,
            "input_validator_revision": ADD_INPUT_CONTRACT.validator_revision,
            "input_schema_fingerprint": ADD_INPUT_CONTRACT.schema_fingerprint,
            "output_schema_revision": ADD_OUTPUT_CONTRACT.schema_revision,
            "output_validator_revision": ADD_OUTPUT_CONTRACT.validator_revision,
            "output_schema_fingerprint": ADD_OUTPUT_CONTRACT.schema_fingerprint,
        },
        "calls": {
            "valid": _snapshot(valid_result),
            "invalid_arguments": {
                "result": _snapshot(invalid_result),
                "rejected_value_disclosed": secret in invalid_serialized,
            },
            "unknown_tool_protocol_error": _snapshot(unknown_error),
        },
        "transcript": {
            "message_count": len(transcript),
            "request_ids": [1, 2, 3, 4, 5],
            "client_methods": [
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/call",
                "tools/call",
                "tools/call",
            ],
            "projection_fingerprint": projection_fingerprint,
            "projection_fields": [
                "direction",
                "jsonrpc",
                "id",
                "method",
                "response_kind",
                "tool_is_error",
                "error_code",
            ],
            "raw_messages_published": False,
        },
        "scope": {
            "real_local_subprocess_and_os_pipes_executed": True,
            "external_network_or_remote_server_called": False,
            "official_mcp_sdk_used": False,
            "full_mcp_schema_or_conformance_suite_executed": False,
            "streamable_http_or_authentication_executed": False,
            "a2a_client_or_server_executed": False,
            "cross_vendor_interoperability_proven": False,
            "business_authorization_or_human_approval_executed": False,
            "tool_is_bounded_local_read_only_fixture": True,
            "transcript_projection_fingerprint_proves_authenticity": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "control"))
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve_stdio()
    report = run_stdio_control()
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
