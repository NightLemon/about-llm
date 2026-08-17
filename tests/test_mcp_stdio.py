from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from about_llm.agents.mcp_stdio import (
    ADD_INPUT_CONTRACT,
    INVALID_PARAMS,
    INVALID_REQUEST,
    MCP_PROTOCOL_VERSION,
    MCPFrameError,
    MCPStdioSession,
    SessionPhase,
    decode_stdio_message,
    encode_stdio_message,
    run_stdio_control,
    serve_stdio,
)
from about_llm.llmops import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONTROL = ROOT / "projects" / "safe-agent" / "mcp_stdio_control.py"
SECRET = "TOP-SECRET-MCP-VALUE"


def request(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }


def ready_session() -> MCPStdioSession:
    session = MCPStdioSession()
    initialized = session.process_message(
        request(
            1,
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        )
    )
    assert initialized is not None and "result" in initialized
    assert (
        session.process_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        is None
    )
    assert session.phase is SessionPhase.READY
    return session


def test_stdio_frame_is_one_strict_utf8_json_object_per_lf_line() -> None:
    message = request(1, "ping", {})

    assert decode_stdio_message(encode_stdio_message(message)) == message
    for invalid in (
        b'{"jsonrpc":"2.0","method":"ping"}',
        b"\n",
        b"[]\n",
        b'{"jsonrpc":"2.0","method":"ping","method":"other"}\n',
        b'{"jsonrpc":"2.0","method":"ping","params":{"x":NaN}}\n',
        b'{"jsonrpc":"2.0",\n"method":"ping"}\n',
        b'\xff\n',
    ):
        with pytest.raises(MCPFrameError):
            decode_stdio_message(invalid)


def test_frame_byte_limit_is_checked_before_json_decoding() -> None:
    line = b'{"value":"' + (b"x" * 50) + b'"}\n'

    with pytest.raises(MCPFrameError, match="byte limit"):
        decode_stdio_message(line, max_message_bytes=20)


def test_initialize_must_precede_tools_and_initialized_notification() -> None:
    session = MCPStdioSession()

    before_initialize = session.process_message(request(1, "tools/list", {}))
    assert before_initialize is not None
    assert before_initialize["error"] == {
        "code": INVALID_REQUEST,
        "message": "Session not initialized",
    }

    initialize = session.process_message(
        request(
            2,
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        )
    )
    assert initialize is not None
    assert initialize["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert initialize["result"]["capabilities"] == {
        "tools": {"listChanged": False}
    }
    assert session.phase is SessionPhase.AWAITING_INITIALIZED

    before_notification = session.process_message(request(3, "tools/list", {}))
    assert before_notification is not None
    assert before_notification["error"]["code"] == INVALID_REQUEST

    assert (
        session.process_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        is None
    )
    assert session.phase is SessionPhase.READY


def test_server_returns_its_supported_version_when_client_requests_another() -> None:
    session = MCPStdioSession()

    response = session.process_message(
        request(
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "old-client", "version": "1.0.0"},
            },
        )
    )

    assert response is not None
    assert response["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert session.client_requested_version == "2024-11-05"


def test_tool_advertisement_uses_explicit_closed_input_and_output_schemas() -> None:
    session = ready_session()

    response = session.process_message(request(2, "tools/list", {}))

    assert response is not None
    tools = response["result"]["tools"]
    assert len(tools) == 1
    tool = tools[0]
    assert tool["name"] == "fixture.add"
    assert tool["execution"] == {"taskSupport": "forbidden"}
    assert tool["inputSchema"]["$schema"].endswith("draft/2020-12/schema")
    assert tool["inputSchema"]["additionalProperties"] is False
    assert tool["outputSchema"]["additionalProperties"] is False
    assert tool["inputSchema"] == json.loads(
        canonical_json_bytes(ADD_INPUT_CONTRACT.arguments_schema)
    )


def test_tool_success_structured_content_and_redacted_validation_error() -> None:
    session = ready_session()

    success = session.process_message(
        request(
            2,
            "tools/call",
            {"name": "fixture.add", "arguments": {"a": 7, "b": 5}},
        )
    )
    rejected = session.process_message(
        request(
            3,
            "tools/call",
            {"name": "fixture.add", "arguments": {"a": 7, "b": SECRET}},
        )
    )

    assert success is not None
    assert success["result"] == {
        "content": [{"type": "text", "text": '{"sum":12}'}],
        "structuredContent": {"sum": 12},
        "isError": False,
    }
    assert rejected is not None
    assert rejected["result"]["isError"] is True
    serialized = canonical_json_bytes(rejected).decode("utf-8")
    assert "keyword=type" in serialized
    assert "instance_path=/b" in serialized
    assert SECRET not in serialized


def test_unknown_tool_is_a_protocol_error_not_a_tool_result() -> None:
    session = ready_session()

    response = session.process_message(
        request(
            2,
            "tools/call",
            {"name": "fixture.unknown", "arguments": {}},
        )
    )

    assert response is not None
    assert "result" not in response
    assert response["error"] == {"code": INVALID_PARAMS, "message": "Unknown tool"}


@pytest.mark.parametrize(
    "message",
    [
        {"jsonrpc": "1.0", "id": 1, "method": "ping", "params": {}},
        {"jsonrpc": "2.0", "id": True, "method": "ping", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "ping",
            "params": {},
            "unknown": True,
        },
    ],
)
def test_invalid_jsonrpc_envelope_fails_closed(message: dict[str, Any]) -> None:
    response = MCPStdioSession().process_message(message)

    assert response is not None
    assert response["error"]["code"] == INVALID_REQUEST


def test_binary_server_loop_emits_only_protocol_responses() -> None:
    initialize = request(
        1,
        "initialize",
        {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "bytes-client", "version": "1.0.0"},
        },
    )
    source = io.BytesIO(
        encode_stdio_message(initialize)
        + encode_stdio_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        + encode_stdio_message(request(2, "tools/list", {}))
    )
    sink = io.BytesIO()

    assert serve_stdio(source, sink) == 0
    lines = sink.getvalue().splitlines(keepends=True)
    assert len(lines) == 2
    assert [decode_stdio_message(line)["id"] for line in lines] == [1, 2]


def test_real_subprocess_control_is_deterministic_and_scope_limited() -> None:
    report = run_stdio_control(cwd=ROOT)

    assert report["implementation"] == "about-llm.mcp-stdio-control.v1"
    assert report["protocol_version"] == MCP_PROTOCOL_VERSION
    assert report["transport"] == {
        "client_launched_server_subprocess": True,
        "utf8_jsonrpc_lf_framing_executed": True,
        "server_stdout_protocol_only": True,
        "server_stderr": "",
    }
    assert report["calls"]["valid"]["structuredContent"] == {"sum": 12}
    assert report["calls"]["invalid_arguments"]["rejected_value_disclosed"] is False
    assert report["calls"]["unknown_tool_protocol_error"]["code"] == INVALID_PARAMS
    assert report["transcript"] == {
        "message_count": 11,
        "request_ids": [1, 2, 3, 4, 5],
        "client_methods": [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
            "tools/call",
            "tools/call",
        ],
        "projection_fingerprint": (
            "sha256:5be5bed393fd66ef3269fe10452164423c29adfaa5cb59e2a5086b8eb7f64256"
        ),
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
    }
    assert report["scope"] == {
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
    }
    assert SECRET.encode() not in canonical_json_bytes(report)

    completed = subprocess.run(
        [sys.executable, "-m", "about_llm.agents.mcp_stdio", "control"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == report
    assert completed.stderr == b""

    project_completed = subprocess.run(
        [sys.executable, str(PROJECT_CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert json.loads(project_completed.stdout) == report
    assert project_completed.stderr == b""
