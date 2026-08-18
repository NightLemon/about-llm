from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from about_llm.agents import mcp_streamable_http as mcp_http
from about_llm.agents.mcp_stdio import MCP_PROTOCOL_VERSION, SessionPhase
from about_llm.agents.mcp_streamable_http import (
    MCP_ENDPOINT_PATH,
    MCP_STREAMABLE_HTTP_CONTROL_VERSION,
    WAIT_TOOL,
    MCPHTTPBodyError,
    MCPHTTPSession,
    _is_client_response,
    _listed_media_types,
    _sse_event,
    _visible_ascii,
    build_server_app,
    decode_http_json,
    run_streamable_http_control,
    serve,
)

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONTROL = ROOT / "projects" / "safe-agent" / "mcp_streamable_http_control.py"


def initialize_message() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0.0"},
        },
    }


def test_decode_http_json_is_strict_bounded_and_finite() -> None:
    assert decode_http_json(b'{"ok":true}') == {"ok": True}
    invalid = [
        b"",
        b"[]",
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":1e400}',
        b"\xff",
        b"{",
        b" " * 64_001,
    ]
    for raw in invalid:
        with pytest.raises(MCPHTTPBodyError):
            decode_http_json(raw)


def test_header_media_and_visible_ascii_helpers_fail_closed() -> None:
    assert _listed_media_types(
        "application/json; q=1, text/event-stream; q=0.5, text/plain;q=0"
    ) == frozenset({"application/json", "text/event-stream"})
    assert _listed_media_types("application/json;q=bad") == frozenset()
    assert _visible_ascii("abc-123")
    assert not _visible_ascii("")
    assert not _visible_ascii("contains space")
    assert not _visible_ascii("换行\n")


def test_jsonrpc_client_response_and_sse_framing_are_strict() -> None:
    assert _is_client_response({"jsonrpc": "2.0", "id": 1, "result": {}})
    assert _is_client_response(
        {
            "jsonrpc": "2.0",
            "id": "request-a",
            "error": {"code": -32600, "message": "Invalid Request"},
        }
    )
    assert not _is_client_response({"jsonrpc": "2.0", "id": True, "result": {}})
    assert not _is_client_response(
        {"jsonrpc": "2.0", "id": 1, "result": {}, "extra": True}
    )
    assert _sse_event("stream.1", "") == b"id: stream.1\ndata: \n\n"
    with pytest.raises(ValueError):
        _sse_event("bad\nid", "")
    with pytest.raises(ValueError):
        _sse_event("stream.1", "two\nlines")


def test_http_session_extends_lifecycle_with_discovery_and_cancellation() -> None:
    async def exercise() -> None:
        session = MCPHTTPSession()
        initialized = session.process_message(initialize_message())
        assert initialized is not None and "result" in initialized
        assert session.phase is SessionPhase.AWAITING_INITIALIZED
        assert (
            session.process_message(
                {"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
            is None
        )
        listed = session.process_message(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )
        assert listed is not None
        assert [tool["name"] for tool in listed["result"]["tools"]] == [
            "fixture.add",
            WAIT_TOOL["name"],
        ]

        malformed = session.begin_wait(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": WAIT_TOOL["name"], "arguments": {"extra": 1}},
            }
        )
        assert isinstance(malformed, dict)
        assert malformed["error"]["code"] == -32602
        pending = session.begin_wait(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": WAIT_TOOL["name"], "arguments": {}},
            }
        )
        assert not isinstance(pending, dict)
        session.process_message(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": "unknown", "reason": "ignored"},
            }
        )
        assert not pending.cancelled.is_set()
        session.process_message(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": 4, "reason": "test"},
            }
        )
        assert pending.cancelled.is_set()
        session.finish_wait(4)
        assert not session.pending

    asyncio.run(exercise())


def test_in_process_asgi_transport_rejects_http_and_session_drift() -> None:
    async def exercise() -> None:
        token = "t" * 32
        app = build_server_app(
            bearer_token=token,
            allowed_origin="https://client.invalid",
        )
        transport = httpx.ASGITransport(app=app)
        base_headers = {
            "Authorization": f"Bearer {token}",
            "Origin": "https://client.invalid",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            invalid_origin = await client.post(
                MCP_ENDPOINT_PATH,
                headers={**base_headers, "Origin": "https://attacker.invalid"},
                json=initialize_message(),
            )
            missing_auth_headers = dict(base_headers)
            del missing_auth_headers["Authorization"]
            missing_auth = await client.post(
                MCP_ENDPOINT_PATH,
                headers=missing_auth_headers,
                json=initialize_message(),
            )
            duplicate_auth = await client.post(
                MCP_ENDPOINT_PATH,
                headers=[
                    ("Authorization", f"Bearer {token}"),
                    ("Authorization", f"Bearer {token}"),
                    ("Origin", "https://client.invalid"),
                    ("Accept", "application/json, text/event-stream"),
                    ("Content-Type", "application/json"),
                ],
                json=initialize_message(),
            )
            assert invalid_origin.status_code == 403
            assert missing_auth.status_code == duplicate_auth.status_code == 401
            assert missing_auth.headers["www-authenticate"].startswith("Bearer ")

            duplicate_accept = await client.post(
                MCP_ENDPOINT_PATH,
                headers=[
                    ("Authorization", f"Bearer {token}"),
                    ("Origin", "https://client.invalid"),
                    ("Accept", "application/json"),
                    ("Accept", "text/event-stream"),
                    ("Content-Type", "application/json"),
                ],
                json=initialize_message(),
            )
            unacceptable = await client.post(
                MCP_ENDPOINT_PATH,
                headers={**base_headers, "Accept": "application/json"},
                json=initialize_message(),
            )
            wrong_content_type = await client.post(
                MCP_ENDPOINT_PATH,
                headers={**base_headers, "Content-Type": "text/plain"},
                content=b"{}",
            )
            invalid_json = await client.post(
                MCP_ENDPOINT_PATH,
                headers=base_headers,
                content=b'{"x":1,"x":2}',
            )
            oversized = await client.post(
                MCP_ENDPOINT_PATH,
                headers=base_headers,
                content=b" " * 64_001,
            )
            assert duplicate_accept.status_code == 400
            assert unacceptable.status_code == 406
            assert wrong_content_type.status_code == 415
            assert invalid_json.status_code == 400
            assert oversized.status_code == 413

            not_initialize = await client.post(
                MCP_ENDPOINT_PATH,
                headers=base_headers,
                json={"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}},
            )
            invalid_initialize = await client.post(
                MCP_ENDPOINT_PATH,
                headers=base_headers,
                json={"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}},
            )
            assert not_initialize.status_code == 400
            assert invalid_initialize.status_code == 200
            assert invalid_initialize.json()["error"]["code"] == -32602
            assert "mcp-session-id" not in invalid_initialize.headers

            initialized = await client.post(
                MCP_ENDPOINT_PATH,
                headers=base_headers,
                json=initialize_message(),
            )
            assert initialized.status_code == 200
            session_id = initialized.headers["mcp-session-id"]
            session_headers = {
                **base_headers,
                "MCP-Session-Id": session_id,
                "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            }
            missing_version_headers = dict(session_headers)
            del missing_version_headers["MCP-Protocol-Version"]
            missing_version = await client.post(
                MCP_ENDPOINT_PATH,
                headers=missing_version_headers,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            unknown_session = await client.post(
                MCP_ENDPOINT_PATH,
                headers={**session_headers, "MCP-Session-Id": "unknown-session"},
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            assert missing_version.status_code == 400
            assert unknown_session.status_code == 404

            notification = await client.post(
                MCP_ENDPOINT_PATH,
                headers=session_headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            client_response = await client.post(
                MCP_ENDPOINT_PATH,
                headers=session_headers,
                json={"jsonrpc": "2.0", "id": "server-request", "result": {}},
            )
            assert notification.status_code == client_response.status_code == 202
            assert not notification.content and not client_response.content

            listed = await client.post(
                MCP_ENDPOINT_PATH,
                headers=session_headers,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
            assert [tool["name"] for tool in listed.json()["result"]["tools"]] == [
                "fixture.add",
                "fixture.wait",
            ]
            invalid_wait = await client.post(
                MCP_ENDPOINT_PATH,
                headers=session_headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "fixture.wait",
                        "arguments": {"unexpected": True},
                    },
                },
            )
            assert invalid_wait.json()["error"]["code"] == -32602

            add = await client.post(
                MCP_ENDPOINT_PATH,
                headers=session_headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "fixture.add", "arguments": {"a": 1, "b": 2}},
                },
            )
            assert add.status_code == 200
            assert add.headers["content-type"].startswith("text/event-stream")
            assert b'"sum":3' in add.content

            get_headers = {
                key: value
                for key, value in session_headers.items()
                if key not in {"Content-Type", "Accept"}
            }
            get_headers["Accept"] = "text/event-stream"
            unsupported_resume = await client.get(
                MCP_ENDPOINT_PATH,
                headers={**get_headers, "Last-Event-ID": "unretained.1"},
            )
            listened = await client.get(MCP_ENDPOINT_PATH, headers=get_headers)
            assert unsupported_resume.status_code == 400
            assert listened.status_code == 200
            assert listened.content.count(b"data: \n\n") == 1

            deleted = await client.delete(MCP_ENDPOINT_PATH, headers=session_headers)
            after_delete = await client.get(MCP_ENDPOINT_PATH, headers=get_headers)
            assert deleted.status_code == 204 and not deleted.content
            assert after_delete.status_code == 404

    asyncio.run(exercise())


def test_wait_timeout_and_malformed_cancellations_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        session = MCPHTTPSession()
        session.process_message(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": 4},
            }
        )
        session.process_message(initialize_message())
        session.process_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        pending = session.begin_wait(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "fixture.wait", "arguments": {}},
            }
        )
        assert not isinstance(pending, dict)
        duplicate = session.begin_wait(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "fixture.wait", "arguments": {}},
            }
        )
        assert isinstance(duplicate, dict)
        assert duplicate["error"]["code"] == -32600
        malformed = [
            {"jsonrpc": "1.0", "method": "notifications/cancelled", "params": {"requestId": 4}},
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": []},
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": 4, "extra": True},
            },
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": True},
            },
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": 4, "reason": 7},
            },
        ]
        for message in malformed:
            session.process_message(message)
            assert not pending.cancelled.is_set()

        monkeypatch.setattr(mcp_http, "WAIT_SERVER_TIMEOUT_SECONDS", 0.001)
        chunks = [
            chunk
            async for chunk in mcp_http.MCPStreamableHTTPServer._cancelled_wait_stream(
                session, pending
            )
        ]
        assert len(chunks) == 2
        assert b'"code":-32000' in chunks[1]
        assert 4 not in session.pending

        second = session.begin_wait(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "fixture.wait", "arguments": {}},
            }
        )
        assert not isinstance(second, dict)
        session.terminate()
        assert second.cancelled.is_set()
        assert not session.pending

    asyncio.run(exercise())


def test_server_configuration_rejects_unsafe_fixture_inputs() -> None:
    with pytest.raises(ValueError, match="bearer token"):
        build_server_app(bearer_token="short", allowed_origin="https://client.invalid")
    with pytest.raises(ValueError, match="HTTPS"):
        build_server_app(bearer_token="x" * 32, allowed_origin="http://client.invalid")
    with pytest.raises(ValueError, match="loopback"):
        serve(
            "0.0.0.0",
            1,
            bearer_token="x" * 32,
            allowed_origin="https://client.invalid",
        )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.extended
def test_project_control_executes_real_streamable_http_loopback() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)

    assert completed.stderr == ""
    assert report["implementation"] == MCP_STREAMABLE_HTTP_CONTROL_VERSION
    assert report["protocol_version"] == MCP_PROTOCOL_VERSION
    assert report["binding"] == "Streamable HTTP"
    assert report["network"] == {
        "scheme": "http",
        "address_scope": "IPv4 loopback",
        "real_tcp_http": True,
        "tls": False,
    }
    assert report["transport"] == {
        "single_endpoint_path": MCP_ENDPOINT_PATH,
        "post_json_response_executed": True,
        "post_sse_response_executed": True,
        "get_sse_executed": True,
        "delete_session_executed": True,
        "notification_empty_202_verified": True,
        "sse_priming_event_with_id_verified": True,
        "event_ids_unique_within_session": True,
    }
    assert report["security_controls"] == {
        "origin_allowlist_executed": True,
        "invalid_origin_status": 403,
        "bearer_header_gate_executed": True,
        "missing_or_wrong_bearer_status": 401,
        "oauth_flow_executed": False,
    }
    assert report["session"] == {
        "server_assigned_on_initialize": True,
        "visible_ascii_and_minimum_length_verified": True,
        "included_on_subsequent_requests": True,
        "missing_session_status": 400,
        "missing_or_unsupported_protocol_version_status": 400,
        "terminated_session_status": 404,
    }
    assert report["cancellation"] == {
        "concurrent_request_and_notification_executed": True,
        "notification_status": 202,
        "jsonrpc_response_after_cancellation_count": 0,
        "stream_closed_after_cancellation": True,
    }
    assert report["tool_result"] == {
        "structured_output_local_verifier_passed": True,
        "raw_arguments_or_result_published": False,
    }
    assert report["projection_fingerprint"] == (
        "sha256:5a5cc3be24268d3dec80edb3613e51ffed3dc0d0d6535f7039c74386ce7c8915"
    )
    assert report["server_process"]["stdout_stderr_empty"] is True
    assert report["raw_http_messages_published"] is False
    assert report["secret_or_session_identifiers_published"] is False
    assert all(value is False for value in report["evidence_limits"].values())
    serialized = json.dumps(report, ensure_ascii=False)
    assert "REJECTED-MCP-HTTP-TOKEN" not in serialized
    assert "about-llm-mcp-http-client" not in serialized
    assert '"sum": 12' not in serialized
    assert '"a": 7' not in serialized
def test_invalid_control_timeouts() -> None:
    with pytest.raises(ValueError, match="timeouts"):
        run_streamable_http_control(request_timeout_seconds=0)
