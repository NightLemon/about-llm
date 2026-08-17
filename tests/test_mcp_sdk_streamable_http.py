from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from about_llm.agents.mcp_sdk_streamable_http import (
    MCP_SDK_HTTP_EVIDENCE_BOUNDARY,
    MCP_SDK_HTTP_TOKEN_ENV,
    MCP_SDK_PROTOCOL_VERSION,
    MCP_SDK_REVIEWED_VERSION,
    _load_server_receipt,
    run_mcp_sdk_http_control,
    verify_mcp_sdk_http_report,
)
from about_llm.llmops import artifact_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONTROL = (
    ROOT / "projects" / "safe-agent" / "mcp_sdk_streamable_http_control.py"
)


@pytest.fixture(scope="module")
def official_http_report() -> dict[str, Any]:
    return run_mcp_sdk_http_control()


def test_official_sdk_http_control_executes_real_transport(
    official_http_report: dict[str, Any],
) -> None:
    report = official_http_report

    assert report["runtime"]["sdk_version"] == MCP_SDK_REVIEWED_VERSION
    assert report["initialization"]["protocol_version"] == MCP_SDK_PROTOCOL_VERSION
    assert report["transport"] == {
        "kind": "official_sdk_streamable_http_subprocess",
        "client_transport": "mcp.client.streamable_http.streamable_http_client",
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
    assert report["http_observations"] == {
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
    assert report["server_receipt"]["handler_events"] == [
        "fixture.add",
        "fixture.missing",
    ]
    assert report["scope"]["malformed_http_controls_executed"] is False
    assert report["scope"]["private_control_token_gate_executed"] is True
    assert report["scope"]["authentication_or_authorization_proven"] is False
    assert "not MCP authentication or authorization" in MCP_SDK_HTTP_EVIDENCE_BOUNDARY
    assert verify_mcp_sdk_http_report(report) == report


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["transport"].__setitem__(
            "mcp_session_termination_delete_observed", False
        ),
        lambda report: report["http_observations"].__setitem__("delete_count", 0),
        lambda report: report["scope"].__setitem__(
            "remote_or_cross_vendor_interop_proven", True
        ),
    ],
)
def test_report_rejects_unrehased_drift(
    official_http_report: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    tampered = copy.deepcopy(official_http_report)
    mutate(tampered)

    with pytest.raises(ValueError, match="fingerprint"):
        verify_mcp_sdk_http_report(tampered)


def test_report_rejects_cooperatively_rehashed_semantic_drift(
    official_http_report: dict[str, Any],
) -> None:
    tampered = copy.deepcopy(official_http_report)
    tampered["scope"]["production_readiness_proven"] = True
    unsigned = copy.deepcopy(tampered)
    del unsigned["report_fingerprint"]
    tampered["report_fingerprint"] = artifact_fingerprint(unsigned)

    with pytest.raises(ValueError, match="semantic drift"):
        verify_mcp_sdk_http_report(tampered)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"receipt_version":"x","receipt_version":"y"}',
        b'{"x":NaN}',
        b'{"x":1}\n',
        b"x" * 4_097,
    ],
)
def test_server_receipt_loader_rejects_noncanonical_or_unsafe_json(
    tmp_path: Path, payload: bytes
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(payload)

    with pytest.raises(ValueError):
        _load_server_receipt(receipt)


def test_server_mode_refuses_to_overwrite_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("sentinel", encoding="utf-8")
    environment = {**os.environ, MCP_SDK_HTTP_TOKEN_ENV: "x" * 32}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "about_llm.agents.mcp_sdk_streamable_http",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "65534",
            "--receipt",
            str(receipt.resolve()),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode != 0
    assert receipt.read_text(encoding="utf-8") == "sentinel"


def test_project_control_emits_closed_verified_json() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)

    assert verify_mcp_sdk_http_report(report) == report
    assert completed.stderr == ""
    assert "TOP-SECRET" not in completed.stdout
