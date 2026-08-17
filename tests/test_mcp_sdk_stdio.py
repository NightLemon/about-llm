from __future__ import annotations

import copy
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from about_llm.agents.mcp_sdk_stdio import (
    MCP_SDK_PROTOCOL_VERSION,
    MCP_SDK_REVIEWED_VERSION,
    MCP_SDK_STDIO_EVIDENCE_BOUNDARY,
    _load_server_receipt,
    run_mcp_sdk_stdio_control,
    verify_mcp_sdk_stdio_report,
)
from about_llm.llmops import artifact_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONTROL = ROOT / "projects" / "safe-agent" / "mcp_sdk_stdio_control.py"


def test_official_sdk_real_stdio_control_executes_sdk_and_os_pipes() -> None:
    report = run_mcp_sdk_stdio_control()

    assert report["runtime"]["sdk_version"] == MCP_SDK_REVIEWED_VERSION
    assert report["initialization"]["protocol_version"] == MCP_SDK_PROTOCOL_VERSION
    assert report["transport"] == {
        "kind": "official_sdk_stdio_subprocess",
        "client_transport": "mcp.client.stdio.stdio_client",
        "server_transport": "mcp.server.stdio.stdio_server",
        "client_launched_server_subprocess": True,
        "os_stdin_stdout_pipes": True,
        "encoding_profile": "client=utf-8-strict;server-stdin=utf-8-replace",
        "server_process_distinct": True,
        "graceful_eof_shutdown_observed": True,
        "server_stderr_empty": True,
        "raw_transcript_published": False,
    }
    assert report["calls"] == {
        "successful_sum": 5,
        "success_is_error": False,
        "invalid_schema_is_error": True,
        "invalid_schema_handler_delta": 0,
        "unknown_tool_is_error": True,
        "unknown_tool_handler_delta": 1,
        "raw_error_content_published": False,
    }
    assert report["server_receipt"]["handler_events"] == [
        "fixture.add",
        "fixture.missing",
    ]
    assert report["scope"]["malformed_raw_framing_controls_executed"] is False
    assert report["scope"]["authentication_or_authorization_proven"] is False
    assert "does not independently inject malformed framing" in (
        MCP_SDK_STDIO_EVIDENCE_BOUNDARY
    )
    assert verify_mcp_sdk_stdio_report(report) == report


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["transport"].__setitem__(
            "server_process_distinct", False
        ),
        lambda report: report["server_receipt"].__setitem__(
            "handler_events", ["fixture.add"]
        ),
        lambda report: report["scope"].__setitem__(
            "remote_or_cross_vendor_interop_proven", True
        ),
    ],
)
def test_report_rejects_unrehased_drift(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    report = run_mcp_sdk_stdio_control()
    tampered = copy.deepcopy(report)
    mutate(tampered)

    with pytest.raises(ValueError, match="fingerprint"):
        verify_mcp_sdk_stdio_report(tampered)


def test_report_rejects_cooperatively_rehashed_semantic_drift() -> None:
    report = run_mcp_sdk_stdio_control()
    report["scope"]["production_readiness_proven"] = True
    unsigned = copy.deepcopy(report)
    del unsigned["report_fingerprint"]
    report["report_fingerprint"] = artifact_fingerprint(unsigned)

    with pytest.raises(ValueError, match="semantic drift"):
        verify_mcp_sdk_stdio_report(report)


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

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "about_llm.agents.mcp_sdk_stdio",
            "serve",
            "--receipt",
            str(receipt.resolve()),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert receipt.read_text(encoding="utf-8") == "sentinel"


