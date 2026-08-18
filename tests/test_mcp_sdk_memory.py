from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from about_llm.agents.mcp_sdk_memory import (
    MCP_SDK_MEMORY_EVIDENCE_BOUNDARY,
    MCP_SDK_PROTOCOL_VERSION,
    MCP_SDK_REVIEWED_VERSION,
    run_mcp_sdk_memory_control,
    verify_mcp_sdk_memory_report,
)
from about_llm.llmops import artifact_fingerprint

pytestmark = [pytest.mark.contract, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONTROL = ROOT / "projects" / "safe-agent" / "mcp_sdk_memory_control.py"


def test_official_sdk_memory_control_executes_reviewed_protocol_and_gates() -> None:
    report = run_mcp_sdk_memory_control()

    assert report["runtime"]["sdk_version"] == MCP_SDK_REVIEWED_VERSION
    assert report["initialization"]["protocol_version"] == MCP_SDK_PROTOCOL_VERSION
    assert report["calls"] == {
        "successful_sum": 5,
        "success_is_error": False,
        "invalid_schema_is_error": True,
        "invalid_schema_handler_delta": 0,
        "unknown_tool_is_error": True,
        "unknown_tool_handler_delta": 1,
        "recognized_handler_calls": 1,
        "total_handler_calls": 2,
        "raw_error_content_published": False,
    }
    assert report["transport"]["official_sdk_memory_stream"] is True
    assert report["transport"]["os_stdio"] is False
    assert report["scope"]["official_conformance_suite_executed"] is False
    assert "application handler for an unlisted tool" in MCP_SDK_MEMORY_EVIDENCE_BOUNDARY
    assert verify_mcp_sdk_memory_report(report) == report


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda report: report["calls"].__setitem__(
                "invalid_schema_handler_delta", 1
            ),
            "fingerprint",
        ),
        (
            lambda report: report["scope"].__setitem__(
                "remote_or_cross_vendor_interop_proven", True
            ),
            "fingerprint",
        ),
        (
            lambda report: report["transport"].__setitem__("tcp_http", True),
            "fingerprint",
        ),
    ],
)
def test_report_rejects_unrehashed_drift(mutate: object, match: str) -> None:
    report = run_mcp_sdk_memory_control()
    tampered = copy.deepcopy(report)
    mutate(tampered)  # type: ignore[operator]

    with pytest.raises(ValueError, match=match):
        verify_mcp_sdk_memory_report(tampered)


def test_report_rejects_cooperatively_rehashed_semantic_drift() -> None:
    report = run_mcp_sdk_memory_control()
    report["scope"]["production_readiness_proven"] = True
    unsigned = copy.deepcopy(report)
    del unsigned["report_fingerprint"]
    report["report_fingerprint"] = artifact_fingerprint(unsigned)

    with pytest.raises(ValueError, match="semantic drift"):
        verify_mcp_sdk_memory_report(report)
