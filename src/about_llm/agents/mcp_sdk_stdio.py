"""Official MCP SDK client/server control over a real stdio subprocess.

The repository keeps this control separate from both the official-SDK memory
control and the authored strict stdio protocol subset.  This module proves that
the reviewed SDK client and server can communicate through a subprocess and OS
stdin/stdout pipes.  It does not independently test malformed byte framing,
conformance, authentication, authorization, remote interoperability, or
production process supervision.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import logging
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Final, TextIO, cast

import anyio
import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.stdio import stdio_server
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

from about_llm.agents.mcp_sdk_memory import (
    MCP_SDK_PROTOCOL_VERSION,
    MCP_SDK_REVIEWED_VERSION,
    build_mcp_sdk_fixture_server,
)
from about_llm.agents.mcp_stdio import ADD_INPUT_CONTRACT, ADD_OUTPUT_CONTRACT
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

MCP_SDK_STDIO_CONTROL_VERSION: Final = "about-llm.mcp-sdk-stdio-control.v1"
MCP_SDK_STDIO_RECEIPT_VERSION: Final = "about-llm.mcp-sdk-stdio-receipt.v1"
MCP_SDK_STDIO_CHECKED_AT: Final = "2026-08-14"
MCP_SDK_STDIO_MAX_RECEIPT_BYTES: Final = 4_096
MCP_SDK_STDIO_EVIDENCE_BOUNDARY: Final = (
    "This control uses mcp 1.29.0 ClientSession, low-level Server, generated "
    "types, stdio_client, and stdio_server for MCP 2025-11-25. The official "
    "client launches a distinct Python server subprocess and exchanges protocol "
    "messages over real OS stdin/stdout pipes. The client is configured for "
    "strict UTF-8 while the reviewed official server decodes stdin as UTF-8 "
    "with replacement. It "
    "executes discovery, one successful structured call, SDK schema rejection, "
    "and an application unknown-tool gate. It does not publish the raw transcript "
    "or SDK error content and does not independently inject malformed framing, "
    "duplicate-key, invalid-UTF-8, byte-cap, forced-termination, or cancellation "
    "cases. It does not execute HTTP/SSE, TLS, OAuth, remote or cross-vendor "
    "interoperability, the official conformance suite, authorization, approval, "
    "side effects, or production supervision. The minimized local receipt and "
    "unkeyed fingerprints do not authenticate the process, execution, or source."
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
        "client_launched_server_subprocess",
        "os_stdin_stdout_pipes",
        "encoding_profile",
        "server_process_distinct",
        "graceful_eof_shutdown_observed",
        "server_stderr_empty",
        "raw_transcript_published",
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
_SERVER_RECEIPT_FIELDS: Final = frozenset(
    {
        "handler_events",
        "recognized_handler_calls",
        "total_handler_calls",
        "server_run_completed",
        "raw_arguments_or_results_published",
        "receipt_fingerprint",
    }
)
_SCOPE_FIELDS: Final = frozenset(
    {
        "official_sdk_client_executed",
        "official_sdk_server_executed",
        "official_sdk_stdio_client_executed",
        "official_sdk_stdio_server_executed",
        "real_subprocess_and_os_pipes_executed",
        "mcp_2025_11_25_negotiated",
        "official_generated_types_executed",
        "sdk_json_schema_validation_executed",
        "application_unknown_tool_gate_executed",
        "malformed_raw_framing_controls_executed",
        "streamable_http_transport_executed",
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
        "server_run_completed",
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
    if not raw or len(raw) > MCP_SDK_STDIO_MAX_RECEIPT_BYTES:
        raise ValueError("official MCP SDK stdio receipt size is invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("official MCP SDK stdio receipt is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError("official MCP SDK stdio receipt must be an object")
    if canonical_json_bytes(value) != raw:
        raise ValueError("official MCP SDK stdio receipt is not canonical JSON")
    _exact(value, _RECEIPT_FIELDS, "server receipt")
    return value


def _write_server_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(receipt)
    if len(raw) > MCP_SDK_STDIO_MAX_RECEIPT_BYTES:
        raise ValueError("official MCP SDK stdio receipt exceeds its byte cap")
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


async def _serve_sdk_stdio(receipt_path: Path) -> None:
    sdk_version = _assert_reviewed_sdk()
    if not receipt_path.is_absolute() or not receipt_path.parent.is_dir():
        raise ValueError("server receipt path must be absolute with an existing parent")
    if receipt_path.exists():
        raise FileExistsError("server receipt path already exists")

    handler_events: list[str] = []
    server, counters = build_mcp_sdk_fixture_server(
        handler_events=handler_events,
        server_name="about-llm-mcp-sdk-stdio",
    )
    run_completed = False
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
                raise_exceptions=False,
            )
        run_completed = True
    finally:
        receipt = {
            "receipt_version": MCP_SDK_STDIO_RECEIPT_VERSION,
            "sdk_version": sdk_version,
            "protocol_version": MCP_SDK_PROTOCOL_VERSION,
            "server_pid": os.getpid(),
            "handler_events": list(handler_events),
            "recognized_handler_calls": counters["recognized"],
            "total_handler_calls": counters["total"],
            "server_run_completed": run_completed,
            "raw_arguments_or_results_published": False,
        }
        _write_server_receipt(receipt_path, receipt)


async def _run_mcp_sdk_stdio_control() -> dict[str, Any]:
    sdk_version = _assert_reviewed_sdk()
    sdk_loggers = ("mcp", "FastMCP")
    prior_levels = {name: logging.getLogger(name).level for name in sdk_loggers}
    for name in sdk_loggers:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    try:
        with (
            tempfile.TemporaryDirectory(prefix="about-llm-mcp-sdk-stdio-") as temp,
            tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as error_file,
        ):
            receipt_path = (Path(temp) / "server-receipt.json").resolve()
            parameters = StdioServerParameters(
                command=sys.executable,
                args=[
                    "-m",
                    "about_llm.agents.mcp_sdk_stdio",
                    "serve",
                    "--receipt",
                    str(receipt_path),
                ],
                env={"PYTHONUTF8": "1"},
                encoding="utf-8",
                encoding_error_handler="strict",
            )
            async with (
                stdio_client(
                    parameters, errlog=cast(TextIO, error_file)
                ) as (read_stream, write_stream),
                ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    read_timeout_seconds=timedelta(seconds=5),
                    client_info=mcp_types.Implementation(
                        name="about-llm-sdk-stdio-client", version="1.0.0"
                    ),
                ) as session,
            ):
                initialized = await session.initialize()
                await session.send_ping()
                listed = await session.list_tools()
                if len(listed.tools) != 1:
                    raise ValueError("official SDK stdio tool discovery count drift")
                tool = listed.tools[0]
                if tool.name != ADD_INPUT_CONTRACT.name:
                    raise ValueError("official SDK stdio tool identity drift")
                success = await session.call_tool(
                    ADD_INPUT_CONTRACT.name, {"a": 2, "b": 3}
                )
                invalid = await session.call_tool(
                    ADD_INPUT_CONTRACT.name,
                    {"a": 2, "b": 3, "extra": 1},
                )
                unknown = await session.call_tool("fixture.missing", {})

            error_file.seek(0)
            server_stderr = error_file.read()
            if server_stderr:
                raise ValueError("official MCP SDK stdio server wrote to stderr")
            receipt = _load_server_receipt(receipt_path)
    finally:
        for name, level in prior_levels.items():
            logging.getLogger(name).setLevel(level)

    if success.isError or success.structuredContent != {"sum": 5}:
        raise ValueError("official SDK stdio successful result drift")
    if invalid.isError is not True or unknown.isError is not True:
        raise ValueError("official SDK stdio negative control did not fail closed")
    if receipt != {
        "receipt_version": MCP_SDK_STDIO_RECEIPT_VERSION,
        "sdk_version": MCP_SDK_REVIEWED_VERSION,
        "protocol_version": MCP_SDK_PROTOCOL_VERSION,
        "server_pid": receipt.get("server_pid"),
        "handler_events": [ADD_INPUT_CONTRACT.name, "fixture.missing"],
        "recognized_handler_calls": 1,
        "total_handler_calls": 2,
        "server_run_completed": True,
        "raw_arguments_or_results_published": False,
    }:
        raise ValueError("official SDK stdio server receipt semantic drift")
    server_pid = receipt.get("server_pid")
    if isinstance(server_pid, bool) or not isinstance(server_pid, int):
        raise ValueError("official SDK stdio server pid is invalid")
    if server_pid <= 0 or server_pid == os.getpid():
        raise ValueError("official SDK stdio server was not a distinct process")

    capabilities = initialized.capabilities
    if initialized.protocolVersion != MCP_SDK_PROTOCOL_VERSION:
        raise ValueError("official SDK stdio negotiated protocol drift")
    if capabilities.tools is None:
        raise ValueError("official SDK stdio did not negotiate tools capability")

    input_schema = tool.inputSchema
    output_schema = cast(dict[str, Any], tool.outputSchema)
    receipt_projection = {
        "receipt_version": receipt["receipt_version"],
        "sdk_version": receipt["sdk_version"],
        "protocol_version": receipt["protocol_version"],
        "handler_events": receipt["handler_events"],
        "recognized_handler_calls": receipt["recognized_handler_calls"],
        "total_handler_calls": receipt["total_handler_calls"],
        "server_run_completed": receipt["server_run_completed"],
        "raw_arguments_or_results_published": receipt[
            "raw_arguments_or_results_published"
        ],
    }
    report: dict[str, Any] = {
        "control_version": MCP_SDK_STDIO_CONTROL_VERSION,
        "checked_at": MCP_SDK_STDIO_CHECKED_AT,
        "runtime": {
            "sdk_distribution": "mcp",
            "sdk_version": sdk_version,
            "latest_protocol": mcp_types.LATEST_PROTOCOL_VERSION,
            "supported_protocols": list(SUPPORTED_PROTOCOL_VERSIONS),
        },
        "transport": {
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
            "closed_output_schema": output_schema.get("additionalProperties")
            is False,
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
        "server_receipt": {
            "handler_events": receipt["handler_events"],
            "recognized_handler_calls": receipt["recognized_handler_calls"],
            "total_handler_calls": receipt["total_handler_calls"],
            "server_run_completed": receipt["server_run_completed"],
            "raw_arguments_or_results_published": receipt[
                "raw_arguments_or_results_published"
            ],
            "receipt_fingerprint": artifact_fingerprint(receipt_projection),
        },
        "scope": {
            "official_sdk_client_executed": True,
            "official_sdk_server_executed": True,
            "official_sdk_stdio_client_executed": True,
            "official_sdk_stdio_server_executed": True,
            "real_subprocess_and_os_pipes_executed": True,
            "mcp_2025_11_25_negotiated": True,
            "official_generated_types_executed": True,
            "sdk_json_schema_validation_executed": True,
            "application_unknown_tool_gate_executed": True,
            "malformed_raw_framing_controls_executed": False,
            "streamable_http_transport_executed": False,
            "remote_or_cross_vendor_interop_proven": False,
            "official_conformance_suite_executed": False,
            "authentication_or_authorization_proven": False,
            "production_readiness_proven": False,
        },
        "evidence_boundary": MCP_SDK_STDIO_EVIDENCE_BOUNDARY,
    }
    report["report_fingerprint"] = artifact_fingerprint(report)
    return report


def run_mcp_sdk_stdio_control() -> dict[str, Any]:
    """Run the official-SDK subprocess/stdio control and verify its report."""

    report = anyio.run(_run_mcp_sdk_stdio_control)
    return verify_mcp_sdk_stdio_report(report)


def verify_mcp_sdk_stdio_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the closed report and its deliberately narrow evidence scope."""

    _exact(report, _TOP_LEVEL_FIELDS, "report")
    nested_fields = (
        ("runtime", _RUNTIME_FIELDS),
        ("transport", _TRANSPORT_FIELDS),
        ("initialization", _INITIALIZATION_FIELDS),
        ("discovery", _DISCOVERY_FIELDS),
        ("calls", _CALL_FIELDS),
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
        "receipt_version": MCP_SDK_STDIO_RECEIPT_VERSION,
        "sdk_version": MCP_SDK_REVIEWED_VERSION,
        "protocol_version": MCP_SDK_PROTOCOL_VERSION,
        "handler_events": [ADD_INPUT_CONTRACT.name, "fixture.missing"],
        "recognized_handler_calls": 1,
        "total_handler_calls": 2,
        "server_run_completed": True,
        "raw_arguments_or_results_published": False,
    }
    if (
        report.get("control_version") != MCP_SDK_STDIO_CONTROL_VERSION
        or report.get("checked_at") != MCP_SDK_STDIO_CHECKED_AT
        or report.get("evidence_boundary") != MCP_SDK_STDIO_EVIDENCE_BOUNDARY
        or nested["runtime"]
        != {
            "sdk_distribution": "mcp",
            "sdk_version": MCP_SDK_REVIEWED_VERSION,
            "latest_protocol": MCP_SDK_PROTOCOL_VERSION,
            "supported_protocols": list(SUPPORTED_PROTOCOL_VERSIONS),
        }
        or nested["transport"]
        != {
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
        or nested["initialization"]
        != {
            "protocol_version": MCP_SDK_PROTOCOL_VERSION,
            "server_name": "about-llm-mcp-sdk-stdio",
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
        or nested["server_receipt"]
        != {
            "handler_events": [ADD_INPUT_CONTRACT.name, "fixture.missing"],
            "recognized_handler_calls": 1,
            "total_handler_calls": 2,
            "server_run_completed": True,
            "raw_arguments_or_results_published": False,
            "receipt_fingerprint": artifact_fingerprint(receipt_projection),
        }
        or nested["scope"]
        != {
            "official_sdk_client_executed": True,
            "official_sdk_server_executed": True,
            "official_sdk_stdio_client_executed": True,
            "official_sdk_stdio_server_executed": True,
            "real_subprocess_and_os_pipes_executed": True,
            "mcp_2025_11_25_negotiated": True,
            "official_generated_types_executed": True,
            "sdk_json_schema_validation_executed": True,
            "application_unknown_tool_gate_executed": True,
            "malformed_raw_framing_controls_executed": False,
            "streamable_http_transport_executed": False,
            "remote_or_cross_vendor_interop_proven": False,
            "official_conformance_suite_executed": False,
            "authentication_or_authorization_proven": False,
            "production_readiness_proven": False,
        }
    ):
        raise ValueError("official MCP SDK stdio report semantic drift")
    return copy.deepcopy(dict(report))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or serve the official MCP SDK real-stdio control."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("control", help="run the client/subprocess control")
    serve_parser = subparsers.add_parser("serve", help="serve one SDK stdio session")
    serve_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "serve":
        logging.disable(logging.CRITICAL)
        anyio.run(_serve_sdk_stdio, args.receipt)
        return 0
    if args.command not in {None, "control"}:
        raise ValueError("unknown command")  # pragma: no cover - argparse owns this
    print(
        json.dumps(
            run_mcp_sdk_stdio_control(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
