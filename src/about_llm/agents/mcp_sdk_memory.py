"""Official MCP SDK in-memory client/server interoperability control.

This control complements the repository's authored stdio and Streamable HTTP
implementations.  It uses the official Python SDK on both sides of an in-memory
transport, negotiates MCP 2025-11-25, lists one explicitly described tool, and
exercises success, schema rejection, and unknown-tool rejection.

It is intentionally not a transport, conformance, authentication, authorization,
or cross-vendor test.  In particular, the low-level SDK invokes the application
handler for an unknown tool when no cached schema exists; the handler must still
fail closed.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any, Final, cast

import anyio
import mcp.types as mcp_types
from mcp import ClientSession
from mcp.server import Server
from mcp.shared.memory import create_client_server_memory_streams
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

from about_llm.agents.mcp_stdio import ADD_INPUT_CONTRACT, ADD_OUTPUT_CONTRACT
from about_llm.agents.schema import ToolArgumentValidationError
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

MCP_SDK_MEMORY_CONTROL_VERSION: Final = "about-llm.mcp-sdk-memory-control.v1"
MCP_SDK_REVIEWED_VERSION: Final = "1.29.0"
MCP_SDK_PROTOCOL_VERSION: Final = "2025-11-25"
MCP_SDK_MEMORY_CHECKED_AT: Final = "2026-08-14"
MCP_SDK_MEMORY_EVIDENCE_BOUNDARY: Final = (
    "This control uses the official mcp 1.29.0 ClientSession, low-level Server, "
    "generated types, JSON-Schema validation, and AnyIO in-memory streams for MCP "
    "2025-11-25. It does not execute stdio, TCP/HTTP, SSE, session resumption, "
    "OAuth, TLS, remote or cross-vendor interoperability, the official conformance "
    "suite, authorization, approval, side effects, or production logging. The SDK "
    "calls the application handler for an unlisted tool when no cached schema exists, "
    "so the application still rejects unknown names. Error content can contain "
    "validation detail and is excluded from the public report. The unkeyed report "
    "fingerprint does not authenticate execution or source."
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
        "scope",
        "evidence_boundary",
        "report_fingerprint",
    }
)
_RUNTIME_FIELDS: Final = frozenset(
    {"sdk_distribution", "sdk_version", "latest_protocol", "supported_protocols"}
)
_TRANSPORT_FIELDS: Final = frozenset(
    {"kind", "official_sdk_memory_stream", "os_stdio", "tcp_http", "subprocess"}
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
        "recognized_handler_calls",
        "total_handler_calls",
        "raw_error_content_published",
    }
)
_SCOPE_FIELDS: Final = frozenset(
    {
        "official_sdk_client_executed",
        "official_sdk_server_executed",
        "mcp_2025_11_25_negotiated",
        "official_generated_types_executed",
        "sdk_json_schema_validation_executed",
        "application_unknown_tool_gate_executed",
        "stdio_transport_executed",
        "streamable_http_transport_executed",
        "remote_or_cross_vendor_interop_proven",
        "official_conformance_suite_executed",
        "authentication_or_authorization_proven",
        "production_readiness_proven",
    }
)


def _snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(canonical_json_bytes(value)))


def _exact(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{label} fields mismatch")


def _error_result(message: str) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=message)],
        isError=True,
    )


def _sanitized_contract_error(error: ToolArgumentValidationError) -> str:
    fields = [f"code={error.code}"]
    if error.keyword is not None:
        fields.append(f"keyword={error.keyword}")
    if error.instance_pointer:
        fields.append(f"instance_path={error.instance_pointer}")
    return "Tool arguments rejected (" + ", ".join(fields) + ")"


def build_mcp_sdk_fixture_server(
    *,
    handler_events: list[str] | None = None,
    server_name: str = "about-llm-mcp-sdk-memory",
) -> tuple[Server[Any], dict[str, int]]:
    """Build the transport-neutral official-SDK fixture server."""

    server: Server[Any] = Server(
        server_name,
        version="1.0.0",
        instructions=(
            "Fixture only: discovery is not authorization and results are untrusted."
        ),
    )
    counters = {"total": 0, "recognized": 0}
    input_schema = _snapshot(ADD_INPUT_CONTRACT.arguments_schema)
    output_schema = _snapshot(ADD_OUTPUT_CONTRACT.arguments_schema)

    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name=ADD_INPUT_CONTRACT.name,
                title="Fixture integer addition",
                description=ADD_INPUT_CONTRACT.description,
                inputSchema=input_schema,
                outputSchema=output_schema,
            )
        ]

    @server.call_tool(validate_input=True)  # type: ignore[untyped-decorator]
    async def call_tool(
        name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | mcp_types.CallToolResult:
        if handler_events is not None:
            handler_events.append(name)
        counters["total"] += 1
        if name != ADD_INPUT_CONTRACT.name:
            return _error_result("Unknown tool")
        counters["recognized"] += 1
        try:
            ADD_INPUT_CONTRACT.validate(arguments)
        except ToolArgumentValidationError as error:
            return _error_result(_sanitized_contract_error(error))
        result = {
            "sum": cast(int, arguments["a"]) + cast(int, arguments["b"])
        }
        try:
            ADD_OUTPUT_CONTRACT.validate(result)
        except ToolArgumentValidationError as error:  # pragma: no cover - invariant
            return _error_result(_sanitized_contract_error(error))
        return result

    return server, counters


def build_mcp_sdk_memory_server() -> tuple[Server[Any], dict[str, int]]:
    """Build the fixture server without exporting application handler events."""

    return build_mcp_sdk_fixture_server()


async def _run_mcp_sdk_memory_control() -> dict[str, Any]:
    sdk_version = importlib.metadata.version("mcp")
    if sdk_version != MCP_SDK_REVIEWED_VERSION:
        raise ValueError(
            f"mcp SDK version drift: expected {MCP_SDK_REVIEWED_VERSION}, got {sdk_version}"
        )
    if (
        mcp_types.LATEST_PROTOCOL_VERSION != MCP_SDK_PROTOCOL_VERSION
        or MCP_SDK_PROTOCOL_VERSION not in SUPPORTED_PROTOCOL_VERSIONS
    ):
        raise ValueError("mcp SDK protocol-version drift")

    server, counters = build_mcp_sdk_memory_server()
    sdk_loggers = ("mcp", "FastMCP")
    prior_levels = {name: logging.getLogger(name).level for name in sdk_loggers}
    for name in sdk_loggers:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    try:
        async with create_client_server_memory_streams() as (
            client_streams,
            server_streams,
        ):
            client_read, client_write = client_streams
            server_read, server_write = server_streams

            async def serve() -> None:
                await server.run(
                    server_read,
                    server_write,
                    server.create_initialization_options(),
                    raise_exceptions=False,
                )

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(serve)
                try:
                    async with ClientSession(
                        read_stream=client_read,
                        write_stream=client_write,
                        read_timeout_seconds=timedelta(seconds=5),
                        client_info=mcp_types.Implementation(
                            name="about-llm-sdk-memory-client", version="1.0.0"
                        ),
                    ) as session:
                        initialized = await session.initialize()
                        await session.send_ping()
                        listed = await session.list_tools()
                        if len(listed.tools) != 1:
                            raise ValueError("official SDK tool discovery count drift")
                        tool = listed.tools[0]
                        if tool.name != ADD_INPUT_CONTRACT.name:
                            raise ValueError("official SDK tool discovery identity drift")

                        before_success = dict(counters)
                        success = await session.call_tool(
                            ADD_INPUT_CONTRACT.name, {"a": 2, "b": 3}
                        )
                        after_success = dict(counters)
                        invalid = await session.call_tool(
                            ADD_INPUT_CONTRACT.name,
                            {"a": 2, "b": 3, "extra": 1},
                        )
                        after_invalid = dict(counters)
                        unknown = await session.call_tool("fixture.missing", {})
                        after_unknown = dict(counters)
                finally:
                    task_group.cancel_scope.cancel()
    finally:
        for name, level in prior_levels.items():
            logging.getLogger(name).setLevel(level)

    if before_success != {"total": 0, "recognized": 0}:
        raise ValueError("official SDK handler counters were not clean")
    if after_success != {"total": 1, "recognized": 1}:
        raise ValueError("official SDK successful call counter drift")
    if after_invalid != after_success:
        raise ValueError("SDK schema-invalid request entered application handler")
    if after_unknown != {"total": 2, "recognized": 1}:
        raise ValueError("application unknown-tool gate counter drift")
    if success.isError or success.structuredContent != {"sum": 5}:
        raise ValueError("official SDK successful structured result drift")
    if invalid.isError is not True or unknown.isError is not True:
        raise ValueError("official SDK negative control did not fail closed")

    capabilities = initialized.capabilities
    if initialized.protocolVersion != MCP_SDK_PROTOCOL_VERSION:
        raise ValueError("official SDK negotiated protocol drift")
    if capabilities.tools is None:
        raise ValueError("official SDK did not negotiate tools capability")

    input_schema = tool.inputSchema
    output_schema = cast(dict[str, Any], tool.outputSchema)
    report: dict[str, Any] = {
        "control_version": MCP_SDK_MEMORY_CONTROL_VERSION,
        "checked_at": MCP_SDK_MEMORY_CHECKED_AT,
        "runtime": {
            "sdk_distribution": "mcp",
            "sdk_version": sdk_version,
            "latest_protocol": mcp_types.LATEST_PROTOCOL_VERSION,
            "supported_protocols": list(SUPPORTED_PROTOCOL_VERSIONS),
        },
        "transport": {
            "kind": "official_sdk_anyio_memory_object_stream",
            "official_sdk_memory_stream": True,
            "os_stdio": False,
            "tcp_http": False,
            "subprocess": False,
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
            "invalid_schema_handler_delta": after_invalid["total"]
            - after_success["total"],
            "unknown_tool_is_error": bool(unknown.isError),
            "unknown_tool_handler_delta": after_unknown["total"]
            - after_invalid["total"],
            "recognized_handler_calls": after_unknown["recognized"],
            "total_handler_calls": after_unknown["total"],
            "raw_error_content_published": False,
        },
        "scope": {
            "official_sdk_client_executed": True,
            "official_sdk_server_executed": True,
            "mcp_2025_11_25_negotiated": True,
            "official_generated_types_executed": True,
            "sdk_json_schema_validation_executed": True,
            "application_unknown_tool_gate_executed": True,
            "stdio_transport_executed": False,
            "streamable_http_transport_executed": False,
            "remote_or_cross_vendor_interop_proven": False,
            "official_conformance_suite_executed": False,
            "authentication_or_authorization_proven": False,
            "production_readiness_proven": False,
        },
        "evidence_boundary": MCP_SDK_MEMORY_EVIDENCE_BOUNDARY,
    }
    report["report_fingerprint"] = artifact_fingerprint(report)
    return report


def run_mcp_sdk_memory_control() -> dict[str, Any]:
    """Run the official-SDK in-memory control and verify its report."""

    report = anyio.run(_run_mcp_sdk_memory_control)
    return verify_mcp_sdk_memory_report(report)


def verify_mcp_sdk_memory_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the closed report and its reviewed semantic boundary."""

    _exact(report, _TOP_LEVEL_FIELDS, "report")
    nested_fields = (
        ("runtime", _RUNTIME_FIELDS),
        ("transport", _TRANSPORT_FIELDS),
        ("initialization", _INITIALIZATION_FIELDS),
        ("discovery", _DISCOVERY_FIELDS),
        ("calls", _CALL_FIELDS),
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

    runtime = nested["runtime"]
    transport = nested["transport"]
    initialization = nested["initialization"]
    discovery = nested["discovery"]
    calls = nested["calls"]
    scope = nested["scope"]
    if (
        report.get("control_version") != MCP_SDK_MEMORY_CONTROL_VERSION
        or report.get("checked_at") != MCP_SDK_MEMORY_CHECKED_AT
        or report.get("evidence_boundary") != MCP_SDK_MEMORY_EVIDENCE_BOUNDARY
        or runtime.get("sdk_distribution") != "mcp"
        or runtime.get("sdk_version") != MCP_SDK_REVIEWED_VERSION
        or runtime.get("latest_protocol") != MCP_SDK_PROTOCOL_VERSION
        or runtime.get("supported_protocols") != list(SUPPORTED_PROTOCOL_VERSIONS)
        or initialization.get("protocol_version") != MCP_SDK_PROTOCOL_VERSION
        or initialization.get("server_name") != "about-llm-mcp-sdk-memory"
        or initialization.get("server_version") != "1.0.0"
        or initialization.get("tools_capability") is not True
        or transport
        != {
            "kind": "official_sdk_anyio_memory_object_stream",
            "official_sdk_memory_stream": True,
            "os_stdio": False,
            "tcp_http": False,
            "subprocess": False,
        }
        or discovery.get("tool_count") != 1
        or discovery.get("tool_name") != ADD_INPUT_CONTRACT.name
        or discovery.get("input_schema_fingerprint")
        != artifact_fingerprint(_snapshot(ADD_INPUT_CONTRACT.arguments_schema))
        or discovery.get("output_schema_fingerprint")
        != artifact_fingerprint(_snapshot(ADD_OUTPUT_CONTRACT.arguments_schema))
        or discovery.get("closed_input_schema") is not True
        or discovery.get("closed_output_schema") is not True
        or calls
        != {
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
        or scope
        != {
            "official_sdk_client_executed": True,
            "official_sdk_server_executed": True,
            "mcp_2025_11_25_negotiated": True,
            "official_generated_types_executed": True,
            "sdk_json_schema_validation_executed": True,
            "application_unknown_tool_gate_executed": True,
            "stdio_transport_executed": False,
            "streamable_http_transport_executed": False,
            "remote_or_cross_vendor_interop_proven": False,
            "official_conformance_suite_executed": False,
            "authentication_or_authorization_proven": False,
            "production_readiness_proven": False,
        }
    ):
        raise ValueError("official MCP SDK report semantic drift")
    return copy.deepcopy(dict(report))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the official MCP SDK in-memory interoperability control."
    )
    parser.parse_args(argv)
    print(
        json.dumps(
            run_mcp_sdk_memory_control(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
