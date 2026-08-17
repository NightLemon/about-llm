"""A2A 1.0 loopback interoperability control using the official Python SDK.

The control intentionally exercises a narrow, auditable surface:

* an official-SDK server subprocess bound to IPv4 loopback;
* Agent Card discovery at the well-known URI;
* the A2A 1.0 JSON-RPC-over-HTTP binding;
* ``SendMessage`` followed by ``GetTask``;
* official generated protobuf parsing and required-field validation;
* rejection of an unsupported protocol version and a legacy ``kind`` field;
* an optional, hash-pinned validation against the official v1.0.0 JSON Schema.

It is not a conformance suite, authentication test, remote interoperability
test, authorization system, or proof that a remote ``completed`` state is true.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import logging
import re
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Final, NoReturn, cast

import httpx
import uvicorn
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Artifact,
    GetTaskRequest,
    SendMessageRequest,
    Task,
    TaskState,
    TaskStatus,
)
from a2a.utils.constants import TransportProtocol
from a2a.utils.errors import UnsupportedOperationError
from google.protobuf import json_format  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from starlette.applications import Starlette

A2A_CONTROL_VERSION: Final = "about-llm.a2a-loopback-control.v1"
A2A_PROTOCOL_VERSION: Final = "1.0"
A2A_SDK_DISTRIBUTION: Final = "a2a-sdk"
A2A_SDK_REVIEWED_VERSION: Final = "1.1.2"
A2A_RPC_PATH: Final = "/rpc"
A2A_CARD_PATH: Final = "/.well-known/agent-card.json"
A2A_SCHEMA_URL: Final = "https://a2a-protocol.org/v1.0.0/spec/a2a.json"
A2A_SCHEMA_SHA256: Final = (
    "sha256:6b6560c726289734799b7d5883be84e4cc0452600736db0f811341bac43b8d62"
)
MAX_SCHEMA_BYTES: Final = 1_000_000
MAX_ERROR_BODY_BYTES: Final = 65_536
SERVER_START_TIMEOUT_SECONDS: Final = 10.0
REQUEST_TIMEOUT_SECONDS: Final = 5.0


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _definition_key(value: str) -> str:
    basename = value.rsplit("/", maxsplit=1)[-1]
    basename = re.sub(r"^(?:lf\.a2a\.v1\.|google\.protobuf\.)", "", basename)
    basename = re.sub(r"\.jsonschema\.json$", "", basename)
    return re.sub(r"[^a-z0-9]", "", basename.lower())


def bundle_official_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the official multi-file-style definitions into one local bundle.

    The published ``a2a.json`` document stores schemas in ``definitions`` but
    retains generated relative references such as
    ``lf.a2a.v1.Message.jsonschema.json``. Rewriting those references to local
    JSON Pointers lets Draft 2020-12 validators consume the downloaded document
    without making hidden follow-up network requests.
    """

    definitions_value = schema.get("definitions")
    if not isinstance(definitions_value, dict) or not definitions_value:
        raise ValueError("official schema has no definitions object")
    definitions = copy.deepcopy(definitions_value)
    name_by_key: dict[str, str] = {}
    for name in definitions:
        if not isinstance(name, str):
            raise ValueError("official schema definition name is not a string")
        key = _definition_key(name)
        if not key or key in name_by_key:
            raise ValueError("official schema definition names are ambiguous")
        name_by_key[key] = name

    def rewrite(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "$ref" and isinstance(child, str) and not child.startswith("#"):
                    target = name_by_key.get(_definition_key(child))
                    if target is None:
                        raise ValueError("official schema contains an unresolved reference")
                    value[key] = f"#/definitions/{_json_pointer_token(target)}"
                else:
                    rewrite(child)
        elif isinstance(value, list):
            for child in value:
                rewrite(child)

    rewrite(definitions)
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError("official schema does not declare Draft 2020-12")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "definitions": definitions,
    }


def validate_official_instances(
    schema: Mapping[str, Any], instances: Mapping[str, Mapping[str, Any]]
) -> None:
    """Validate named instances against matching official schema definitions."""

    bundled = bundle_official_schema(schema)
    definitions = cast(dict[str, Any], bundled["definitions"])
    for name, instance in instances.items():
        if name not in definitions:
            raise ValueError(f"official schema definition is missing: {name}")
        wrapper = {
            **bundled,
            "$ref": f"#/definitions/{_json_pointer_token(name)}",
        }
        errors = sorted(
            Draft202012Validator(wrapper).iter_errors(instance),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            error = errors[0]
            path = "/" + "/".join(str(part) for part in error.absolute_path)
            raise ValueError(
                f"official schema rejected {name} at {path or '/'} "
                f"with validator {error.validator}"
            )


async def fetch_official_schema(*, timeout_seconds: float = 10.0) -> dict[str, Any]:
    """Fetch and hash-check the frozen official A2A 1.0.0 JSON Schema."""

    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        trust_env=False,
    ) as client:
        response = await client.get(
            A2A_SCHEMA_URL,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        raw = response.content
    if len(raw) > MAX_SCHEMA_BYTES:
        raise ValueError("official schema exceeds byte limit")
    actual_fingerprint = _sha256_bytes(raw)
    if actual_fingerprint != A2A_SCHEMA_SHA256:
        raise ValueError("official schema fingerprint mismatch")
    schema = _strict_json_object(raw)
    if schema.get("title") != "A2A Protocol Schemas" or schema.get("version") != "v1":
        raise ValueError("official schema metadata mismatch")
    bundle_official_schema(schema)
    return schema


def build_agent_card(endpoint: str) -> AgentCard:
    """Build the deterministic public Agent Card served by the control."""

    return AgentCard(
        name="about-llm fixture agent",
        description="Local deterministic A2A protocol control.",
        supported_interfaces=[
            AgentInterface(
                url=endpoint,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=A2A_PROTOCOL_VERSION,
            )
        ],
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=False,
            push_notifications=False,
            extended_agent_card=False,
        ),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="fixture.add",
                name="Fixture add",
                description="Adds two numeric fields in a structured fixture.",
                tags=["fixture", "structured-data"],
                input_modes=["application/json"],
                output_modes=["application/json"],
            )
        ],
    )


def build_send_request() -> SendMessageRequest:
    """Create the valid structured request used by the positive control."""

    request = SendMessageRequest()
    json_format.ParseDict(
        {
            "message": {
                "messageId": "control-message-1",
                "role": "ROLE_USER",
                "parts": [
                    {
                        "data": {"a": 7, "b": 5},
                        "mediaType": "application/json",
                    }
                ],
            },
            "configuration": {
                "acceptedOutputModes": ["application/json"],
                "historyLength": 1,
            },
        },
        request,
    )
    return request


def _structured_part(part: Any) -> dict[str, Any]:
    if not part.HasField("data"):
        raise ValueError("fixture requires a structured data part")
    value = json_format.MessageToDict(part.data)
    if not isinstance(value, dict):
        raise ValueError("fixture data part must contain an object")
    return value


class FixtureAddExecutor(AgentExecutor):
    """Deterministic official-SDK executor for the local protocol control."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message = context.message
        if message is None or len(message.parts) != 1:
            raise ValueError("fixture requires exactly one message part")
        values = _structured_part(message.parts[0])
        if set(values) != {"a", "b"}:
            raise ValueError("fixture requires exactly two allowed fields")
        a = values["a"]
        b = values["b"]
        if (
            not isinstance(a, int | float)
            or isinstance(a, bool)
            or not isinstance(b, int | float)
            or isinstance(b, bool)
        ):
            raise ValueError("fixture fields must be finite numbers")
        total = a + b
        if not isinstance(total, int | float) or total != total or abs(total) == float("inf"):
            raise ValueError("fixture result must be finite")

        artifact = Artifact()
        json_format.ParseDict(
            {
                "artifactId": "fixture-sum",
                "name": "sum",
                "parts": [
                    {
                        "data": {"sum": total},
                        "mediaType": "application/json",
                    }
                ],
            },
            artifact,
        )
        task_id = context.task_id
        context_id = context.context_id
        if not task_id or not context_id:
            raise ValueError("official request context did not assign identifiers")
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                artifacts=[artifact],
                history=[message],
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        del context, event_queue
        raise UnsupportedOperationError(message="fixture cancellation is unsupported")


def build_server_app(host: str, port: int) -> Starlette:
    """Create the official-SDK Starlette server application."""

    endpoint = f"http://{host}:{port}{A2A_RPC_PATH}"
    card = build_agent_card(endpoint)
    handler = DefaultRequestHandler(
        agent_executor=FixtureAddExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = [
        *create_agent_card_routes(card, card_url=A2A_CARD_PATH),
        *create_jsonrpc_routes(handler, rpc_url=A2A_RPC_PATH),
    ]
    return Starlette(routes=routes)


def serve(host: str, port: int) -> int:
    """Run the loopback server until the parent process terminates it."""

    if host != "127.0.0.1":
        raise ValueError("the learning control only permits IPv4 loopback")
    if not 0 < port < 65_536:
        raise ValueError("port must be between 1 and 65535")
    # The child process is a protocol endpoint. Expected negative controls must
    # not leak request validation tracebacks or library logs to stderr.
    logging.disable(logging.CRITICAL)
    config = uvicorn.Config(
        build_server_app(host, port),
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
    client: httpx.AsyncClient,
    base_url: str,
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("A2A server exited before readiness")
        try:
            response = await client.get(f"{base_url}{A2A_CARD_PATH}")
            if response.status_code == 200:
                return
        except httpx.RequestError:
            pass
        await asyncio.sleep(0.05)
    raise TimeoutError("A2A server readiness timed out")


def _error_code(response: httpx.Response) -> int:
    raw = response.content
    if len(raw) > MAX_ERROR_BODY_BYTES:
        raise ValueError("A2A error response exceeds byte limit")
    body = _strict_json_object(raw)
    error = body.get("error")
    if not isinstance(error, dict):
        raise ValueError("A2A response has no JSON-RPC error object")
    code = error.get("code")
    if not isinstance(code, int) or isinstance(code, bool):
        raise ValueError("A2A error code is not an integer")
    return code


def _verify_task(task: Task) -> bool:
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        return False
    if not task.id or not task.context_id or len(task.artifacts) != 1:
        return False
    artifact = task.artifacts[0]
    if artifact.artifact_id != "fixture-sum" or len(artifact.parts) != 1:
        return False
    try:
        output = _structured_part(artifact.parts[0])
    except ValueError:
        return False
    return set(output) == {"sum"} and output["sum"] == 12


def _projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the content-free allowlist projection used for fingerprinting."""

    agent_card = cast(Mapping[str, Any], report["agent_card"])
    operations = cast(Sequence[Mapping[str, Any]], report["operations"])
    task = cast(Mapping[str, Any], report["task"])
    negative = cast(Mapping[str, Any], report["negative_controls"])
    schema = cast(Mapping[str, Any], report["official_schema"])
    return {
        "implementation": report["implementation"],
        "protocol_version": report["protocol_version"],
        "binding": report["binding"],
        "network": report["network"],
        "agent_card": {
            "well_known_resolved": agent_card["well_known_resolved"],
            "sdk_parsed": agent_card["sdk_parsed"],
            "interface_binding": agent_card["interface_binding"],
            "interface_protocol_version": agent_card["interface_protocol_version"],
            "signed": agent_card["signed"],
        },
        "operation_methods": [operation["method"] for operation in operations],
        "task": {
            "remote_state": task["remote_state"],
            "artifact_count": task["artifact_count"],
            "local_verifier_passed": task["local_verifier_passed"],
        },
        "negative_error_codes": {
            "legacy_kind": negative["legacy_kind_error_code"],
            "unsupported_version": negative["unsupported_version_error_code"],
        },
        "official_schema_validated": schema["validated"],
    }


def _terminate_process(process: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    if process.poll() is None:
        process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5.0)
    return stdout, stderr


async def _run_loopback_control_async(
    *,
    verify_official_schema: bool,
    schema_timeout_seconds: float,
) -> dict[str, Any]:
    host = "127.0.0.1"
    port = _reserve_candidate_port()
    base_url = f"http://{host}:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "about_llm.agents.a2a_loopback",
            "serve",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = b""
    stderr = b""
    try:
        timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as http_client:
            await _wait_until_ready(
                http_client,
                base_url,
                process,
                timeout_seconds=SERVER_START_TIMEOUT_SECONDS,
            )
            resolver = A2ACardResolver(http_client, base_url)
            card = await resolver.get_agent_card()
            if len(card.supported_interfaces) != 1:
                raise ValueError("fixture Agent Card must publish one interface")
            interface = card.supported_interfaces[0]
            if (
                interface.protocol_binding != TransportProtocol.JSONRPC.value
                or interface.protocol_version != A2A_PROTOCOL_VERSION
                or interface.url != f"{base_url}{A2A_RPC_PATH}"
            ):
                raise ValueError("fixture Agent Card interface mismatch")

            factory = ClientFactory(
                ClientConfig(
                    streaming=False,
                    httpx_client=http_client,
                    supported_protocol_bindings=[TransportProtocol.JSONRPC.value],
                )
            )
            client = factory.create(card)
            send_request = build_send_request()
            events = [event async for event in client.send_message(send_request)]
            if len(events) != 1 or not events[0].HasField("task"):
                raise ValueError("SendMessage did not return exactly one task")
            sent_task = events[0].task
            fetched_task = await client.get_task(
                GetTaskRequest(id=sent_task.id, history_length=1)
            )
            if sent_task.id != fetched_task.id:
                raise ValueError("GetTask returned a different task")
            local_verifier_passed = _verify_task(fetched_task)
            if not local_verifier_passed:
                raise ValueError("local task verifier rejected the remote result")

            valid_params = json_format.MessageToDict(send_request)
            legacy_params = copy.deepcopy(valid_params)
            legacy_message = cast(dict[str, Any], legacy_params["message"])
            legacy_parts = cast(list[dict[str, Any]], legacy_message["parts"])
            legacy_parts[0]["kind"] = "data"
            legacy_response = await http_client.post(
                interface.url,
                headers={"A2A-Version": A2A_PROTOCOL_VERSION},
                json={
                    "jsonrpc": "2.0",
                    "id": "legacy-kind-control",
                    "method": "SendMessage",
                    "params": legacy_params,
                },
            )
            unsupported_version_response = await http_client.post(
                interface.url,
                headers={"A2A-Version": "9.9"},
                json={
                    "jsonrpc": "2.0",
                    "id": "unsupported-version-control",
                    "method": "SendMessage",
                    "params": valid_params,
                },
            )
            legacy_kind_error_code = _error_code(legacy_response)
            unsupported_version_error_code = _error_code(unsupported_version_response)
            if legacy_kind_error_code != -32602:
                raise ValueError("legacy kind field was not rejected as invalid params")
            if unsupported_version_error_code != -32009:
                raise ValueError("unsupported version did not return VersionNotSupportedError")

            schema: dict[str, Any] | None = None
            if verify_official_schema:
                schema = await fetch_official_schema(timeout_seconds=schema_timeout_seconds)
                validate_official_instances(
                    schema,
                    {
                        "Agent Card": json_format.MessageToDict(card),
                        "Send Message Request": valid_params,
                        "Task": json_format.MessageToDict(fetched_task),
                    },
                )

        if process.poll() is not None:
            raise RuntimeError("A2A server exited during control")
    finally:
        stdout, stderr = _terminate_process(process)

    sdk_version = importlib.metadata.version(A2A_SDK_DISTRIBUTION)
    report: dict[str, Any] = {
        "implementation": A2A_CONTROL_VERSION,
        "protocol_version": A2A_PROTOCOL_VERSION,
        "official_sdk": {
            "distribution": A2A_SDK_DISTRIBUTION,
            "runtime_version": sdk_version,
            "reviewed_version": A2A_SDK_REVIEWED_VERSION,
            "client_used": True,
            "server_used": True,
            "generated_proto_models_validated": True,
        },
        "binding": "JSONRPC",
        "network": {
            "scheme": "http",
            "address_scope": "IPv4 loopback",
            "real_tcp_http": True,
            "tls": False,
        },
        "agent_card": {
            "path": A2A_CARD_PATH,
            "well_known_resolved": True,
            "sdk_parsed": True,
            "interface_binding": interface.protocol_binding,
            "interface_protocol_version": interface.protocol_version,
            "signed": bool(card.signatures),
            "signature_verified": False,
        },
        "operations": [
            {"method": "SendMessage", "sdk_client_executed": True},
            {"method": "GetTask", "sdk_client_executed": True},
        ],
        "task": {
            "remote_state": "TASK_STATE_COMPLETED",
            "artifact_count": len(fetched_task.artifacts),
            "history_count": len(fetched_task.history),
            "local_verifier_passed": local_verifier_passed,
            "remote_completed_treated_as_sufficient": False,
        },
        "negative_controls": {
            "legacy_kind_error_code": legacy_kind_error_code,
            "unsupported_version_error_code": unsupported_version_error_code,
            "raw_error_data_published": False,
        },
        "official_schema": {
            "url": A2A_SCHEMA_URL,
            "expected_fingerprint": A2A_SCHEMA_SHA256,
            "validated": schema is not None,
            "definitions_validated": (
                ["Agent Card", "Send Message Request", "Task"] if schema is not None else []
            ),
        },
        "server_process": {
            "subprocess_used": True,
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_stderr_empty": not stdout and not stderr,
        },
        "evidence_limits": {
            "complete_a2a_conformance_proven": False,
            "streaming_sse_proven": False,
            "http_json_rest_proven": False,
            "grpc_proven": False,
            "authentication_proven": False,
            "authorization_or_approval_proven": False,
            "signed_card_identity_proven": False,
            "remote_or_cross_vendor_interoperability_proven": False,
            "tls_proven": False,
            "production_safety_proven": False,
        },
    }
    projection = _projection(report)
    report["projection"] = projection
    report["projection_fingerprint"] = _sha256_bytes(_canonical_json(projection))
    report["raw_messages_published"] = False
    return report


def run_loopback_control(
    *,
    verify_official_schema: bool = False,
    schema_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run the real local A2A client/server control and return a public report."""

    if schema_timeout_seconds <= 0:
        raise ValueError("schema timeout must be positive")
    return asyncio.run(
        _run_loopback_control_async(
            verify_official_schema=verify_official_schema,
            schema_timeout_seconds=schema_timeout_seconds,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="run the local fixture server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", required=True, type=int)

    control_parser = subparsers.add_parser("control", help="run the loopback control")
    control_parser.add_argument(
        "--verify-official-schema",
        action="store_true",
        help="download the hash-pinned official v1.0.0 JSON Schema and validate fixtures",
    )
    control_parser.add_argument("--schema-timeout", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        return serve(args.host, args.port)
    report = run_loopback_control(
        verify_official_schema=args.verify_official_schema,
        schema_timeout_seconds=args.schema_timeout,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
