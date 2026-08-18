from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from google.protobuf import json_format

from about_llm.agents.a2a_loopback import (
    A2A_CARD_PATH,
    A2A_CONTROL_VERSION,
    A2A_PROTOCOL_VERSION,
    A2A_SCHEMA_SHA256,
    A2A_SCHEMA_URL,
    FixtureAddExecutor,
    _strict_json_object,
    build_agent_card,
    build_send_request,
    bundle_official_schema,
    run_loopback_control,
    validate_official_instances,
)

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONTROL = ROOT / "projects" / "safe-agent" / "a2a_loopback_control.py"


def test_strict_json_object_rejects_ambiguous_or_invalid_values() -> None:
    assert _strict_json_object(b'{"ok":true}') == {"ok": True}
    invalid = [
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b"[]",
        b"\xff",
        b"{",
    ]
    for raw in invalid:
        with pytest.raises(ValueError):
            _strict_json_object(raw)


def test_agent_card_and_request_use_a2a_1_0_protojson() -> None:
    card = build_agent_card("http://127.0.0.1:1/rpc")
    card_json = json_format.MessageToDict(card)
    assert card_json["supportedInterfaces"] == [
        {
            "url": "http://127.0.0.1:1/rpc",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ]
    assert card_json["capabilities"] == {
        "streaming": False,
        "pushNotifications": False,
        "extendedAgentCard": False,
    }
    assert not card.signatures

    request_json = json_format.MessageToDict(build_send_request())
    part = request_json["message"]["parts"][0]
    assert part == {
        "data": {"a": 7.0, "b": 5.0},
        "mediaType": "application/json",
    }
    assert "kind" not in part
    assert request_json["message"]["role"] == "ROLE_USER"


def test_bundle_official_schema_rewrites_generated_relative_refs() -> None:
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "definitions": {
            "Message": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "properties": {"part": {"$ref": "lf.a2a.v1.Part.jsonschema.json"}},
                "required": ["part"],
            },
            "Part": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
    bundled = bundle_official_schema(schema)
    assert bundled["definitions"]["Message"]["properties"]["part"]["$ref"] == (
        "#/definitions/Part"
    )
    validate_official_instances(schema, {"Message": {"part": {"text": "ok"}}})
    with pytest.raises(ValueError, match="validator additionalProperties"):
        validate_official_instances(
            schema,
            {"Message": {"part": {"text": "ok", "kind": "text"}}},
        )


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "definitions": {}},
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "definitions": {
                "A B": {"$ref": "lf.a2a.v1.Missing.jsonschema.json"},
            },
        },
        {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "definitions": {"Message": {"type": "object"}},
        },
    ],
)
def test_bundle_official_schema_fails_closed(schema: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        bundle_official_schema(schema)


def test_fixture_executor_rejects_cancel() -> None:
    executor = FixtureAddExecutor()
    with pytest.raises(Exception, match="unsupported"):
        asyncio.run(executor.cancel(None, None))  # type: ignore[arg-type]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.extended
def test_project_control_executes_real_official_sdk_loopback() -> None:
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
    assert report["implementation"] == A2A_CONTROL_VERSION
    assert report["protocol_version"] == A2A_PROTOCOL_VERSION
    assert report["official_sdk"] == {
        "distribution": "a2a-sdk",
        "runtime_version": "1.1.2",
        "reviewed_version": "1.1.2",
        "client_used": True,
        "server_used": True,
        "generated_proto_models_validated": True,
    }
    assert report["network"] == {
        "scheme": "http",
        "address_scope": "IPv4 loopback",
        "real_tcp_http": True,
        "tls": False,
    }
    assert report["agent_card"] == {
        "path": A2A_CARD_PATH,
        "well_known_resolved": True,
        "sdk_parsed": True,
        "interface_binding": "JSONRPC",
        "interface_protocol_version": "1.0",
        "signed": False,
        "signature_verified": False,
    }
    assert [item["method"] for item in report["operations"]] == ["SendMessage", "GetTask"]
    assert report["task"] == {
        "remote_state": "TASK_STATE_COMPLETED",
        "artifact_count": 1,
        "history_count": 1,
        "local_verifier_passed": True,
        "remote_completed_treated_as_sufficient": False,
    }
    assert report["negative_controls"] == {
        "legacy_kind_error_code": -32602,
        "unsupported_version_error_code": -32009,
        "raw_error_data_published": False,
    }
    assert report["official_schema"] == {
        "url": A2A_SCHEMA_URL,
        "expected_fingerprint": A2A_SCHEMA_SHA256,
        "validated": False,
        "definitions_validated": [],
    }
    assert report["server_process"] == {
        "subprocess_used": True,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_stderr_empty": True,
    }
    assert report["projection_fingerprint"] == (
        "sha256:f1ad7ae1c0e18c91caa710d6448f6a503eaf8c8cbf0c0e689166d3f1af4b099e"
    )
    assert report["raw_messages_published"] is False
    assert all(value is False for value in report["evidence_limits"].values())
    serialized = json.dumps(report, ensure_ascii=False)
    assert "control-message-1" not in serialized
    assert '"a": 7' not in serialized
    assert '"sum": 12' not in serialized
def test_invalid_public_control_arguments() -> None:
    with pytest.raises(ValueError, match="timeout"):
        run_loopback_control(schema_timeout_seconds=0)
