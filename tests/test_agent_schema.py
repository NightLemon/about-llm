from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from about_llm.agents import (
    DRAFT_2020_12_URI,
    AgentRuntime,
    CapabilityPolicy,
    ExecutionContext,
    JSONSchemaToolContract,
    ResourceRef,
    SideEffect,
    ToolArgumentValidationError,
    ToolCall,
    ToolRegistry,
)


def object_schema(
    properties: Mapping[str, Any],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "$schema": DRAFT_2020_12_URI,
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def contract(
    schema: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> JSONSchemaToolContract:
    return JSONSchemaToolContract(
        "fixture_tool",
        "Validate and return one local fixture value.",
        "fixture-tool-arguments@v1",
        object_schema({"key": {"type": "string", "minLength": 1}}, required=("key",))
        if schema is None
        else schema,
        **kwargs,
    )


def test_contract_validates_draft_2020_12_and_redacts_rejected_values() -> None:
    typed = contract(
        object_schema({"key": {"type": "integer"}}, required=("key",))
    )

    typed.validate({"key": 7})
    with pytest.raises(ToolArgumentValidationError) as captured:
        typed.validate({"key": "TOP-SECRET-VALUE"})

    error = captured.value
    assert error.code == "schema_violation"
    assert error.schema_revision == "fixture-tool-arguments@v1"
    assert error.keyword == "type"
    assert error.instance_pointer == "/key"
    assert "TOP-SECRET-VALUE" not in str(error)
    assert "integer" not in str(error)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"type": "object", "additionalProperties": False}, "Draft 2020-12"),
        (
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "additionalProperties": False,
            },
            "Draft 2020-12",
        ),
        (
            {
                "$schema": DRAFT_2020_12_URI,
                "type": "array",
                "additionalProperties": False,
            },
            "root type",
        ),
        (
            {"$schema": DRAFT_2020_12_URI, "type": "object"},
            "reject unevaluated fields",
        ),
        (
            {
                "$schema": DRAFT_2020_12_URI,
                "$id": "https://example.invalid/schema",
                "type": "object",
                "additionalProperties": False,
            },
            "must not contain \\$id",
        ),
        (
            object_schema({"value": {"$ref": "https://example.invalid/value"}}),
            "local fragment",
        ),
        (
            object_schema({"value": {"type": "not-a-real-type"}}),
            "invalid Draft 2020-12",
        ),
    ],
)
def test_contract_rejects_ambiguous_or_externally_resolved_schemas(
    schema: Mapping[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        contract(schema)


def test_local_definitions_are_supported_without_remote_resolution() -> None:
    schema = object_schema(
        {"key": {"$ref": "#/$defs/nonempty"}},
        required=("key",),
    )
    schema["$defs"] = {"nonempty": {"type": "string", "minLength": 1}}
    typed = contract(schema)

    typed.validate({"key": "ok"})
    with pytest.raises(ToolArgumentValidationError, match="minLength"):
        typed.validate({"key": ""})


def test_format_is_annotation_by_default_and_explicitly_enforceable() -> None:
    schema = object_schema(
        {"address": {"type": "string", "format": "email"}},
        required=("address",),
    )
    annotation_only = contract(schema)
    enforced = contract(schema, enforce_formats=True)

    annotation_only.validate({"address": "not an email"})
    with pytest.raises(ToolArgumentValidationError, match="format"):
        enforced.validate({"address": "not an email"})
    enforced.validate({"address": "student@example.com"})


def test_enforced_unknown_format_fails_at_contract_construction() -> None:
    schema = object_schema({"value": {"type": "string", "format": "custom-id"}})

    annotation_only = contract(schema)
    annotation_only.validate({"value": "anything"})
    with pytest.raises(ValueError, match="unsupported format"):
        contract(schema, enforce_formats=True)


def test_schema_snapshot_is_detached_deeply_frozen_and_planner_visible() -> None:
    source = object_schema(
        {"key": {"type": "string", "minLength": 1}}, required=("key",)
    )
    typed = contract(source)
    source["properties"]["key"]["type"] = "integer"  # type: ignore[index]

    planner = typed.planner_contract()
    disclosed = planner.to_dict()
    assert disclosed["schema_revision"] == "fixture-tool-arguments@v1"
    assert disclosed["validator_revision"] == typed.validator_revision
    assert disclosed["arguments_schema"]["properties"]["key"]["type"] == "string"
    with pytest.raises(TypeError):
        cast(Any, typed.arguments_schema["properties"])["new"] = True
    typed.validate({"key": "still-string"})


def test_schema_fingerprint_binds_schema_revision_validator_options_and_limit() -> None:
    base = contract()
    revision = JSONSchemaToolContract(
        base.name,
        base.description,
        "fixture-tool-arguments@v2",
        base.arguments_schema,
    )
    limit = contract(max_instance_bytes=63_999)
    format_mode = contract(enforce_formats=True)
    schema = contract(
        object_schema({"key": {"type": "string", "maxLength": 3}})
    )

    assert len(
        {
            base.schema_fingerprint,
            revision.schema_fingerprint,
            limit.schema_fingerprint,
            format_mode.schema_fingerprint,
            schema.schema_fingerprint,
        }
    ) == 5


def test_instance_byte_limit_is_checked_before_schema_evaluation() -> None:
    typed = contract(max_instance_bytes=20)

    with pytest.raises(ToolArgumentValidationError) as captured:
        typed.validate({"key": "x" * 100})

    assert captured.value.code == "instance_too_large"
    assert "x" * 20 not in str(captured.value)


def test_schema_byte_limit_is_checked_before_compilation() -> None:
    schema = object_schema({"key": {"type": "string"}})
    encoded_size = len(str(schema).encode("utf-8"))

    with pytest.raises(ValueError, match="max_schema_bytes"):
        contract(schema, max_schema_bytes=max(1, encoded_size // 2))


def test_built_tool_uses_schema_before_resource_policy_and_handler() -> None:
    typed = contract(
        object_schema(
            {"key": {"const": "allowed", "type": "string"}},
            required=("key",),
        )
    )
    calls: list[str] = []

    def resolve(arguments: Mapping[str, Any]) -> ResourceRef:
        calls.append("resolve")
        return ResourceRef("tenant-a", "fixture", str(arguments["key"]), "fixture@v1")

    def handler(arguments: Mapping[str, Any]) -> dict[str, Any]:
        calls.append("handler")
        return {"key": arguments["key"]}

    tool = typed.build_tool(
        tool_version="fixture-tool@v1",
        side_effect=SideEffect.READ_ONLY,
        handler=handler,
        required_capability="fixture:read",
        resolve_resource=resolve,
    )
    runtime = AgentRuntime(
        ToolRegistry([tool]),
        policy=CapabilityPolicy("fixture-policy@v1"),
    )
    context = ExecutionContext(
        "task", "user", "tenant-a", frozenset({"fixture:read"})
    )

    with pytest.raises(ToolArgumentValidationError, match="const"):
        runtime.execute(
            ToolCall("invalid-call", "fixture_tool", {"key": "denied"}),
            context=context,
        )
    assert calls == []

    outcome = runtime.execute(
        ToolCall("valid-call", "fixture_tool", {"key": "allowed"}),
        context=context,
    )
    assert outcome.value == {"key": "allowed"}
    assert calls == ["resolve", "handler"]
