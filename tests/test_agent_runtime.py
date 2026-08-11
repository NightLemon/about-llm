from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from about_llm.agents import (
    AgentRuntime,
    ApprovalGrant,
    CapabilityPolicy,
    ExecutionContext,
    ExecutionStatus,
    IdempotencyConflict,
    ResourceRef,
    SideEffect,
    Tool,
    ToolCall,
    ToolRegistry,
)

CONTEXT = ExecutionContext(
    "task-1", "user-1", "tenant-a", frozenset({"test:tool"})
)
POLICY = CapabilityPolicy("test-policy@v1")


def fixture_resource(_: Mapping[str, Any]) -> ResourceRef:
    return ResourceRef("tenant-a", "fixture", "resource-1", "fixture@v1")


def runtime_for(tool: Tool, *, max_tool_calls: int = 10) -> AgentRuntime:
    return AgentRuntime(
        ToolRegistry([tool]),
        max_tool_calls=max_tool_calls,
        policy=POLICY,
        clock=lambda: 100.0,
    )


def grant_for(outcome: Any, *, context: ExecutionContext = CONTEXT) -> ApprovalGrant:
    return ApprovalGrant(
        "approval-1",
        "approver-1",
        context.subject_id,
        context.task_id,
        outcome.call.call_id,
        outcome.execution_fingerprint,
        200.0,
    )


def validate_message(arguments: Mapping[str, Any]) -> None:
    if set(arguments) != {"message"} or not isinstance(arguments["message"], str):
        raise ValueError("expected one string field: message")


def test_side_effect_requires_approval_and_duplicate_is_not_reexecuted() -> None:
    sent: list[str] = []
    tool = Tool(
        "send_message",
        "test-tool@v1",
        "Send one external message.",
        SideEffect.IRREVERSIBLE,
        validate_message,
        lambda arguments: sent.append(arguments["message"]) or {"sent": True},
        required_capability="test:tool",
        resolve_resource=fixture_resource,
    )
    runtime = runtime_for(tool)
    call = ToolCall("call-1", "send_message", {"message": "hello"})

    paused = runtime.execute(call, context=CONTEXT)
    completed = runtime.execute(
        call, context=CONTEXT, approval=grant_for(paused)
    )
    replayed = runtime.execute(call, context=CONTEXT)

    assert paused.status is ExecutionStatus.NEEDS_APPROVAL
    assert completed.status is ExecutionStatus.COMPLETED
    assert replayed.status is ExecutionStatus.CACHED
    assert sent == ["hello"]
    assert runtime.executed_tool_calls == 1


def test_call_id_reuse_with_changed_arguments_is_rejected() -> None:
    tool = Tool(
        "echo",
        "test-tool@v1",
        "Return a message.",
        SideEffect.READ_ONLY,
        validate_message,
        lambda arguments: arguments["message"],
        required_capability="test:tool",
        resolve_resource=fixture_resource,
    )
    runtime = runtime_for(tool)
    runtime.execute(
        ToolCall("same-id", "echo", {"message": "first"}), context=CONTEXT
    )

    with pytest.raises(IdempotencyConflict):
        runtime.execute(
            ToolCall("same-id", "echo", {"message": "changed"}), context=CONTEXT
        )


def test_budget_blocks_new_calls_but_allows_cached_replay() -> None:
    tool = Tool(
        "echo",
        "test-tool@v1",
        "Return a message.",
        SideEffect.READ_ONLY,
        validate_message,
        lambda arguments: arguments["message"],
        required_capability="test:tool",
        resolve_resource=fixture_resource,
    )
    runtime = runtime_for(tool, max_tool_calls=1)
    first = ToolCall("first", "echo", {"message": "one"})
    second = ToolCall("second", "echo", {"message": "two"})

    assert runtime.execute(first, context=CONTEXT).status is ExecutionStatus.COMPLETED
    assert runtime.execute(second, context=CONTEXT).status is ExecutionStatus.FAILED
    assert runtime.execute(first, context=CONTEXT).status is ExecutionStatus.CACHED


def test_validation_happens_before_approval_request() -> None:
    tool = Tool(
        "send_message",
        "test-tool@v1",
        "Send one external message.",
        SideEffect.IRREVERSIBLE,
        validate_message,
        lambda arguments: arguments,
        required_capability="test:tool",
        resolve_resource=fixture_resource,
    )
    runtime = runtime_for(tool)
    with pytest.raises(ValueError, match="expected one string field"):
        runtime.execute(
            ToolCall("bad", "send_message", {"message": 123}), context=CONTEXT
        )


def test_tool_call_snapshots_arguments_and_fingerprint() -> None:
    mutable = {"message": "approved value"}
    call = ToolCall("snapshot", "echo", mutable)
    fingerprint = call.fingerprint()
    mutable["message"] = "changed after construction"
    tool = Tool(
        "echo",
        "test-tool@v1",
        "Return one message.",
        SideEffect.READ_ONLY,
        validate_message,
        lambda arguments: arguments["message"],
        required_capability="test:tool",
        resolve_resource=fixture_resource,
    )

    outcome = runtime_for(tool).execute(call, context=CONTEXT)

    assert call.fingerprint() == fingerprint
    assert call.arguments["message"] == "approved value"
    assert outcome.value == "approved value"


def test_tool_call_recursively_freezes_detached_arguments() -> None:
    mutable = {
        "payment": {"amount": 10},
        "recipients": [{"address": "first@example.test"}],
    }
    call = ToolCall("nested", "pay", mutable)

    mutable["payment"]["amount"] = 1000
    mutable["recipients"][0]["address"] = "attacker@example.test"

    assert call.arguments["payment"]["amount"] == 10
    assert call.arguments["recipients"][0]["address"] == "first@example.test"
    with pytest.raises(TypeError):
        call.arguments["payment"]["amount"] = 1000
    with pytest.raises(TypeError):
        call.arguments["recipients"][0]["address"] = "attacker@example.test"


def test_tool_call_fingerprint_is_stable_hash_without_argument_plaintext() -> None:
    first = ToolCall(
        "first",
        "send",
        {"recipient": "person@example.test", "secret": "do-not-log-this"},
    )
    reordered = ToolCall(
        "second",
        "send",
        {"secret": "do-not-log-this", "recipient": "person@example.test"},
    )
    changed = ToolCall(
        "third",
        "send",
        {"recipient": "other@example.test", "secret": "do-not-log-this"},
    )

    assert first.fingerprint() == reordered.fingerprint()
    assert first.fingerprint() != changed.fingerprint()
    assert first.fingerprint().startswith("sha256:")
    assert len(first.fingerprint()) == len("sha256:") + 64
    assert "do-not-log-this" not in first.fingerprint()
    assert "person@example.test" not in first.fingerprint()


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"value": float("nan")}, "non-finite float"),
        ({"value": float("inf")}, "non-finite float"),
        ({"nested": {1: "not a JSON object key"}}, "non-string object key"),
        ({"value": object()}, "non-JSON value"),
    ],
)
def test_tool_call_rejects_values_outside_strict_json_domain(
    arguments: dict[Any, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ToolCall("invalid", "tool", arguments)


def test_failed_handler_attempt_consumes_budget() -> None:
    def fail(_: Mapping[str, Any]) -> None:
        raise TimeoutError("remote state unknown")

    tool = Tool(
        "write",
        "test-tool@v1",
        "Uncertain write.",
        SideEffect.IRREVERSIBLE,
        validate_message,
        fail,
        required_capability="test:tool",
        resolve_resource=fixture_resource,
    )
    runtime = runtime_for(tool, max_tool_calls=1)

    first_call = ToolCall("first", "write", {"message": "one"})
    first_preview = runtime.execute(first_call, context=CONTEXT)
    first = runtime.execute(
        first_call,
        context=CONTEXT,
        approval=grant_for(first_preview),
    )
    second_call = ToolCall("second", "write", {"message": "two"})
    second_preview = runtime.execute(second_call, context=CONTEXT)
    second = runtime.execute(
        second_call,
        context=CONTEXT,
        approval=grant_for(second_preview),
    )

    assert first.status is ExecutionStatus.FAILED
    assert "pending" in first.message
    assert second.status is ExecutionStatus.FAILED
    assert "budget exhausted" in second.message
    assert runtime.executed_tool_calls == 1


def test_handler_result_is_detached_and_recursively_immutable() -> None:
    mutable_result = {"nested": {"items": [1, 2]}}
    tool = Tool(
        "read",
        "test-tool@v1",
        "Return a mutable fixture result.",
        SideEffect.READ_ONLY,
        validate_message,
        lambda arguments: mutable_result,
        required_capability="test:tool",
        resolve_resource=fixture_resource,
    )
    runtime = runtime_for(tool)
    call = ToolCall("immutable-result", "read", {"message": "x"})

    first = runtime.execute(call, context=CONTEXT)
    mutable_result["nested"]["items"].append(3)
    replayed = runtime.execute(call, context=CONTEXT)

    assert first.value["nested"]["items"] == (1, 2)
    assert replayed.value["nested"]["items"] == (1, 2)
    with pytest.raises(TypeError):
        first.value["nested"]["new"] = "mutation"


@pytest.mark.parametrize(
    "invalid_result", [{"score": float("nan")}, {"opaque": object()}]
)
def test_invalid_handler_result_stays_pending_without_reexecution(
    invalid_result: object,
) -> None:
    attempts = 0

    def handler(_: Mapping[str, Any]) -> object:
        nonlocal attempts
        attempts += 1
        return invalid_result

    tool = Tool(
        "read",
        "test-tool@v1",
        "Return an invalid result.",
        SideEffect.READ_ONLY,
        validate_message,
        handler,
        required_capability="test:tool",
        resolve_resource=fixture_resource,
    )
    runtime = runtime_for(tool)
    call = ToolCall("invalid-result", "read", {"message": "x"})

    first = runtime.execute(call, context=CONTEXT)
    replayed = runtime.execute(call, context=CONTEXT)

    assert first.status is ExecutionStatus.FAILED
    assert "invalid strict JSON" in first.message
    assert replayed.status is ExecutionStatus.FAILED
    assert "pending" in replayed.message
    assert attempts == 1
