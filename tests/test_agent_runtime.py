from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from about_llm.agents import (
    AgentRuntime,
    ExecutionStatus,
    IdempotencyConflict,
    SideEffect,
    Tool,
    ToolCall,
    ToolRegistry,
)


def validate_message(arguments: Mapping[str, Any]) -> None:
    if set(arguments) != {"message"} or not isinstance(arguments["message"], str):
        raise ValueError("expected one string field: message")


def test_side_effect_requires_approval_and_duplicate_is_not_reexecuted() -> None:
    sent: list[str] = []
    tool = Tool(
        "send_message",
        "Send one external message.",
        SideEffect.IRREVERSIBLE,
        validate_message,
        lambda arguments: sent.append(arguments["message"]) or {"sent": True},
    )
    runtime = AgentRuntime(ToolRegistry([tool]))
    call = ToolCall("call-1", "send_message", {"message": "hello"})

    paused = runtime.execute(call)
    completed = runtime.execute(call, approved=True)
    replayed = runtime.execute(call, approved=True)

    assert paused.status is ExecutionStatus.NEEDS_APPROVAL
    assert completed.status is ExecutionStatus.COMPLETED
    assert replayed.status is ExecutionStatus.CACHED
    assert sent == ["hello"]
    assert runtime.executed_tool_calls == 1


def test_call_id_reuse_with_changed_arguments_is_rejected() -> None:
    tool = Tool(
        "echo",
        "Return a message.",
        SideEffect.READ_ONLY,
        validate_message,
        lambda arguments: arguments["message"],
    )
    runtime = AgentRuntime(ToolRegistry([tool]))
    runtime.execute(ToolCall("same-id", "echo", {"message": "first"}))

    with pytest.raises(IdempotencyConflict):
        runtime.execute(ToolCall("same-id", "echo", {"message": "changed"}))


def test_budget_blocks_new_calls_but_allows_cached_replay() -> None:
    tool = Tool(
        "echo",
        "Return a message.",
        SideEffect.READ_ONLY,
        validate_message,
        lambda arguments: arguments["message"],
    )
    runtime = AgentRuntime(ToolRegistry([tool]), max_tool_calls=1)
    first = ToolCall("first", "echo", {"message": "one"})
    second = ToolCall("second", "echo", {"message": "two"})

    assert runtime.execute(first).status is ExecutionStatus.COMPLETED
    assert runtime.execute(second).status is ExecutionStatus.FAILED
    assert runtime.execute(first).status is ExecutionStatus.CACHED


def test_validation_happens_before_approval_request() -> None:
    tool = Tool(
        "send_message",
        "Send one external message.",
        SideEffect.IRREVERSIBLE,
        validate_message,
        lambda arguments: arguments,
    )
    runtime = AgentRuntime(ToolRegistry([tool]))
    with pytest.raises(ValueError, match="expected one string field"):
        runtime.execute(ToolCall("bad", "send_message", {"message": 123}))
