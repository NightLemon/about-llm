from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from about_llm.agents import (
    AgentRuntime,
    ApprovalGrant,
    CapabilityPolicy,
    ExecutionContext,
    ExecutionStatus,
    PolicyRequest,
    ResourceRef,
    SideEffect,
    SQLiteLedger,
    Tool,
    ToolCall,
    ToolRegistry,
    execution_fingerprint,
)

pytestmark = [pytest.mark.contract, pytest.mark.security, pytest.mark.integration]

CONTEXT = ExecutionContext(
    "task-ledger", "user-1", "tenant-a", frozenset({"test:write"})
)
POLICY = CapabilityPolicy("test-policy@v1")


def fixture_resource(_: Mapping[str, Any]) -> ResourceRef:
    return ResourceRef("tenant-a", "fixture", "resource-1", "fixture@v1")


def runtime_with_ledger(registry: ToolRegistry, ledger: SQLiteLedger) -> AgentRuntime:
    return AgentRuntime(
        registry, ledger=ledger, policy=POLICY, clock=lambda: 100.0
    )


def grant_for(runtime: AgentRuntime, call: ToolCall) -> ApprovalGrant:
    preview = runtime.execute(call, context=CONTEXT)
    assert preview.status is ExecutionStatus.NEEDS_APPROVAL
    return ApprovalGrant(
        f"approval-{call.call_id}",
        "approver-1",
        CONTEXT.subject_id,
        CONTEXT.task_id,
        call.call_id,
        preview.execution_fingerprint,
        200.0,
    )


def validate_value(arguments: Mapping[str, Any]) -> None:
    if set(arguments) != {"value"}:
        raise ValueError("expected value")


def test_sqlite_ledger_reuses_result_across_runtime_instances(tmp_path: Path) -> None:
    effects: list[str] = []
    tool = Tool(
        "write",
        "test-tool@v1",
        "Append one value.",
        SideEffect.IRREVERSIBLE,
        validate_value,
        lambda arguments: effects.append(arguments["value"]) or {"ok": True},
        required_capability="test:write",
        resolve_resource=fixture_resource,
    )
    registry = ToolRegistry([tool])
    path = tmp_path / "agent.db"
    call = ToolCall("stable-id", "write", {"value": "once"})

    first_runtime = runtime_with_ledger(registry, SQLiteLedger(path))
    first = first_runtime.execute(
        call, context=CONTEXT, approval=grant_for(first_runtime, call)
    )
    second = runtime_with_ledger(registry, SQLiteLedger(path)).execute(
        call, context=CONTEXT
    )

    assert first.status is ExecutionStatus.COMPLETED
    assert second.status is ExecutionStatus.CACHED
    assert second.value == {"ok": True}
    assert effects == ["once"]


def test_failed_side_effect_stays_pending_and_is_not_replayed(tmp_path: Path) -> None:
    attempts = 0

    def uncertain_handler(_: Mapping[str, Any]) -> None:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("remote state unknown")

    tool = Tool(
        "write",
        "test-tool@v1",
        "Perform an uncertain external action.",
        SideEffect.IRREVERSIBLE,
        validate_value,
        uncertain_handler,
        required_capability="test:write",
        resolve_resource=fixture_resource,
    )
    path = tmp_path / "agent.db"
    call = ToolCall("pending-id", "write", {"value": "x"})

    first_runtime = runtime_with_ledger(ToolRegistry([tool]), SQLiteLedger(path))
    first = first_runtime.execute(
        call, context=CONTEXT, approval=grant_for(first_runtime, call)
    )
    second = runtime_with_ledger(ToolRegistry([tool]), SQLiteLedger(path)).execute(
        call, context=CONTEXT
    )

    assert first.status is ExecutionStatus.FAILED
    assert "reconciliation" in first.message
    assert second.status is ExecutionStatus.FAILED
    assert "pending" in second.message
    assert attempts == 1


def test_sqlite_claim_reports_exactly_one_acquirer(tmp_path: Path) -> None:
    first_ledger = SQLiteLedger(tmp_path / "claim.db")
    second_ledger = SQLiteLedger(tmp_path / "claim.db")
    first_entry, first_acquired = first_ledger.claim("id", "fingerprint")
    second_entry, second_acquired = second_ledger.claim("id", "fingerprint")

    assert first_acquired is True
    assert second_acquired is False
    assert first_entry == second_entry


def test_operator_can_confirm_uncertain_external_completion(tmp_path: Path) -> None:
    ledger = SQLiteLedger(tmp_path / "confirm.db")
    ledger.claim("call-1", "fingerprint")

    pending = ledger.list_stale_pending(older_than_seconds=0)
    assert [call.call_id for call in pending] == ["call-1"]

    ledger.resolve_external_completion(
        "call-1", {"remote_id": "r-7"}, note="Confirmed in provider audit log"
    )

    assert ledger.lookup("call-1") is not None
    assert ledger.lookup("call-1").state.value == "completed"  # type: ignore[union-attr]
    assert ledger.lookup("call-1").value == {"remote_id": "r-7"}  # type: ignore[union-attr]
    assert ledger.list_stale_pending(older_than_seconds=0) == ()
    assert ledger.reconciliation_history("call-1")[0].resolution == "externally_confirmed"


def test_abandoned_call_cannot_be_retried_with_same_call_id(tmp_path: Path) -> None:
    effects: list[str] = []
    tool = Tool(
        "write",
        "test-tool@v1",
        "Append one value.",
        SideEffect.IRREVERSIBLE,
        validate_value,
        lambda arguments: effects.append(arguments["value"]),
        required_capability="test:write",
        resolve_resource=fixture_resource,
    )
    ledger = SQLiteLedger(tmp_path / "abandon.db")
    old_call = ToolCall("old-id", "write", {"value": "x"})
    resource = fixture_resource(old_call.arguments)
    decision = POLICY.evaluate(
        PolicyRequest(
            context=CONTEXT,
            call_id=old_call.call_id,
            call_fingerprint=old_call.fingerprint(),
            tool_name=tool.name,
            required_capability=tool.required_capability,
            side_effect=tool.side_effect.value,
            resource=resource,
        )
    )
    identity = execution_fingerprint(
        call=old_call,
        tool=tool,
        context=CONTEXT,
        policy_decision=decision,
        resource=resource,
    )
    ledger.claim(old_call.call_id, identity)
    ledger.resolve_without_completion(old_call.call_id, note="Provider confirms no operation")

    old_outcome = runtime_with_ledger(ToolRegistry([tool]), ledger).execute(
        old_call, context=CONTEXT
    )
    new_runtime = runtime_with_ledger(ToolRegistry([tool]), ledger)
    new_call = ToolCall("new-id", "write", {"value": "x"})
    new_outcome = new_runtime.execute(
        new_call,
        context=CONTEXT,
        approval=grant_for(new_runtime, new_call),
    )

    assert old_outcome.status is ExecutionStatus.FAILED
    assert "newly approved call_id" in old_outcome.message
    assert new_outcome.status is ExecutionStatus.COMPLETED
    assert effects == ["x"]


def test_compensation_is_preserved_in_audit_history(tmp_path: Path) -> None:
    ledger = SQLiteLedger(tmp_path / "compensate.db")
    ledger.claim("call-2", "fingerprint")
    ledger.resolve_without_completion(
        "call-2", note="Refund r-8 verified", compensated=True
    )

    entry = ledger.lookup("call-2")
    assert entry is not None and entry.state.value == "compensated"
    history = ledger.reconciliation_history("call-2")
    assert [(event.resolution, event.note) for event in history] == [
        ("compensated", "Refund r-8 verified")
    ]

    with pytest.raises(RuntimeError, match="conflicting"):
        ledger.complete("call-2", "fingerprint", {"late": "result"})


@pytest.mark.parametrize(
    "value, message",
    [
        ({"score": float("nan")}, "non-finite float"),
        ({"nested": {1: "coerced by permissive json.dumps"}}, "non-string object key"),
        ({"opaque": object()}, "non-JSON value"),
    ],
)
def test_sqlite_ledger_rejects_results_outside_strict_json(
    tmp_path: Path, value: object, message: str
) -> None:
    ledger = SQLiteLedger(tmp_path / "strict-result.db")
    entry, acquired = ledger.claim("call-strict", "sha256:" + "a" * 64)
    assert acquired is True

    with pytest.raises(ValueError, match=message):
        ledger.complete("call-strict", entry.fingerprint, value)
    with pytest.raises(ValueError, match=message):
        ledger.resolve_external_completion(
            "call-strict", value, note="invalid result must not enter audit ledger"
        )


def test_sqlite_operations_release_file_handles(tmp_path: Path) -> None:
    path = tmp_path / "closable.db"
    ledger = SQLiteLedger(path)
    ledger.claim("close-1", "sha256:" + "b" * 64)
    assert ledger.lookup("close-1") is not None

    moved = tmp_path / "moved.db"
    path.rename(moved)

    assert SQLiteLedger(moved).lookup("close-1") is not None
