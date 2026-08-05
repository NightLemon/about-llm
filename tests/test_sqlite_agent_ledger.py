from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from about_llm.agents import (
    AgentRuntime,
    ExecutionStatus,
    SideEffect,
    SQLiteLedger,
    Tool,
    ToolCall,
    ToolRegistry,
)


def validate_value(arguments: Mapping[str, Any]) -> None:
    if set(arguments) != {"value"}:
        raise ValueError("expected value")


def test_sqlite_ledger_reuses_result_across_runtime_instances(tmp_path: Path) -> None:
    effects: list[str] = []
    tool = Tool(
        "write",
        "Append one value.",
        SideEffect.IRREVERSIBLE,
        validate_value,
        lambda arguments: effects.append(arguments["value"]) or {"ok": True},
    )
    registry = ToolRegistry([tool])
    path = tmp_path / "agent.db"
    call = ToolCall("stable-id", "write", {"value": "once"})

    first = AgentRuntime(registry, ledger=SQLiteLedger(path)).execute(call, approved=True)
    second = AgentRuntime(registry, ledger=SQLiteLedger(path)).execute(call, approved=True)

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
        "Perform an uncertain external action.",
        SideEffect.IRREVERSIBLE,
        validate_value,
        uncertain_handler,
    )
    path = tmp_path / "agent.db"
    call = ToolCall("pending-id", "write", {"value": "x"})

    first = AgentRuntime(ToolRegistry([tool]), ledger=SQLiteLedger(path)).execute(
        call, approved=True
    )
    second = AgentRuntime(ToolRegistry([tool]), ledger=SQLiteLedger(path)).execute(
        call, approved=True
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
        "Append one value.",
        SideEffect.IRREVERSIBLE,
        validate_value,
        lambda arguments: effects.append(arguments["value"]),
    )
    ledger = SQLiteLedger(tmp_path / "abandon.db")
    old_call = ToolCall("old-id", "write", {"value": "x"})
    ledger.claim(old_call.call_id, old_call.fingerprint())
    ledger.resolve_without_completion(old_call.call_id, note="Provider confirms no operation")

    old_outcome = AgentRuntime(ToolRegistry([tool]), ledger=ledger).execute(
        old_call, approved=True
    )
    new_outcome = AgentRuntime(ToolRegistry([tool]), ledger=ledger).execute(
        ToolCall("new-id", "write", {"value": "x"}), approved=True
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
