"""A framework-independent, approval-aware tool execution runtime."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class SideEffect(str, Enum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    CACHED = "cached"
    NEEDS_APPROVAL = "needs_approval"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("call_id cannot be empty")
        if not self.tool_name.strip():
            raise ValueError("tool_name cannot be empty")

    def fingerprint(self) -> str:
        try:
            serialized = json.dumps(
                {"tool_name": self.tool_name, "arguments": self.arguments},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("tool arguments must be JSON serializable") from error
        return serialized


Validator = Callable[[Mapping[str, Any]], None]
Handler = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    side_effect: SideEffect
    validate: Validator
    handler: Handler

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name cannot be empty")
        if not self.description.strip():
            raise ValueError("tool description cannot be empty")


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(f"unknown tool: {name}") from error


@dataclass(frozen=True)
class ExecutionOutcome:
    status: ExecutionStatus
    call: ToolCall
    value: Any = None
    message: str = ""


class LedgerState(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    COMPENSATED = "compensated"


@dataclass(frozen=True)
class LedgerEntry:
    fingerprint: str
    state: LedgerState
    value: Any = None


class ExecutionLedger(Protocol):
    def lookup(self, call_id: str) -> LedgerEntry | None: ...

    def claim(self, call_id: str, fingerprint: str) -> tuple[LedgerEntry, bool]: ...

    def complete(self, call_id: str, fingerprint: str, value: Any) -> None: ...


class InMemoryLedger:
    def __init__(self) -> None:
        self.entries: dict[str, LedgerEntry] = {}

    def lookup(self, call_id: str) -> LedgerEntry | None:
        return self.entries.get(call_id)

    def claim(self, call_id: str, fingerprint: str) -> tuple[LedgerEntry, bool]:
        existing = self.entries.get(call_id)
        if existing is not None:
            return existing, False
        entry = LedgerEntry(fingerprint=fingerprint, state=LedgerState.PENDING)
        self.entries[call_id] = entry
        return entry, True

    def complete(self, call_id: str, fingerprint: str, value: Any) -> None:
        existing = self.entries.get(call_id)
        if existing is None or existing.fingerprint != fingerprint:
            raise IdempotencyConflict(f"cannot complete unclaimed call_id {call_id!r}")
        self.entries[call_id] = LedgerEntry(
            fingerprint=fingerprint,
            state=LedgerState.COMPLETED,
            value=value,
        )


class IdempotencyConflict(RuntimeError):
    """A call id was reused for a different tool or argument set."""


class AgentRuntime:
    """Validate, authorize, budget, execute, and deduplicate tool calls.

    The model or planner may propose ToolCall values. Only this runtime decides
    whether a call can execute. A production ledger should be durable and use
    atomic compare-and-set; this in-memory implementation exposes the contract.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_tool_calls: int = 10,
        ledger: ExecutionLedger | None = None,
    ) -> None:
        if max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")
        self.registry = registry
        self.max_tool_calls = max_tool_calls
        self.executed_tool_calls = 0
        self.ledger = ledger or InMemoryLedger()

    def execute(self, call: ToolCall, *, approved: bool = False) -> ExecutionOutcome:
        tool = self.registry.get(call.tool_name)
        tool.validate(call.arguments)
        fingerprint = call.fingerprint()
        existing = self.ledger.lookup(call.call_id)
        if existing is not None:
            return self._existing_outcome(call, fingerprint, existing)

        if tool.side_effect is not SideEffect.READ_ONLY and not approved:
            return ExecutionOutcome(
                ExecutionStatus.NEEDS_APPROVAL,
                call,
                message=f"{tool.side_effect.value} tool requires explicit approval",
            )
        if self.executed_tool_calls >= self.max_tool_calls:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                message=f"tool-call budget exhausted ({self.max_tool_calls})",
            )

        claimed, acquired = self.ledger.claim(call.call_id, fingerprint)
        if not acquired:
            return self._existing_outcome(call, fingerprint, claimed)

        try:
            value = tool.handler(call.arguments)
        except Exception as error:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                message=(
                    f"{type(error).__name__}: {error}; call remains pending "
                    "and requires reconciliation before retry"
                ),
            )
        self.executed_tool_calls += 1
        try:
            self.ledger.complete(call.call_id, fingerprint, value)
        except Exception as error:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                message=(
                    f"handler completed but ledger completion failed: "
                    f"{type(error).__name__}: {error}; reconcile before retry"
                ),
            )
        return ExecutionOutcome(ExecutionStatus.COMPLETED, call, value=value)

    @staticmethod
    def _existing_outcome(
        call: ToolCall,
        fingerprint: str,
        existing: LedgerEntry,
    ) -> ExecutionOutcome:
        if existing.fingerprint != fingerprint:
            raise IdempotencyConflict(
                f"call_id {call.call_id!r} was already used with different input"
            )
        if existing.state is LedgerState.PENDING:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                message="call is pending; reconcile external state before retry",
            )
        if existing.state in {LedgerState.ABANDONED, LedgerState.COMPENSATED}:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                message=(
                    f"call was reconciled as {existing.state.value}; "
                    "retry requires a newly approved call_id"
                ),
            )
        return ExecutionOutcome(
            ExecutionStatus.CACHED,
            call,
            value=existing.value,
            message="Result reused; handler was not executed again.",
        )
