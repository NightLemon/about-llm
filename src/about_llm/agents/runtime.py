"""A framework-independent, approval-aware tool execution runtime."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

from about_llm.agents.approval import ApprovalGrant, approval_rejection_reason
from about_llm.agents.policy import (
    DefaultDenyPolicy,
    ExecutionContext,
    PolicyDecision,
    PolicyEffect,
    PolicyEvaluator,
    PolicyRequest,
    ResourceRef,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes


class SideEffect(str, Enum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    CACHED = "cached"
    NEEDS_APPROVAL = "needs_approval"
    APPROVAL_REJECTED = "approval_rejected"
    POLICY_DENIED = "policy_denied"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("call_id cannot be empty")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name cannot be empty")
        try:
            # Canonical serialization both validates the JSON domain and
            # detaches the approved value from caller-owned mutable objects.
            snapshot = json.loads(canonical_json_bytes(self.arguments))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid JSON tool arguments: {error}") from error
        if not isinstance(snapshot, dict):
            raise ValueError("tool arguments must be a JSON object")
        object.__setattr__(self, "arguments", freeze_json_value(snapshot))
        object.__setattr__(
            self,
            "_fingerprint",
            "sha256:"
            + artifact_fingerprint(
                {"tool_name": self.tool_name, "arguments": snapshot}
            ),
        )

    def fingerprint(self) -> str:
        return self._fingerprint


def freeze_json_value(value: Any) -> Any:
    """Recursively freeze an already validated, detached JSON value."""

    if isinstance(value, dict):
        return MappingProxyType(
            {key: freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(freeze_json_value(item) for item in value)
    return value


Validator = Callable[[Mapping[str, Any]], None]
Handler = Callable[[Mapping[str, Any]], Any]
ResourceResolver = Callable[[Mapping[str, Any]], ResourceRef]


@dataclass(frozen=True)
class Tool:
    name: str
    version: str
    description: str
    side_effect: SideEffect
    validate: Validator
    handler: Handler
    required_capability: str
    resolve_resource: ResourceResolver

    def __post_init__(self) -> None:
        for field_name, value in (
            ("name", self.name),
            ("version", self.version),
            ("description", self.description),
            ("required_capability", self.required_capability),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"tool {field_name} cannot be empty")
        if not isinstance(self.side_effect, SideEffect):
            raise ValueError("tool side_effect must be a SideEffect")
        if not callable(self.validate) or not callable(self.handler):
            raise ValueError("tool validate and handler must be callable")
        if not callable(self.resolve_resource):
            raise ValueError("tool resolve_resource must be callable")


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
    policy_decision: PolicyDecision
    resource: ResourceRef
    execution_fingerprint: str
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
    """A call id was reused for a different authorized execution identity."""


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
        initial_executed_tool_calls: int = 0,
        ledger: ExecutionLedger | None = None,
        policy: PolicyEvaluator | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(max_tool_calls, bool) or max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")
        if (
            isinstance(initial_executed_tool_calls, bool)
            or not isinstance(initial_executed_tool_calls, int)
            or initial_executed_tool_calls < 0
            or initial_executed_tool_calls > max_tool_calls
        ):
            raise ValueError(
                "initial_executed_tool_calls must be between zero and max_tool_calls"
            )
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.registry = registry
        self.max_tool_calls = max_tool_calls
        self.executed_tool_calls = initial_executed_tool_calls
        self.ledger = ledger if ledger is not None else InMemoryLedger()
        self.policy = policy if policy is not None else DefaultDenyPolicy()
        self.clock = clock

    def execute(
        self,
        call: ToolCall,
        *,
        context: ExecutionContext,
        approval: ApprovalGrant | None = None,
    ) -> ExecutionOutcome:
        tool = self.registry.get(call.tool_name)
        tool.validate(call.arguments)
        resource = tool.resolve_resource(call.arguments)
        if not isinstance(resource, ResourceRef):
            raise TypeError("tool resource resolver must return ResourceRef")
        fingerprint = call.fingerprint()
        policy_decision = self.policy.evaluate(
            PolicyRequest(
                context=context,
                call_id=call.call_id,
                call_fingerprint=fingerprint,
                tool_name=tool.name,
                required_capability=tool.required_capability,
                side_effect=tool.side_effect.value,
                resource=resource,
            )
        )
        if not isinstance(policy_decision, PolicyDecision):
            raise TypeError("policy evaluator must return PolicyDecision")
        execution_identity = execution_fingerprint(
            call=call,
            tool=tool,
            context=context,
            policy_decision=policy_decision,
            resource=resource,
        )
        if policy_decision.effect is not PolicyEffect.ALLOW:
            return ExecutionOutcome(
                ExecutionStatus.POLICY_DENIED,
                call,
                policy_decision,
                resource,
                execution_identity,
                message=f"policy denied call: {policy_decision.reason_code}",
            )
        existing = self.ledger.lookup(call.call_id)
        if existing is not None:
            return self._existing_outcome(
                call,
                execution_identity,
                existing,
                policy_decision,
                resource,
            )

        if tool.side_effect is not SideEffect.READ_ONLY:
            if approval is None:
                return ExecutionOutcome(
                    ExecutionStatus.NEEDS_APPROVAL,
                    call,
                    policy_decision,
                    resource,
                    execution_identity,
                    message=f"{tool.side_effect.value} tool requires explicit approval",
                )
            rejection = approval_rejection_reason(
                approval,
                context=context,
                call_id=call.call_id,
                execution_fingerprint=execution_identity,
                now_epoch_seconds=self.clock(),
            )
            if rejection is not None:
                return ExecutionOutcome(
                    ExecutionStatus.APPROVAL_REJECTED,
                    call,
                    policy_decision,
                    resource,
                    execution_identity,
                    message=f"approval rejected: {rejection}",
                )
        if self.executed_tool_calls >= self.max_tool_calls:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                policy_decision,
                resource,
                execution_identity,
                message=f"tool-call budget exhausted ({self.max_tool_calls})",
            )

        claimed, acquired = self.ledger.claim(call.call_id, execution_identity)
        if not acquired:
            return self._existing_outcome(
                call,
                execution_identity,
                claimed,
                policy_decision,
                resource,
            )

        # A claimed handler attempt consumes budget even when the remote state
        # becomes uncertain. Otherwise repeated failures bypass the hard cap.
        self.executed_tool_calls += 1
        try:
            raw_value = tool.handler(call.arguments)
        except Exception as error:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                policy_decision,
                resource,
                execution_identity,
                message=(
                    f"{type(error).__name__}: {error}; call remains pending "
                    "and requires reconciliation before retry"
                ),
            )
        try:
            value = freeze_json_value(json.loads(canonical_json_bytes(raw_value)))
        except (TypeError, ValueError) as error:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                policy_decision,
                resource,
                execution_identity,
                message=(
                    f"handler returned invalid strict JSON: {error}; call remains "
                    "pending because external effect state may require reconciliation"
                ),
            )
        try:
            self.ledger.complete(call.call_id, execution_identity, value)
        except Exception as error:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                policy_decision,
                resource,
                execution_identity,
                message=(
                    f"handler completed but ledger completion failed: "
                    f"{type(error).__name__}: {error}; reconcile before retry"
                ),
            )
        return ExecutionOutcome(
            ExecutionStatus.COMPLETED,
            call,
            policy_decision,
            resource,
            execution_identity,
            value=value,
        )

    @staticmethod
    def _existing_outcome(
        call: ToolCall,
        fingerprint: str,
        existing: LedgerEntry,
        policy_decision: PolicyDecision,
        resource: ResourceRef,
    ) -> ExecutionOutcome:
        if existing.fingerprint != fingerprint:
            raise IdempotencyConflict(
                f"call_id {call.call_id!r} was already used with a different "
                "execution identity"
            )
        if existing.state is LedgerState.PENDING:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                policy_decision,
                resource,
                fingerprint,
                message="call is pending; reconcile external state before retry",
            )
        if existing.state in {LedgerState.ABANDONED, LedgerState.COMPENSATED}:
            return ExecutionOutcome(
                ExecutionStatus.FAILED,
                call,
                policy_decision,
                resource,
                fingerprint,
                message=(
                    f"call was reconciled as {existing.state.value}; "
                    "retry requires a newly approved call_id"
                ),
            )
        return ExecutionOutcome(
            ExecutionStatus.CACHED,
            call,
            policy_decision,
            resource,
            fingerprint,
            value=existing.value,
            message="Result reused; handler was not executed again.",
        )


def execution_fingerprint(
    *,
    call: ToolCall,
    tool: Tool,
    context: ExecutionContext,
    policy_decision: PolicyDecision,
    resource: ResourceRef,
) -> str:
    """Bind idempotent execution to trusted contract and authorization state."""

    return "sha256:" + artifact_fingerprint(
        {
            "proposal_fingerprint": call.fingerprint(),
            "tool": {
                "name": tool.name,
                "version": tool.version,
                "side_effect": tool.side_effect.value,
                "required_capability": tool.required_capability,
            },
            "context": {
                "task_id": context.task_id,
                "subject_id": context.subject_id,
                "tenant_id": context.tenant_id,
            },
            "resource": {
                "tenant_id": resource.tenant_id,
                "resource_type": resource.resource_type,
                "resource_id": resource.resource_id,
                "version": resource.version,
            },
            "policy": {
                "version": policy_decision.policy_version,
                "effect": policy_decision.effect.value,
                "reason_code": policy_decision.reason_code,
            },
        }
    )
