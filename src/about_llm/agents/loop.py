"""Budgeted, verifier-driven Agent loop with deterministic stop conditions."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol, TypeAlias, cast

from about_llm.agents.approval import ApprovalGrant
from about_llm.agents.policy import ExecutionContext
from about_llm.agents.runtime import (
    AgentRuntime,
    ExecutionStatus,
    ToolCall,
    freeze_json_value,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"


class LoopTermination(str, Enum):
    COMPLETED = "completed"
    ESCALATED = "escalated"
    NEEDS_APPROVAL = "needs_approval"
    STEP_BUDGET = "step_budget"
    MODEL_TOKEN_BUDGET = "model_token_budget"
    COST_BUDGET = "cost_budget"
    WALL_TIME_BUDGET = "wall_time_budget"
    REPEATED_ACTION = "repeated_action"
    ACTION_CYCLE = "action_cycle"
    REPEATED_ERROR = "repeated_error"
    PLANNER_ERROR = "planner_error"
    RUNTIME_ERROR = "runtime_error"
    VERIFIER_ERROR = "verifier_error"
    APPROVAL_REJECTED = "approval_rejected"


@dataclass(frozen=True)
class ToolProposal:
    call: ToolCall


@dataclass(frozen=True)
class FinishProposal:
    answer: str
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("finish answer cannot be empty")
        evidence = tuple(self.evidence_ids)
        if any(
            not isinstance(evidence_id, str) or not evidence_id.strip()
            for evidence_id in evidence
        ):
            raise ValueError("evidence_ids must contain non-empty strings")
        if len(evidence) != len(set(evidence)):
            raise ValueError("evidence_ids must not contain duplicates")
        object.__setattr__(self, "evidence_ids", evidence)


@dataclass(frozen=True)
class EscalationProposal:
    reason_code: str
    message: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reason_code, str)
            or not self.reason_code.strip()
            or not isinstance(self.message, str)
            or not self.message.strip()
        ):
            raise ValueError("escalation reason_code and message cannot be empty")


AgentAction: TypeAlias = ToolProposal | FinishProposal | EscalationProposal


@dataclass(frozen=True)
class PlannerDecision:
    decision_id: str
    model_revision: str
    action: AgentAction
    input_tokens: int
    output_tokens: int
    cost_units: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.decision_id, str)
            or not self.decision_id.strip()
            or not isinstance(self.model_revision, str)
            or not self.model_revision.strip()
        ):
            raise ValueError("decision_id and model_revision cannot be empty")
        if not isinstance(
            self.action, (ToolProposal, FinishProposal, EscalationProposal)
        ):
            raise ValueError("planner action has an unsupported type")
        for field_name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if isinstance(self.cost_units, bool) or not isinstance(
            self.cost_units, (int, float)
        ) or not math.isfinite(self.cost_units) or self.cost_units < 0:
            raise ValueError("cost_units must be a finite non-negative number")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    verifier_version: str
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, VerificationStatus):
            raise ValueError("verification status must be a VerificationStatus")
        if (
            not isinstance(self.verifier_version, str)
            or not self.verifier_version.strip()
            or not isinstance(self.reason_code, str)
            or not self.reason_code.strip()
        ):
            raise ValueError("verifier_version and reason_code cannot be empty")


@dataclass(frozen=True)
class LoopBudget:
    max_steps: int
    max_model_tokens: int
    max_cost_units: float
    max_wall_time_seconds: float
    repeated_action_limit: int = 3
    repeated_error_limit: int = 3

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_steps", self.max_steps),
            ("max_model_tokens", self.max_model_tokens),
            ("repeated_action_limit", self.repeated_action_limit),
            ("repeated_error_limit", self.repeated_error_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.repeated_action_limit < 2 or self.repeated_error_limit < 2:
            raise ValueError("repetition limits must be at least 2")
        if isinstance(self.max_cost_units, bool) or not isinstance(
            self.max_cost_units, (int, float)
        ):
            raise ValueError("max_cost_units must be numeric")
        if not math.isfinite(self.max_cost_units) or self.max_cost_units < 0:
            raise ValueError("max_cost_units has an invalid boundary")
        if isinstance(self.max_wall_time_seconds, bool) or not isinstance(
            self.max_wall_time_seconds, (int, float)
        ):
            raise ValueError("max_wall_time_seconds must be numeric")
        if (
            not math.isfinite(self.max_wall_time_seconds)
            or self.max_wall_time_seconds <= 0
        ):
            raise ValueError("max_wall_time_seconds has an invalid boundary")


@dataclass(frozen=True)
class RemainingBudget:
    steps: int
    model_tokens: int
    cost_units: float
    wall_time_seconds: float


@dataclass(frozen=True)
class LoopEvent:
    step: int
    decision_id: str
    model_revision: str
    action_kind: str
    action_fingerprint: str
    status: str
    message: str
    call_id: str | None = None
    proposal_fingerprint: str | None = None
    execution_fingerprint: str | None = None
    handler_attempted: bool = False
    value: Any = None
    verification: VerificationResult | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step <= 0:
            raise ValueError("loop event step must be a positive integer")
        for field_name, value in (
            ("decision_id", self.decision_id),
            ("model_revision", self.model_revision),
            ("action_kind", self.action_kind),
            ("action_fingerprint", self.action_fingerprint),
            ("status", self.status),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"loop event {field_name} must be non-empty")
        if not _is_sha256_fingerprint(self.action_fingerprint):
            raise ValueError("loop event action_fingerprint is malformed")
        if not isinstance(self.message, str):
            raise ValueError("loop event message must be a string")
        for field_name, optional_value in (
            ("call_id", self.call_id),
            ("proposal_fingerprint", self.proposal_fingerprint),
            ("execution_fingerprint", self.execution_fingerprint),
        ):
            if optional_value is not None and (
                not isinstance(optional_value, str) or not optional_value.strip()
            ):
                raise ValueError(f"loop event {field_name} must be non-empty when set")
            if (
                "fingerprint" in field_name
                and optional_value is not None
                and not _is_sha256_fingerprint(optional_value)
            ):
                raise ValueError(f"loop event {field_name} is malformed")
        if not isinstance(self.handler_attempted, bool):
            raise ValueError("loop event handler_attempted must be boolean")
        if self.verification is not None and not isinstance(
            self.verification, VerificationResult
        ):
            raise ValueError("loop event verification has an unsupported type")
        try:
            snapshot = freeze_json_value(json.loads(canonical_json_bytes(self.value)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"loop event value must be strict JSON: {error}") from error
        object.__setattr__(self, "value", snapshot)


@dataclass(frozen=True)
class AgentLoopState:
    task_id: str
    events: tuple[LoopEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("loop state task_id must be non-empty")
        events = tuple(self.events)
        if any(not isinstance(event, LoopEvent) for event in events):
            raise ValueError("loop state events must contain LoopEvent values")
        if any(event.step != index for index, event in enumerate(events, 1)):
            raise ValueError("loop state event steps must be contiguous and one-based")
        object.__setattr__(self, "events", events)


@dataclass(frozen=True)
class AgentLoopCheckpoint:
    """Strict-JSON pause state for one pending approved-tool transition.

    The fingerprint detects accidental identity drift only. Authenticity still
    requires trusted storage or an external signature/MAC.
    """

    state: AgentLoopState
    subject_id: str
    tenant_id: str
    budget: LoopBudget
    model_tokens_used: int
    cost_units_used: float
    active_wall_time_seconds: float
    handler_attempts_used: int
    runtime_executed_tool_calls: int
    runtime_max_tool_calls: int
    action_history: tuple[str, ...]
    consecutive_error_key: str | None
    consecutive_error_count: int
    pending_decision: PlannerDecision
    pending_execution_fingerprint: str
    schema_version: int = 1
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported Agent loop checkpoint schema version")
        if not isinstance(self.state, AgentLoopState):
            raise ValueError("checkpoint state must be AgentLoopState")
        for field_name, string_value in (
            ("subject_id", self.subject_id),
            ("tenant_id", self.tenant_id),
            ("pending_execution_fingerprint", self.pending_execution_fingerprint),
        ):
            if not isinstance(string_value, str) or not string_value.strip():
                raise ValueError(f"checkpoint {field_name} must be non-empty")
        if not isinstance(self.budget, LoopBudget):
            raise ValueError("checkpoint budget must be LoopBudget")
        for field_name, integer_value in (
            ("model_tokens_used", self.model_tokens_used),
            ("handler_attempts_used", self.handler_attempts_used),
            ("runtime_executed_tool_calls", self.runtime_executed_tool_calls),
            ("consecutive_error_count", self.consecutive_error_count),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value < 0
            ):
                raise ValueError(f"checkpoint {field_name} must be non-negative integer")
        for field_name, numeric_value in (
            ("cost_units_used", self.cost_units_used),
            ("active_wall_time_seconds", self.active_wall_time_seconds),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, (int, float))
                or not math.isfinite(numeric_value)
                or numeric_value < 0
            ):
                raise ValueError(f"checkpoint {field_name} must be finite and non-negative")
        history = tuple(self.action_history)
        if history != tuple(event.action_fingerprint for event in self.state.events):
            raise ValueError("checkpoint action history must match recorded event count")
        object.__setattr__(self, "action_history", history)
        if self.consecutive_error_key is not None and (
            not isinstance(self.consecutive_error_key, str)
            or not self.consecutive_error_key.strip()
        ):
            raise ValueError("checkpoint error key must be non-empty when set")
        if (self.consecutive_error_key is None) != (
            self.consecutive_error_count == 0
        ):
            raise ValueError("checkpoint error key/count are inconsistent")
        if not isinstance(self.pending_decision, PlannerDecision) or not isinstance(
            self.pending_decision.action, ToolProposal
        ):
            raise ValueError("checkpoint pending decision must be a tool proposal")
        if not _is_sha256_fingerprint(self.pending_execution_fingerprint):
            raise ValueError("checkpoint execution fingerprint is malformed")
        if not self.state.events:
            raise ValueError("checkpoint must contain a pending approval event")
        pending_event = self.state.events[-1]
        pending_call = self.pending_decision.action.call
        if not (
            pending_event.status == ExecutionStatus.NEEDS_APPROVAL.value
            and pending_event.step == len(self.state.events)
            and pending_event.decision_id == self.pending_decision.decision_id
            and pending_event.call_id == pending_call.call_id
            and pending_event.action_fingerprint == pending_call.fingerprint()
            and pending_event.execution_fingerprint
            == self.pending_execution_fingerprint
            and not pending_event.handler_attempted
            and history[-1] == pending_call.fingerprint()
        ):
            raise ValueError("checkpoint pending decision does not match its last event")
        if self.model_tokens_used > self.budget.max_model_tokens:
            raise ValueError("checkpoint exceeds model-token budget")
        if self.model_tokens_used < self.pending_decision.total_tokens:
            raise ValueError("checkpoint model-token usage omits pending decision")
        if Decimal(str(self.cost_units_used)) > Decimal(
            str(self.budget.max_cost_units)
        ):
            raise ValueError("checkpoint exceeds cost budget")
        if Decimal(str(self.cost_units_used)) < Decimal(
            str(self.pending_decision.cost_units)
        ):
            raise ValueError("checkpoint cost usage omits pending decision")
        if self.handler_attempts_used > self.runtime_executed_tool_calls:
            raise ValueError("checkpoint handler attempts exceed runtime counter")
        if (
            isinstance(self.runtime_max_tool_calls, bool)
            or not isinstance(self.runtime_max_tool_calls, int)
            or self.runtime_max_tool_calls <= 0
            or self.runtime_executed_tool_calls > self.runtime_max_tool_calls
        ):
            raise ValueError("checkpoint runtime max_tool_calls is inconsistent")
        if self.active_wall_time_seconds >= self.budget.max_wall_time_seconds:
            raise ValueError("checkpoint has no remaining active wall-time budget")
        if len(self.state.events) > self.budget.max_steps:
            raise ValueError("checkpoint exceeds step budget")
        object.__setattr__(
            self,
            "_fingerprint",
            "sha256:" + artifact_fingerprint(_checkpoint_payload(self)),
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def to_dict(self) -> dict[str, Any]:
        payload = cast(
            dict[str, Any],
            json.loads(canonical_json_bytes(_checkpoint_payload(self))),
        )
        payload["checkpoint_fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True)
class AgentLoopReport:
    termination: LoopTermination
    state: AgentLoopState
    steps_used: int
    model_tokens_used: int
    cost_units_used: float
    wall_time_seconds: float
    handler_attempts: int
    message: str | None = None
    final_answer: str | None = None
    pending_call_id: str | None = None
    pending_execution_fingerprint: str | None = None
    checkpoint: AgentLoopCheckpoint | None = None

    @property
    def completed(self) -> bool:
        return self.termination is LoopTermination.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "termination": self.termination.value,
            "completed": self.completed,
            "task_id": self.state.task_id,
            "steps_used": self.steps_used,
            "model_tokens_used": self.model_tokens_used,
            "cost_units_used": self.cost_units_used,
            "wall_time_seconds": self.wall_time_seconds,
            "handler_attempts": self.handler_attempts,
            "message": self.message,
            "final_answer": self.final_answer,
            "pending_call_id": self.pending_call_id,
            "pending_execution_fingerprint": self.pending_execution_fingerprint,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "events": [_event_dict(event) for event in self.state.events],
        }


class Planner(Protocol):
    def decide(
        self, state: AgentLoopState, remaining: RemainingBudget
    ) -> PlannerDecision: ...


class CompletionVerifier(Protocol):
    def verify(
        self, state: AgentLoopState, proposal: FinishProposal
    ) -> VerificationResult: ...


class ScriptedPlanner:
    """Deterministic offline planner fixture; it performs no model call."""

    def __init__(
        self, decisions: Sequence[PlannerDecision], *, start_index: int = 0
    ) -> None:
        if not decisions:
            raise ValueError("scripted planner requires at least one decision")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or start_index < 0
            or start_index > len(decisions)
        ):
            raise ValueError("scripted planner start_index is out of range")
        self._decisions = tuple(decisions)
        self._index = start_index

    def decide(
        self, state: AgentLoopState, remaining: RemainingBudget
    ) -> PlannerDecision:
        if self._index >= len(self._decisions):
            raise RuntimeError("scripted planner exhausted before a terminal action")
        decision = self._decisions[self._index]
        self._index += 1
        return decision


def run_agent_loop(
    *,
    runtime: AgentRuntime,
    planner: Planner,
    verifier: CompletionVerifier,
    context: ExecutionContext,
    budget: LoopBudget,
    approvals: Mapping[str, ApprovalGrant] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AgentLoopReport:
    """Run until verified completion, safe pause/escalation, or hard stop."""

    return _run_agent_loop(
        runtime=runtime,
        planner=planner,
        verifier=verifier,
        context=context,
        budget=budget,
        approvals=approvals,
        clock=clock,
        checkpoint=None,
    )


def resume_agent_loop(
    *,
    checkpoint: AgentLoopCheckpoint,
    runtime: AgentRuntime,
    planner: Planner,
    verifier: CompletionVerifier,
    context: ExecutionContext,
    approval: ApprovalGrant,
    clock: Callable[[], float] = time.monotonic,
) -> AgentLoopReport:
    """Resume one pending decision without replanning or recharging its usage."""

    if not isinstance(checkpoint, AgentLoopCheckpoint):
        raise ValueError("checkpoint must be AgentLoopCheckpoint")
    if not isinstance(approval, ApprovalGrant):
        raise ValueError("approval must be ApprovalGrant")
    if (
        context.task_id != checkpoint.state.task_id
        or context.subject_id != checkpoint.subject_id
        or context.tenant_id != checkpoint.tenant_id
    ):
        raise ValueError("resume context identity does not match checkpoint")
    if runtime.executed_tool_calls != checkpoint.runtime_executed_tool_calls:
        raise ValueError(
            "runtime tool-call counter must be restored from checkpoint before resume"
        )
    if runtime.max_tool_calls != checkpoint.runtime_max_tool_calls:
        raise ValueError("runtime max_tool_calls does not match checkpoint")
    pending_action = checkpoint.pending_decision.action
    if not isinstance(pending_action, ToolProposal):
        raise ValueError("checkpoint pending decision is not a tool proposal")
    return _run_agent_loop(
        runtime=runtime,
        planner=planner,
        verifier=verifier,
        context=context,
        budget=checkpoint.budget,
        approvals={pending_action.call.call_id: approval},
        clock=clock,
        checkpoint=checkpoint,
    )


def _run_agent_loop(
    *,
    runtime: AgentRuntime,
    planner: Planner,
    verifier: CompletionVerifier,
    context: ExecutionContext,
    budget: LoopBudget,
    approvals: Mapping[str, ApprovalGrant] | None,
    clock: Callable[[], float],
    checkpoint: AgentLoopCheckpoint | None,
) -> AgentLoopReport:
    """Shared fresh/resumed loop engine."""

    if not callable(clock):
        raise ValueError("clock must be callable")
    started_at = _read_clock(clock)
    state = (
        AgentLoopState(task_id=context.task_id)
        if checkpoint is None
        else checkpoint.state
    )
    model_tokens_used = 0 if checkpoint is None else checkpoint.model_tokens_used
    cost_units_used = Decimal(
        0 if checkpoint is None else str(checkpoint.cost_units_used)
    )
    elapsed_before_run = (
        0.0 if checkpoint is None else checkpoint.active_wall_time_seconds
    )
    max_cost_units = Decimal(str(budget.max_cost_units))
    initial_runtime_tool_calls = runtime.executed_tool_calls
    handler_attempts_before_run = (
        0 if checkpoint is None else checkpoint.handler_attempts_used
    )
    action_history = list(
        () if checkpoint is None else checkpoint.action_history
    )
    consecutive_error_key = (
        None if checkpoint is None else checkpoint.consecutive_error_key
    )
    consecutive_error_count = (
        0 if checkpoint is None else checkpoint.consecutive_error_count
    )
    pending_decision = None if checkpoint is None else checkpoint.pending_decision
    approval_map = approvals if approvals is not None else {}

    def current_elapsed() -> float:
        return elapsed_before_run + _elapsed(clock, started_at)

    def handler_attempts_used() -> int:
        return handler_attempts_before_run + (
            runtime.executed_tool_calls - initial_runtime_tool_calls
        )

    while True:
        elapsed = current_elapsed()
        if elapsed >= budget.max_wall_time_seconds:
            return _report(
                LoopTermination.WALL_TIME_BUDGET,
                state,
                model_tokens_used,
                cost_units_used,
                elapsed,
                handler_attempts_used(),
            )
        resumed_decision = pending_decision
        replacing_pending = resumed_decision is not None
        decision: PlannerDecision
        if resumed_decision is not None:
            decision = resumed_decision
            pending_decision = None
            action_fingerprint = _action_fingerprint(decision.action)
        else:
            if len(state.events) >= budget.max_steps:
                return _report(
                    LoopTermination.STEP_BUDGET,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    elapsed,
                    handler_attempts_used(),
                )
            remaining = RemainingBudget(
                steps=budget.max_steps - len(state.events),
                model_tokens=budget.max_model_tokens - model_tokens_used,
                cost_units=float(max_cost_units - cost_units_used),
                wall_time_seconds=max(0.0, budget.max_wall_time_seconds - elapsed),
            )
            try:
                decision = planner.decide(state, remaining)
            except Exception as error:
                return _report(
                    LoopTermination.PLANNER_ERROR,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    current_elapsed(),
                    handler_attempts_used(),
                    message=f"{type(error).__name__}: {error}",
                )
            if not isinstance(decision, PlannerDecision):
                return _report(
                    LoopTermination.PLANNER_ERROR,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    current_elapsed(),
                    handler_attempts_used(),
                    message="TypeError: planner must return PlannerDecision",
                )
            model_tokens_used += decision.total_tokens
            cost_units_used += Decimal(str(decision.cost_units))
            action_fingerprint = _action_fingerprint(decision.action)
            action_history.append(action_fingerprint)

            if model_tokens_used > budget.max_model_tokens:
                state = _append_budget_event(
                    state, decision, action_fingerprint, "model_token_budget_exceeded"
                )
                return _report(
                    LoopTermination.MODEL_TOKEN_BUDGET,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    current_elapsed(),
                    handler_attempts_used(),
                )
            if cost_units_used > max_cost_units:
                state = _append_budget_event(
                    state, decision, action_fingerprint, "cost_budget_exceeded"
                )
                return _report(
                    LoopTermination.COST_BUDGET,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    current_elapsed(),
                    handler_attempts_used(),
                )
            elapsed = current_elapsed()
            if elapsed >= budget.max_wall_time_seconds:
                state = _append_budget_event(
                    state, decision, action_fingerprint, "wall_time_budget_exceeded"
                )
                return _report(
                    LoopTermination.WALL_TIME_BUDGET,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    elapsed,
                    handler_attempts_used(),
                )

        error_key: str | None = None
        if isinstance(decision.action, ToolProposal):
            call = decision.action.call
            attempts_before = runtime.executed_tool_calls
            try:
                outcome = runtime.execute(
                    call,
                    context=context,
                    approval=approval_map.get(call.call_id),
                )
            except Exception as error:
                error_event = LoopEvent(
                    step=(
                        len(state.events)
                        if replacing_pending
                        else len(state.events) + 1
                    ),
                    decision_id=decision.decision_id,
                    model_revision=decision.model_revision,
                    action_kind="tool",
                    action_fingerprint=action_fingerprint,
                    status="runtime_error",
                    message=f"{type(error).__name__}: {error}",
                    call_id=call.call_id,
                    proposal_fingerprint=call.fingerprint(),
                )
                state = (
                    _replace_pending_event(state, error_event)
                    if replacing_pending
                    else AgentLoopState(state.task_id, (*state.events, error_event))
                )
                return _report(
                    LoopTermination.RUNTIME_ERROR,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    current_elapsed(),
                    handler_attempts_used(),
                    message=f"{type(error).__name__}: {error}",
                )
            handler_attempted = runtime.executed_tool_calls > attempts_before
            event = LoopEvent(
                step=len(state.events) if replacing_pending else len(state.events) + 1,
                decision_id=decision.decision_id,
                model_revision=decision.model_revision,
                action_kind="tool",
                action_fingerprint=action_fingerprint,
                status=outcome.status.value,
                message=outcome.message,
                call_id=call.call_id,
                proposal_fingerprint=call.fingerprint(),
                execution_fingerprint=outcome.execution_fingerprint,
                handler_attempted=handler_attempted,
                value=outcome.value,
            )
            state = (
                _replace_pending_event(state, event)
                if replacing_pending
                else AgentLoopState(state.task_id, (*state.events, event))
            )
            if outcome.status is ExecutionStatus.NEEDS_APPROVAL:
                elapsed = current_elapsed()
                if elapsed >= budget.max_wall_time_seconds:
                    return _report(
                        LoopTermination.WALL_TIME_BUDGET,
                        state,
                        model_tokens_used,
                        cost_units_used,
                        elapsed,
                        handler_attempts_used(),
                    )
                new_checkpoint = AgentLoopCheckpoint(
                    state=state,
                    subject_id=context.subject_id,
                    tenant_id=context.tenant_id,
                    budget=budget,
                    model_tokens_used=model_tokens_used,
                    cost_units_used=float(cost_units_used),
                    active_wall_time_seconds=elapsed,
                    handler_attempts_used=handler_attempts_used(),
                    runtime_executed_tool_calls=runtime.executed_tool_calls,
                    runtime_max_tool_calls=runtime.max_tool_calls,
                    action_history=tuple(action_history),
                    consecutive_error_key=consecutive_error_key,
                    consecutive_error_count=consecutive_error_count,
                    pending_decision=decision,
                    pending_execution_fingerprint=outcome.execution_fingerprint,
                )
                return _report(
                    LoopTermination.NEEDS_APPROVAL,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    elapsed,
                    handler_attempts_used(),
                    pending_call_id=call.call_id,
                    pending_execution_fingerprint=outcome.execution_fingerprint,
                    checkpoint=new_checkpoint,
                )
            if (
                replacing_pending
                and outcome.status is ExecutionStatus.APPROVAL_REJECTED
            ):
                return _report(
                    LoopTermination.APPROVAL_REJECTED,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    current_elapsed(),
                    handler_attempts_used(),
                    message=outcome.message,
                    checkpoint=checkpoint,
                )
            elapsed = current_elapsed()
            if elapsed >= budget.max_wall_time_seconds:
                return _report(
                    LoopTermination.WALL_TIME_BUDGET,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    elapsed,
                    handler_attempts_used(),
                )
            if outcome.status in {
                ExecutionStatus.FAILED,
                ExecutionStatus.POLICY_DENIED,
                ExecutionStatus.APPROVAL_REJECTED,
            }:
                error_key = _runtime_error_key(outcome.status, outcome.message)
        elif isinstance(decision.action, FinishProposal):
            try:
                verification = verifier.verify(state, decision.action)
            except Exception as error:
                state = _append_simple_event(
                    state,
                    decision,
                    action_fingerprint,
                    "finish",
                    "verifier_error",
                    f"{type(error).__name__}: {error}",
                )
                return _report(
                    LoopTermination.VERIFIER_ERROR,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    current_elapsed(),
                    handler_attempts_used(),
                    message=f"{type(error).__name__}: {error}",
                )
            if not isinstance(verification, VerificationResult):
                state = _append_simple_event(
                    state,
                    decision,
                    action_fingerprint,
                    "finish",
                    "verifier_error",
                    "TypeError: completion verifier must return VerificationResult",
                )
                return _report(
                    LoopTermination.VERIFIER_ERROR,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    current_elapsed(),
                    handler_attempts_used(),
                    message=(
                        "TypeError: completion verifier must return VerificationResult"
                    ),
                )
            state = AgentLoopState(
                state.task_id,
                (
                    *state.events,
                    LoopEvent(
                        step=len(state.events) + 1,
                        decision_id=decision.decision_id,
                        model_revision=decision.model_revision,
                        action_kind="finish",
                        action_fingerprint=action_fingerprint,
                        status=verification.status.value,
                        message=verification.reason_code,
                        verification=verification,
                    ),
                ),
            )
            elapsed = current_elapsed()
            if elapsed >= budget.max_wall_time_seconds:
                return _report(
                    LoopTermination.WALL_TIME_BUDGET,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    elapsed,
                    handler_attempts_used(),
                )
            if verification.status is VerificationStatus.PASSED:
                return _report(
                    LoopTermination.COMPLETED,
                    state,
                    model_tokens_used,
                    cost_units_used,
                    current_elapsed(),
                    handler_attempts_used(),
                    final_answer=decision.action.answer,
                )
            error_key = (
                f"verification:{verification.status.value}:{verification.reason_code}"
            )
        else:
            state = _append_simple_event(
                state,
                decision,
                action_fingerprint,
                "escalate",
                "escalated",
                decision.action.reason_code,
            )
            return _report(
                LoopTermination.ESCALATED,
                state,
                model_tokens_used,
                cost_units_used,
                current_elapsed(),
                handler_attempts_used(),
                message=decision.action.message,
            )

        termination = _action_loop_termination(
            action_history, budget.repeated_action_limit
        )
        if termination is not None:
            return _report(
                termination,
                state,
                model_tokens_used,
                cost_units_used,
                current_elapsed(),
                handler_attempts_used(),
            )
        if error_key is None:
            consecutive_error_key = None
            consecutive_error_count = 0
        elif error_key == consecutive_error_key:
            consecutive_error_count += 1
        else:
            consecutive_error_key = error_key
            consecutive_error_count = 1
        if consecutive_error_count >= budget.repeated_error_limit:
            return _report(
                LoopTermination.REPEATED_ERROR,
                state,
                model_tokens_used,
                cost_units_used,
                current_elapsed(),
                handler_attempts_used(),
            )


def _action_fingerprint(action: AgentAction) -> str:
    if isinstance(action, ToolProposal):
        return action.call.fingerprint()
    if isinstance(action, FinishProposal):
        components: dict[str, object] = {
            "kind": "finish",
            "answer": action.answer,
            "evidence_ids": list(action.evidence_ids),
        }
    else:
        components = {
            "kind": "escalate",
            "reason_code": action.reason_code,
            "message": action.message,
        }
    return "sha256:" + artifact_fingerprint(components)


def _action_loop_termination(
    history: Sequence[str], repeated_action_limit: int
) -> LoopTermination | None:
    if len(history) >= repeated_action_limit and len(
        set(history[-repeated_action_limit:])
    ) == 1:
        return LoopTermination.REPEATED_ACTION
    if (
        len(history) >= 4
        and history[-4] == history[-2]
        and history[-3] == history[-1]
        and history[-4] != history[-3]
    ):
        return LoopTermination.ACTION_CYCLE
    return None


def _runtime_error_key(status: ExecutionStatus, message: str) -> str:
    return f"runtime:{status.value}:{message}"


def _append_budget_event(
    state: AgentLoopState,
    decision: PlannerDecision,
    action_fingerprint: str,
    reason: str,
) -> AgentLoopState:
    return _append_simple_event(
        state,
        decision,
        action_fingerprint,
        _action_kind(decision.action),
        "budget_rejected",
        reason,
    )


def _append_simple_event(
    state: AgentLoopState,
    decision: PlannerDecision,
    action_fingerprint: str,
    action_kind: str,
    status: str,
    message: str,
) -> AgentLoopState:
    event = LoopEvent(
        step=len(state.events) + 1,
        decision_id=decision.decision_id,
        model_revision=decision.model_revision,
        action_kind=action_kind,
        action_fingerprint=action_fingerprint,
        status=status,
        message=message,
    )
    return AgentLoopState(state.task_id, (*state.events, event))


def _replace_pending_event(state: AgentLoopState, event: LoopEvent) -> AgentLoopState:
    if not state.events or state.events[-1].status != ExecutionStatus.NEEDS_APPROVAL.value:
        raise ValueError("resume state does not end in a pending approval event")
    if event.step != state.events[-1].step:
        raise ValueError("resumed event must preserve the pending decision step")
    return AgentLoopState(state.task_id, (*state.events[:-1], event))


def _action_kind(action: AgentAction) -> str:
    if isinstance(action, ToolProposal):
        return "tool"
    if isinstance(action, FinishProposal):
        return "finish"
    return "escalate"


def _event_dict(event: LoopEvent) -> dict[str, Any]:
    verification = event.verification
    return {
        "step": event.step,
        "decision_id": event.decision_id,
        "model_revision": event.model_revision,
        "action_kind": event.action_kind,
        "action_fingerprint": event.action_fingerprint,
        "status": event.status,
        "message": event.message,
        "call_id": event.call_id,
        "proposal_fingerprint": event.proposal_fingerprint,
        "execution_fingerprint": event.execution_fingerprint,
        "handler_attempted": event.handler_attempted,
        "value": event.value,
        "verification": (
            {
                "status": verification.status.value,
                "verifier_version": verification.verifier_version,
                "reason_code": verification.reason_code,
            }
            if verification is not None
            else None
        ),
    }


def _checkpoint_payload(checkpoint: AgentLoopCheckpoint) -> dict[str, Any]:
    return {
        "schema_version": checkpoint.schema_version,
        "context_identity": {
            "task_id": checkpoint.state.task_id,
            "subject_id": checkpoint.subject_id,
            "tenant_id": checkpoint.tenant_id,
        },
        "budget": {
            "max_steps": checkpoint.budget.max_steps,
            "max_model_tokens": checkpoint.budget.max_model_tokens,
            "max_cost_units": checkpoint.budget.max_cost_units,
            "max_wall_time_seconds": checkpoint.budget.max_wall_time_seconds,
            "repeated_action_limit": checkpoint.budget.repeated_action_limit,
            "repeated_error_limit": checkpoint.budget.repeated_error_limit,
        },
        "usage": {
            "model_tokens_used": checkpoint.model_tokens_used,
            "cost_units_used": checkpoint.cost_units_used,
            "active_wall_time_seconds": checkpoint.active_wall_time_seconds,
            "handler_attempts_used": checkpoint.handler_attempts_used,
            "runtime_executed_tool_calls": checkpoint.runtime_executed_tool_calls,
            "runtime_max_tool_calls": checkpoint.runtime_max_tool_calls,
        },
        "loop_control": {
            "action_history": list(checkpoint.action_history),
            "consecutive_error_key": checkpoint.consecutive_error_key,
            "consecutive_error_count": checkpoint.consecutive_error_count,
        },
        "state": {
            "task_id": checkpoint.state.task_id,
            "events": [_event_dict(event) for event in checkpoint.state.events],
        },
        "pending_decision": _decision_dict(checkpoint.pending_decision),
        "pending_execution_fingerprint": checkpoint.pending_execution_fingerprint,
    }


def load_agent_loop_checkpoint(value: Mapping[str, Any]) -> AgentLoopCheckpoint:
    """Parse and internally verify one strict-JSON checkpoint mapping."""

    try:
        snapshot = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint must be strict JSON: {error}") from error
    record = _record(snapshot, "checkpoint")
    _expect_fields(
        record,
        {
            "schema_version",
            "context_identity",
            "budget",
            "usage",
            "loop_control",
            "state",
            "pending_decision",
            "pending_execution_fingerprint",
            "checkpoint_fingerprint",
        },
        "checkpoint",
    )
    if record["schema_version"] != 1:
        raise ValueError("unsupported Agent loop checkpoint schema version")
    context = _record(record["context_identity"], "checkpoint.context_identity")
    _expect_fields(
        context, {"task_id", "subject_id", "tenant_id"}, "checkpoint.context_identity"
    )
    budget_record = _record(record["budget"], "checkpoint.budget")
    _expect_fields(
        budget_record,
        {
            "max_steps",
            "max_model_tokens",
            "max_cost_units",
            "max_wall_time_seconds",
            "repeated_action_limit",
            "repeated_error_limit",
        },
        "checkpoint.budget",
    )
    budget = LoopBudget(
        max_steps=budget_record["max_steps"],
        max_model_tokens=budget_record["max_model_tokens"],
        max_cost_units=budget_record["max_cost_units"],
        max_wall_time_seconds=budget_record["max_wall_time_seconds"],
        repeated_action_limit=budget_record["repeated_action_limit"],
        repeated_error_limit=budget_record["repeated_error_limit"],
    )
    usage = _record(record["usage"], "checkpoint.usage")
    _expect_fields(
        usage,
        {
            "model_tokens_used",
            "cost_units_used",
            "active_wall_time_seconds",
            "handler_attempts_used",
            "runtime_executed_tool_calls",
            "runtime_max_tool_calls",
        },
        "checkpoint.usage",
    )
    control = _record(record["loop_control"], "checkpoint.loop_control")
    _expect_fields(
        control,
        {"action_history", "consecutive_error_key", "consecutive_error_count"},
        "checkpoint.loop_control",
    )
    history_raw = control["action_history"]
    if not isinstance(history_raw, list) or any(
        not isinstance(item, str) for item in history_raw
    ):
        raise ValueError("checkpoint.loop_control.action_history must be string array")
    error_key = control["consecutive_error_key"]
    if error_key is not None and not isinstance(error_key, str):
        raise ValueError("checkpoint consecutive_error_key must be string or null")
    state_record = _record(record["state"], "checkpoint.state")
    _expect_fields(state_record, {"task_id", "events"}, "checkpoint.state")
    events_raw = state_record["events"]
    if not isinstance(events_raw, list):
        raise ValueError("checkpoint.state.events must be an array")
    state = AgentLoopState(
        task_id=_string(state_record["task_id"], "checkpoint.state.task_id"),
        events=tuple(
            _event_from_dict(item, index)
            for index, item in enumerate(events_raw, start=1)
        ),
    )
    checkpoint = AgentLoopCheckpoint(
        state=state,
        subject_id=_string(context["subject_id"], "checkpoint.subject_id"),
        tenant_id=_string(context["tenant_id"], "checkpoint.tenant_id"),
        budget=budget,
        model_tokens_used=usage["model_tokens_used"],
        cost_units_used=usage["cost_units_used"],
        active_wall_time_seconds=usage["active_wall_time_seconds"],
        handler_attempts_used=usage["handler_attempts_used"],
        runtime_executed_tool_calls=usage["runtime_executed_tool_calls"],
        runtime_max_tool_calls=usage["runtime_max_tool_calls"],
        action_history=tuple(history_raw),
        consecutive_error_key=error_key,
        consecutive_error_count=control["consecutive_error_count"],
        pending_decision=_decision_from_dict(record["pending_decision"]),
        pending_execution_fingerprint=_string(
            record["pending_execution_fingerprint"],
            "checkpoint.pending_execution_fingerprint",
        ),
        schema_version=record["schema_version"],
    )
    supplied_fingerprint = _string(
        record["checkpoint_fingerprint"], "checkpoint.checkpoint_fingerprint"
    )
    if supplied_fingerprint != checkpoint.fingerprint:
        raise ValueError("checkpoint fingerprint mismatch")
    if _string(context["task_id"], "checkpoint.context_identity.task_id") != state.task_id:
        raise ValueError("checkpoint context/state task identity mismatch")
    return checkpoint


def _decision_dict(decision: PlannerDecision) -> dict[str, Any]:
    action = decision.action
    if isinstance(action, ToolProposal):
        action_payload: dict[str, Any] = {
            "kind": "tool",
            "call_id": action.call.call_id,
            "tool_name": action.call.tool_name,
            "arguments": action.call.arguments,
        }
    elif isinstance(action, FinishProposal):
        action_payload = {
            "kind": "finish",
            "answer": action.answer,
            "evidence_ids": list(action.evidence_ids),
        }
    else:
        action_payload = {
            "kind": "escalate",
            "reason_code": action.reason_code,
            "message": action.message,
        }
    return {
        "decision_id": decision.decision_id,
        "model_revision": decision.model_revision,
        "input_tokens": decision.input_tokens,
        "output_tokens": decision.output_tokens,
        "cost_units": decision.cost_units,
        "action": action_payload,
    }


def _decision_from_dict(value: Any) -> PlannerDecision:
    record = _record(value, "checkpoint.pending_decision")
    _expect_fields(
        record,
        {
            "decision_id",
            "model_revision",
            "input_tokens",
            "output_tokens",
            "cost_units",
            "action",
        },
        "checkpoint.pending_decision",
    )
    action_record = _record(record["action"], "checkpoint.pending_decision.action")
    kind = action_record.get("kind")
    if kind != "tool":
        raise ValueError("checkpoint pending action must be a tool proposal")
    _expect_fields(
        action_record,
        {"kind", "call_id", "tool_name", "arguments"},
        "checkpoint.pending_decision.action",
    )
    arguments = _record(
        action_record["arguments"], "checkpoint.pending_decision.action.arguments"
    )
    return PlannerDecision(
        decision_id=_string(record["decision_id"], "checkpoint decision_id"),
        model_revision=_string(record["model_revision"], "checkpoint model_revision"),
        action=ToolProposal(
            ToolCall(
                _string(action_record["call_id"], "checkpoint call_id"),
                _string(action_record["tool_name"], "checkpoint tool_name"),
                arguments,
            )
        ),
        input_tokens=record["input_tokens"],
        output_tokens=record["output_tokens"],
        cost_units=record["cost_units"],
    )


def _event_from_dict(value: Any, index: int) -> LoopEvent:
    prefix = f"checkpoint.state.events[{index}]"
    record = _record(value, prefix)
    _expect_fields(
        record,
        {
            "step",
            "decision_id",
            "model_revision",
            "action_kind",
            "action_fingerprint",
            "status",
            "message",
            "call_id",
            "proposal_fingerprint",
            "execution_fingerprint",
            "handler_attempted",
            "value",
            "verification",
        },
        prefix,
    )
    verification_raw = record["verification"]
    verification: VerificationResult | None = None
    if verification_raw is not None:
        verification_record = _record(verification_raw, f"{prefix}.verification")
        _expect_fields(
            verification_record,
            {"status", "verifier_version", "reason_code"},
            f"{prefix}.verification",
        )
        try:
            status = VerificationStatus(verification_record["status"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{prefix}.verification.status is invalid") from error
        verification = VerificationResult(
            status,
            _string(verification_record["verifier_version"], "verifier_version"),
            _string(verification_record["reason_code"], "reason_code"),
        )
    return LoopEvent(
        step=record["step"],
        decision_id=_string(record["decision_id"], f"{prefix}.decision_id"),
        model_revision=_string(
            record["model_revision"], f"{prefix}.model_revision"
        ),
        action_kind=_string(record["action_kind"], f"{prefix}.action_kind"),
        action_fingerprint=_string(
            record["action_fingerprint"], f"{prefix}.action_fingerprint"
        ),
        status=_string(record["status"], f"{prefix}.status"),
        message=_string_allow_empty(record["message"], f"{prefix}.message"),
        call_id=_optional_string(record["call_id"], f"{prefix}.call_id"),
        proposal_fingerprint=_optional_string(
            record["proposal_fingerprint"], f"{prefix}.proposal_fingerprint"
        ),
        execution_fingerprint=_optional_string(
            record["execution_fingerprint"], f"{prefix}.execution_fingerprint"
        ),
        handler_attempted=record["handler_attempted"],
        value=record["value"],
        verification=verification,
    )


def _record(value: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} must be a JSON object")
    return value


def _expect_fields(record: Mapping[str, Any], expected: set[str], prefix: str) -> None:
    missing = expected - set(record)
    unknown = set(record) - expected
    if missing or unknown:
        raise ValueError(
            f"{prefix} field mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _string(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix} must be a non-empty string")
    return value


def _string_allow_empty(value: Any, prefix: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{prefix} must be a string")
    return value


def _optional_string(value: Any, prefix: str) -> str | None:
    if value is None:
        return None
    return _string(value, prefix)


def _is_sha256_fingerprint(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    if len(digest) != 64:
        return False
    try:
        int(digest, 16)
    except ValueError:
        return False
    return True


def _read_clock(clock: Callable[[], float]) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("clock must return a finite numeric timestamp")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("clock must return a finite numeric timestamp")
    return result


def _elapsed(clock: Callable[[], float], started_at: float) -> float:
    current = _read_clock(clock)
    if current < started_at:
        raise ValueError("monotonic clock moved backwards")
    return current - started_at


def _report(
    termination: LoopTermination,
    state: AgentLoopState,
    model_tokens_used: int,
    cost_units_used: Decimal,
    wall_time_seconds: float,
    handler_attempts: int,
    *,
    message: str | None = None,
    final_answer: str | None = None,
    pending_call_id: str | None = None,
    pending_execution_fingerprint: str | None = None,
    checkpoint: AgentLoopCheckpoint | None = None,
) -> AgentLoopReport:
    return AgentLoopReport(
        termination=termination,
        state=state,
        steps_used=len(state.events),
        model_tokens_used=model_tokens_used,
        cost_units_used=float(cost_units_used),
        wall_time_seconds=wall_time_seconds,
        handler_attempts=handler_attempts,
        message=message,
        final_answer=final_answer,
        pending_call_id=pending_call_id,
        pending_execution_fingerprint=pending_execution_fingerprint,
        checkpoint=checkpoint,
    )
