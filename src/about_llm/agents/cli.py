"""Offline safe-agent scenario runner and SQLite reconciliation CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast
from urllib.parse import urlsplit

from about_llm.agents.approval import ApprovalGrant
from about_llm.agents.evaluation import evaluate_agent_traces, load_trace_cases
from about_llm.agents.loop import (
    AgentAction,
    AgentLoopCheckpoint,
    AgentLoopReport,
    AgentLoopState,
    EscalationProposal,
    FinishProposal,
    LoopBudget,
    LoopTermination,
    PlannerDecision,
    ScriptedPlanner,
    ToolProposal,
    VerificationResult,
    VerificationStatus,
    load_agent_loop_checkpoint,
    resume_agent_loop,
    run_agent_loop,
)
from about_llm.agents.policy import (
    CapabilityPolicy,
    ExecutionContext,
    PolicyEffect,
    ResourceRef,
)
from about_llm.agents.runtime import (
    AgentRuntime,
    ExecutionStatus,
    SideEffect,
    Tool,
    ToolCall,
    ToolRegistry,
)
from about_llm.agents.sqlite_ledger import SQLiteLedger
from about_llm.llmops import canonical_json_bytes


@dataclass(frozen=True)
class ScenarioStep:
    """One proposed model action plus the externally supplied approval decision."""

    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    context: ExecutionContext
    approved: bool
    expected_status: ExecutionStatus | None = None
    expected_handler_attempted: bool | None = None
    expected_policy_reason: str | None = None


@dataclass(frozen=True)
class LoopFixtureCase:
    """One deterministic planner-loop case with a local exact-rule verifier."""

    case_id: str
    context: ExecutionContext
    budget: LoopBudget
    decisions: tuple[PlannerDecision, ...]
    expected_termination: LoopTermination
    expected_resume_termination: LoopTermination | None
    expected_answer: str
    required_evidence_ids: tuple[str, ...]


class ExactFixtureVerifier:
    """Verify an exact answer and completed/cached evidence call identities."""

    def __init__(self, expected_answer: str, required_evidence_ids: tuple[str, ...]):
        self.expected_answer = expected_answer
        self.required_evidence_ids = required_evidence_ids

    def verify(
        self, state: AgentLoopState, proposal: FinishProposal
    ) -> VerificationResult:
        available = {
            event.call_id
            for event in state.events
            if event.call_id is not None and event.status in {"completed", "cached"}
        }
        answer_matches = proposal.answer == self.expected_answer
        evidence_matches = proposal.evidence_ids == self.required_evidence_ids
        evidence_available = set(self.required_evidence_ids) <= available
        passed = answer_matches and evidence_matches and evidence_available
        return VerificationResult(
            VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            "offline-exact-fixture@v1",
            "exact_answer_and_evidence_match"
            if passed
            else "exact_answer_or_evidence_mismatch",
        )


def load_scenario(path: Path) -> tuple[ScenarioStep, ...]:
    """Load strict JSONL steps without executing arbitrary code or network calls."""
    steps: list[ScenarioStep] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value: Any = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: step must be a JSON object")
        record = cast(dict[str, Any], value)
        unknown = set(record) - {
            "call_id",
            "tool_name",
            "arguments",
            "context",
            "approved",
            "expected_status",
            "expected_handler_attempted",
            "expected_policy_reason",
        }
        if unknown:
            raise ValueError(
                f"{path}:{line_number}: unknown scenario field(s): {sorted(unknown)}"
            )
        call_id = _required_string(record, "call_id", path, line_number)
        tool_name = _required_string(record, "tool_name", path, line_number)
        arguments = record.get("arguments")
        if not isinstance(arguments, dict) or not all(isinstance(key, str) for key in arguments):
            raise ValueError(f"{path}:{line_number}: arguments must be a JSON object")
        context = _parse_context(record.get("context"), path, line_number)
        approved = record.get("approved", False)
        if not isinstance(approved, bool):
            raise ValueError(f"{path}:{line_number}: approved must be boolean")
        expected_raw = record.get("expected_status")
        try:
            expected = ExecutionStatus(expected_raw) if expected_raw is not None else None
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: unknown expected_status {expected_raw!r}"
            ) from error
        expected_handler_attempted = record.get("expected_handler_attempted")
        if expected_handler_attempted is not None and not isinstance(
            expected_handler_attempted, bool
        ):
            raise ValueError(
                f"{path}:{line_number}: expected_handler_attempted must be boolean"
            )
        expected_policy_reason = record.get("expected_policy_reason")
        if expected_policy_reason is not None and (
            not isinstance(expected_policy_reason, str)
            or not expected_policy_reason.strip()
        ):
            raise ValueError(
                f"{path}:{line_number}: expected_policy_reason must be a non-empty string"
            )
        steps.append(
            ScenarioStep(
                call_id=call_id,
                tool_name=tool_name,
                arguments=cast(dict[str, Any], arguments),
                context=context,
                approved=approved,
                expected_status=expected,
                expected_handler_attempted=expected_handler_attempted,
                expected_policy_reason=expected_policy_reason,
            )
        )
    if not steps:
        raise ValueError(f"{path} contains no scenario steps")
    return tuple(steps)


def demo_registry() -> ToolRegistry:
    """Return deterministic offline tools; none performs a real external action."""
    return ToolRegistry(
        [
            Tool(
                "demo_lookup",
                "offline-fixture@v1",
                "Return a deterministic local demo value.",
                SideEffect.READ_ONLY,
                _validate_lookup,
                lambda arguments: {
                    "key": arguments["key"],
                    "value": f"demo:{arguments['key']}",
                    "simulated": True,
                },
                required_capability="demo:lookup",
                resolve_resource=_resolve_lookup,
            ),
            Tool(
                "simulated_send",
                "offline-fixture@v1",
                "Simulate an irreversible message send without network access.",
                SideEffect.IRREVERSIBLE,
                _validate_send,
                lambda arguments: {
                    "recipient": arguments["recipient"],
                    "accepted": True,
                    "simulated": True,
                },
                required_capability="demo:send",
                resolve_resource=_resolve_send,
            ),
            Tool(
                "uncertain_write",
                "offline-fixture@v1",
                "Simulate a timeout whose external state would be unknown.",
                SideEffect.IRREVERSIBLE,
                _validate_write,
                _raise_uncertain_timeout,
                required_capability="demo:write",
                resolve_resource=_resolve_write,
            ),
        ]
    )


def run_scenario(
    steps: Sequence[ScenarioStep],
    ledger: SQLiteLedger,
    *,
    max_tool_calls: int,
) -> dict[str, Any]:
    """Execute a deterministic scenario and compare optional expected statuses."""
    runtime = AgentRuntime(
        demo_registry(),
        max_tool_calls=max_tool_calls,
        ledger=ledger,
        policy=CapabilityPolicy("offline-capability-policy@v1"),
        clock=lambda: 1_000_000.0,
    )
    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for index, step in enumerate(steps, start=1):
        tool = runtime.registry.get(step.tool_name)
        call = ToolCall(step.call_id, step.tool_name, step.arguments)
        fingerprint = call.fingerprint()
        attempts_before = runtime.executed_tool_calls
        approval: ApprovalGrant | None = None
        if step.approved:
            preflight = runtime.execute(call, context=step.context)
            if preflight.status is ExecutionStatus.NEEDS_APPROVAL:
                approval = ApprovalGrant(
                    approval_id=f"offline-approval-{index}",
                    approver_id="offline-fixture-operator",
                    authorized_subject_id=step.context.subject_id,
                    task_id=step.context.task_id,
                    call_id=step.call_id,
                    execution_fingerprint=preflight.execution_fingerprint,
                    expires_at_epoch_seconds=2_000_000.0,
                )
                outcome = runtime.execute(
                    call,
                    context=step.context,
                    approval=approval,
                )
            else:
                outcome = preflight
        else:
            outcome = runtime.execute(call, context=step.context)
        handler_attempted = runtime.executed_tool_calls > attempts_before
        if step.expected_status is not None and outcome.status is not step.expected_status:
            mismatches.append(
                f"step {index} {step.call_id!r}: expected {step.expected_status.value}, "
                f"got {outcome.status.value}"
            )
        if (
            step.expected_handler_attempted is not None
            and handler_attempted is not step.expected_handler_attempted
        ):
            mismatches.append(
                f"step {index} {step.call_id!r}: expected handler_attempted "
                f"{step.expected_handler_attempted}, got {handler_attempted}"
            )
        if (
            step.expected_policy_reason is not None
            and outcome.policy_decision.reason_code != step.expected_policy_reason
        ):
            mismatches.append(
                f"step {index} {step.call_id!r}: expected policy reason "
                f"{step.expected_policy_reason!r}, got "
                f"{outcome.policy_decision.reason_code!r}"
            )
        ledger_entry = ledger.lookup(step.call_id)
        unresolved_pending = (
            ledger_entry is not None and ledger_entry.state.value == "pending"
        )
        rows.append(
            {
                "step": index,
                "call_id": step.call_id,
                "tool_name": step.tool_name,
                "tool_side_effect": tool.side_effect.value,
                "context": {
                    "task_id": step.context.task_id,
                    "subject_id": step.context.subject_id,
                    "tenant_id": step.context.tenant_id,
                    "capabilities": sorted(step.context.capabilities),
                },
                "approved": step.approved,
                "approval": (
                    {
                        "approval_id": approval.approval_id,
                        "approver_id": approval.approver_id,
                        "expires_at_epoch_seconds": (
                            approval.expires_at_epoch_seconds
                        ),
                        "simulated_unsigned_fixture": True,
                    }
                    if approval is not None
                    else None
                ),
                "handler_attempted": handler_attempted,
                "fingerprint": fingerprint,
                "execution_fingerprint": outcome.execution_fingerprint,
                "status": outcome.status.value,
                "policy": {
                    "effect": outcome.policy_decision.effect.value,
                    "policy_version": outcome.policy_decision.policy_version,
                    "reason_code": outcome.policy_decision.reason_code,
                    "matched_capability": (
                        outcome.policy_decision.matched_capability
                    ),
                },
                "policy_allowed": (
                    outcome.policy_decision.effect is PolicyEffect.ALLOW
                ),
                "resource": {
                    "tenant_id": outcome.resource.tenant_id,
                    "resource_type": outcome.resource.resource_type,
                    "resource_id": outcome.resource.resource_id,
                    "version": outcome.resource.version,
                },
                "unresolved_pending": unresolved_pending,
                "simulated_effect_applied": (
                    handler_attempted
                    and tool.side_effect is not SideEffect.READ_ONLY
                    and outcome.status is ExecutionStatus.COMPLETED
                ),
                "value": outcome.value,
                "message": outcome.message,
            }
        )
    return {
        "passed": not mismatches,
        "simulated_offline": True,
        "handler_attempts": runtime.executed_tool_calls,
        "max_tool_calls": runtime.max_tool_calls,
        "mismatches": mismatches,
        "outcomes": rows,
    }


def load_loop_fixtures(path: Path) -> tuple[LoopFixtureCase, ...]:
    """Load deterministic planner-loop cases from strict JSONL."""

    cases: list[LoopFixtureCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value: Any = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        prefix = f"{path}:{line_number}"
        if not isinstance(value, dict):
            raise ValueError(f"{prefix}: case must be a JSON object")
        record = cast(dict[str, Any], value)
        _reject_unknown_fields(
            record,
            {
                "case_id",
                "context",
                "budget",
                "decisions",
                "verifier",
                "expected_termination",
                "expected_resume_termination",
            },
            prefix,
        )
        decisions_raw = record.get("decisions")
        if not isinstance(decisions_raw, list) or not decisions_raw:
            raise ValueError(f"{prefix}: decisions must be a non-empty array")
        verifier = _parse_fixture_verifier(record.get("verifier"), prefix)
        expected_raw = record.get("expected_termination")
        try:
            expected_termination = LoopTermination(expected_raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{prefix}: unknown expected_termination {expected_raw!r}"
            ) from error
        expected_resume_raw = record.get("expected_resume_termination")
        try:
            expected_resume_termination = (
                LoopTermination(expected_resume_raw)
                if expected_resume_raw is not None
                else None
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{prefix}: unknown expected_resume_termination "
                f"{expected_resume_raw!r}"
            ) from error
        try:
            decisions = tuple(
                _parse_planner_decision(item, prefix, index)
                for index, item in enumerate(decisions_raw, 1)
            )
            case = LoopFixtureCase(
                case_id=_required_string(record, "case_id", path, line_number),
                context=_parse_context(record.get("context"), path, line_number),
                budget=_parse_loop_budget(record.get("budget"), prefix),
                decisions=decisions,
                expected_termination=expected_termination,
                expected_resume_termination=expected_resume_termination,
                expected_answer=verifier[0],
                required_evidence_ids=verifier[1],
            )
        except ValueError as error:
            raise ValueError(f"{prefix}: {error}") from error
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} contains no loop fixture cases")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{path}: case_id values must be unique")
    return tuple(cases)


def run_loop_fixtures(cases: Sequence[LoopFixtureCase]) -> dict[str, Any]:
    """Run scripted local cases; usage fields are supplied fixture observations."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        runtime = _loop_runtime(case)
        report = run_agent_loop(
            runtime=runtime,
            planner=ScriptedPlanner(case.decisions),
            verifier=ExactFixtureVerifier(
                case.expected_answer, case.required_evidence_ids
            ),
            context=case.context,
            budget=case.budget,
            clock=lambda: 0.0,
        )
        matched = report.termination is case.expected_termination
        payload = report.to_dict()
        payload.update(
            {
                "case_id": case.case_id,
                "expected_termination": case.expected_termination.value,
                "expected_resume_termination": (
                    case.expected_resume_termination.value
                    if case.expected_resume_termination is not None
                    else None
                ),
                "matched_expectation": matched,
            }
        )
        rows.append(payload)
    return {
        "passed": all(row["matched_expectation"] for row in rows),
        "simulated_offline": True,
        "scripted_planner": True,
        "provider_usage_measured": False,
        "cases": rows,
    }


def pause_loop_fixture(
    case: LoopFixtureCase, ledger: SQLiteLedger
) -> AgentLoopReport:
    """Run one case to approval pause using a durable execution ledger."""

    report = run_agent_loop(
        runtime=_loop_runtime(case, ledger=ledger),
        planner=ScriptedPlanner(case.decisions),
        verifier=ExactFixtureVerifier(case.expected_answer, case.required_evidence_ids),
        context=case.context,
        budget=case.budget,
        clock=lambda: 0.0,
    )
    if report.termination is not LoopTermination.NEEDS_APPROVAL:
        raise ValueError(
            f"case {case.case_id!r} did not pause for approval: "
            f"{report.termination.value}"
        )
    if report.checkpoint is None:
        raise ValueError("approval pause did not produce a checkpoint")
    return report


def resume_loop_fixture(
    case: LoopFixtureCase,
    checkpoint: AgentLoopCheckpoint,
    ledger: SQLiteLedger,
) -> AgentLoopReport:
    """Resume one offline checkpoint with an explicitly unsigned fixture grant."""

    if checkpoint.state.task_id != case.context.task_id:
        raise ValueError("checkpoint task does not match selected fixture case")
    decision_index = len(checkpoint.state.events) - 1
    if (
        decision_index < 0
        or decision_index >= len(case.decisions)
        or checkpoint.pending_decision != case.decisions[decision_index]
    ):
        raise ValueError("checkpoint pending decision does not match fixture script")
    pending_action = checkpoint.pending_decision.action
    if not isinstance(pending_action, ToolProposal):
        raise ValueError("checkpoint pending decision is not a tool proposal")
    grant = ApprovalGrant(
        approval_id="offline-resume-approval",
        approver_id="offline-fixture-operator",
        authorized_subject_id=case.context.subject_id,
        task_id=case.context.task_id,
        call_id=pending_action.call.call_id,
        execution_fingerprint=checkpoint.pending_execution_fingerprint,
        expires_at_epoch_seconds=2_000_000.0,
    )
    return resume_agent_loop(
        checkpoint=checkpoint,
        runtime=_loop_runtime(
            case,
            ledger=ledger,
            initial_executed_tool_calls=checkpoint.runtime_executed_tool_calls,
        ),
        planner=ScriptedPlanner(
            case.decisions, start_index=len(checkpoint.state.events)
        ),
        verifier=ExactFixtureVerifier(case.expected_answer, case.required_evidence_ids),
        context=case.context,
        approval=grant,
        clock=lambda: 0.0,
    )


def load_loop_checkpoint(path: Path) -> AgentLoopCheckpoint:
    """Load a strict JSON checkpoint file, rejecting duplicate/unknown fields."""

    try:
        value: Any = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid checkpoint JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: checkpoint must be a JSON object")
    return load_agent_loop_checkpoint(cast(dict[str, Any], value))


def write_loop_checkpoint(path: Path, checkpoint: AgentLoopCheckpoint) -> None:
    """Create, but never overwrite, one canonical checkpoint artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(canonical_json_bytes(checkpoint.to_dict()))
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(snapshot, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write("\n")


def _loop_runtime(
    case: LoopFixtureCase,
    *,
    ledger: SQLiteLedger | None = None,
    initial_executed_tool_calls: int = 0,
) -> AgentRuntime:
    return AgentRuntime(
        demo_registry(),
        max_tool_calls=case.budget.max_steps,
        initial_executed_tool_calls=initial_executed_tool_calls,
        ledger=ledger,
        policy=CapabilityPolicy("offline-loop-policy@v1"),
        clock=lambda: 1_000_000.0,
    )


def _find_loop_case(
    cases: Sequence[LoopFixtureCase], case_id: str
) -> LoopFixtureCase:
    matches = [case for case in cases if case.case_id == case_id]
    if len(matches) != 1:
        raise ValueError(f"loop fixture case not found: {case_id!r}")
    return matches[0]


def _parse_loop_budget(value: Any, prefix: str) -> LoopBudget:
    if not isinstance(value, dict):
        raise ValueError("budget must be a JSON object")
    record = cast(dict[str, Any], value)
    fields = {
        "max_steps",
        "max_model_tokens",
        "max_cost_units",
        "max_wall_time_seconds",
        "repeated_action_limit",
        "repeated_error_limit",
    }
    _reject_unknown_fields(record, fields, f"{prefix}.budget")
    missing = fields - set(record)
    if missing:
        raise ValueError(f"budget missing field(s): {sorted(missing)}")
    return LoopBudget(
        max_steps=record["max_steps"],
        max_model_tokens=record["max_model_tokens"],
        max_cost_units=record["max_cost_units"],
        max_wall_time_seconds=record["max_wall_time_seconds"],
        repeated_action_limit=record["repeated_action_limit"],
        repeated_error_limit=record["repeated_error_limit"],
    )


def _parse_fixture_verifier(value: Any, prefix: str) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}: verifier must be a JSON object")
    record = cast(dict[str, Any], value)
    _reject_unknown_fields(
        record, {"expected_answer", "required_evidence_ids"}, f"{prefix}.verifier"
    )
    answer = record.get("expected_answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{prefix}: verifier.expected_answer must be non-empty")
    evidence = _parse_string_array(
        record.get("required_evidence_ids"),
        f"{prefix}: verifier.required_evidence_ids",
    )
    return answer, evidence


def _parse_planner_decision(value: Any, prefix: str, index: int) -> PlannerDecision:
    item_prefix = f"{prefix}: decisions[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{item_prefix} must be a JSON object")
    record = cast(dict[str, Any], value)
    fields = {
        "decision_id",
        "model_revision",
        "input_tokens",
        "output_tokens",
        "cost_units",
        "action",
    }
    _reject_unknown_fields(record, fields, item_prefix)
    missing = fields - set(record)
    if missing:
        raise ValueError(f"{item_prefix} missing field(s): {sorted(missing)}")
    return PlannerDecision(
        decision_id=_nonempty_string(record["decision_id"], f"{item_prefix}.decision_id"),
        model_revision=_nonempty_string(
            record["model_revision"], f"{item_prefix}.model_revision"
        ),
        action=_parse_planner_action(record["action"], item_prefix),
        input_tokens=record["input_tokens"],
        output_tokens=record["output_tokens"],
        cost_units=record["cost_units"],
    )


def _parse_planner_action(value: Any, prefix: str) -> AgentAction:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix}.action must be a JSON object")
    record = cast(dict[str, Any], value)
    kind = record.get("kind")
    if kind == "tool":
        _reject_unknown_fields(
            record, {"kind", "call_id", "tool_name", "arguments"}, f"{prefix}.action"
        )
        arguments = record.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"{prefix}.action.arguments must be a JSON object")
        return ToolProposal(
            ToolCall(
                _nonempty_string(record.get("call_id"), f"{prefix}.action.call_id"),
                _nonempty_string(
                    record.get("tool_name"), f"{prefix}.action.tool_name"
                ),
                cast(dict[str, Any], arguments),
            )
        )
    if kind == "finish":
        _reject_unknown_fields(
            record, {"kind", "answer", "evidence_ids"}, f"{prefix}.action"
        )
        return FinishProposal(
            _nonempty_string(record.get("answer"), f"{prefix}.action.answer"),
            _parse_string_array(
                record.get("evidence_ids"), f"{prefix}.action.evidence_ids"
            ),
        )
    if kind == "escalate":
        _reject_unknown_fields(
            record, {"kind", "reason_code", "message"}, f"{prefix}.action"
        )
        return EscalationProposal(
            _nonempty_string(
                record.get("reason_code"), f"{prefix}.action.reason_code"
            ),
            _nonempty_string(record.get("message"), f"{prefix}.action.message"),
        )
    raise ValueError(f"{prefix}.action.kind is unsupported: {kind!r}")


def _parse_string_array(value: Any, prefix: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{prefix} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{prefix} must not contain duplicates")
    return tuple(value)


def _nonempty_string(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix} must be a non-empty string")
    return value


def _reject_unknown_fields(
    record: Mapping[str, Any], allowed: set[str], prefix: str
) -> None:
    unknown = set(record) - allowed
    if unknown:
        raise ValueError(f"{prefix} contains unknown field(s): {sorted(unknown)}")


def _required_string(
    record: Mapping[str, Any], key: str, path: Path, line_number: int
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: {key} must be a non-empty string")
    return value


def _parse_context(value: Any, path: Path, line_number: int) -> ExecutionContext:
    prefix = f"{path}:{line_number}: context"
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} must be a JSON object")
    record = cast(dict[str, Any], value)
    unknown = set(record) - {"task_id", "subject_id", "tenant_id", "capabilities"}
    if unknown:
        raise ValueError(f"{prefix} contains unknown field(s): {sorted(unknown)}")
    capabilities = record.get("capabilities")
    if not isinstance(capabilities, list) or any(
        not isinstance(item, str) or not item.strip() for item in capabilities
    ):
        raise ValueError(f"{prefix}.capabilities must be an array of non-empty strings")
    if len(capabilities) != len(set(capabilities)):
        raise ValueError(f"{prefix}.capabilities must not contain duplicates")
    return ExecutionContext(
        task_id=_required_context_string(record, "task_id", prefix),
        subject_id=_required_context_string(record, "subject_id", prefix),
        tenant_id=_required_context_string(record, "tenant_id", prefix),
        capabilities=frozenset(capabilities),
    )


def _required_context_string(record: Mapping[str, Any], key: str, prefix: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}.{key} must be a non-empty string")
    return value


def _validate_lookup(arguments: Mapping[str, Any]) -> None:
    _validate_string_fields(arguments, ("key",))


def _validate_send(arguments: Mapping[str, Any]) -> None:
    _validate_string_fields(arguments, ("recipient", "message"))


def _validate_write(arguments: Mapping[str, Any]) -> None:
    _validate_string_fields(arguments, ("resource", "value"))


def _validate_string_fields(arguments: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    if set(arguments) != set(fields) or not all(
        isinstance(arguments[field], str) and arguments[field].strip() for field in fields
    ):
        raise ValueError(f"expected non-empty string fields: {', '.join(fields)}")


def _resolve_lookup(arguments: Mapping[str, Any]) -> ResourceRef:
    return ResourceRef(
        tenant_id="tenant-a",
        resource_type="demo_key",
        resource_id=cast(str, arguments["key"]),
        version="offline-fixture@v1",
    )


def _resolve_send(arguments: Mapping[str, Any]) -> ResourceRef:
    return ResourceRef(
        tenant_id="tenant-a",
        resource_type="simulated_recipient",
        resource_id=cast(str, arguments["recipient"]),
        version="offline-fixture@v1",
    )


def _resolve_write(arguments: Mapping[str, Any]) -> ResourceRef:
    resource = urlsplit(cast(str, arguments["resource"]))
    if resource.scheme != "demo" or not resource.netloc or not resource.path.strip("/"):
        raise ValueError(
            "resource must use demo://<tenant>/<resource-path> in the offline fixture"
        )
    return ResourceRef(
        tenant_id=resource.netloc,
        resource_type="demo_record",
        resource_id=resource.path.strip("/"),
        version="offline-fixture@v1",
    )


def _raise_uncertain_timeout(_: Mapping[str, Any]) -> None:
    raise TimeoutError("simulated timeout; external state would require reconciliation")


def _existing_ledger(path: Path) -> SQLiteLedger:
    if not path.is_file():
        raise ValueError(f"ledger does not exist: {path}")
    return SQLiteLedger(path)


def _entry_payload(ledger: SQLiteLedger, call_id: str) -> dict[str, Any]:
    entry = ledger.lookup(call_id)
    if entry is None:
        raise ValueError(f"unknown call_id {call_id!r}")
    return {
        "call_id": call_id,
        "state": entry.state.value,
        "value": entry.value,
        "reconciliation_history": [
            {
                "resolution": event.resolution,
                "note": event.note,
                "resolved_at": event.resolved_at,
            }
            for event in ledger.reconciliation_history(call_id)
        ],
    }


def _run_scenario_command(args: argparse.Namespace) -> int:
    payload = run_scenario(
        load_scenario(args.scenario),
        SQLiteLedger(args.ledger),
        max_tool_calls=args.max_tool_calls,
    )
    _print_json(payload)
    return 0 if payload["passed"] else 1


def _run_pending(args: argparse.Namespace) -> int:
    pending = _existing_ledger(args.ledger).list_stale_pending(
        older_than_seconds=args.older_than_seconds
    )
    _print_json(
        {
            "pending": [
                {
                    "call_id": item.call_id,
                    "fingerprint": item.fingerprint,
                    "created_at": item.created_at,
                    "age_seconds": item.age_seconds,
                }
                for item in pending
            ]
        }
    )
    return 0


def _run_resolve(args: argparse.Namespace) -> int:
    ledger = _existing_ledger(args.ledger)
    if args.resolution == "external":
        if args.value_json is None:
            raise ValueError("--value-json is required for external completion")
        try:
            value = json.loads(
                args.value_json,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("--value-json must contain strict JSON") from error
        ledger.resolve_external_completion(args.call_id, value, note=args.note)
    else:
        if args.value_json is not None:
            raise ValueError("--value-json is only valid for external completion")
        ledger.resolve_without_completion(
            args.call_id,
            note=args.note,
            compensated=args.resolution == "compensated",
        )
    _print_json(_entry_payload(ledger, args.call_id))
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    _print_json(_entry_payload(_existing_ledger(args.ledger), args.call_id))
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    report = evaluate_agent_traces(load_trace_cases(args.traces))
    payload = report.to_dict()
    payload["recorded_observations_only"] = True
    _print_json(payload)
    return 0 if report.gate_passed else 1


def _run_loop(args: argparse.Namespace) -> int:
    payload = run_loop_fixtures(load_loop_fixtures(args.cases))
    _print_json(payload)
    return 0 if payload["passed"] else 1


def _run_pause_loop(args: argparse.Namespace) -> int:
    case = _find_loop_case(load_loop_fixtures(args.cases), args.case_id)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    report = pause_loop_fixture(case, SQLiteLedger(args.ledger))
    checkpoint = report.checkpoint
    if checkpoint is None:
        raise RuntimeError("pause report unexpectedly omitted checkpoint")
    write_loop_checkpoint(args.checkpoint, checkpoint)
    payload = report.to_dict()
    payload.update(
        {
            "case_id": case.case_id,
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_written_without_overwrite": True,
            "simulated_offline": True,
        }
    )
    _print_json(payload)
    return 0


def _run_resume_loop(args: argparse.Namespace) -> int:
    case = _find_loop_case(load_loop_fixtures(args.cases), args.case_id)
    checkpoint = load_loop_checkpoint(args.checkpoint)
    report = resume_loop_fixture(case, checkpoint, _existing_ledger(args.ledger))
    matched = (
        case.expected_resume_termination is None
        or report.termination is case.expected_resume_termination
    )
    payload = report.to_dict()
    payload.update(
        {
            "case_id": case.case_id,
            "expected_resume_termination": (
                case.expected_resume_termination.value
                if case.expected_resume_termination is not None
                else None
            ),
            "matched_expectation": matched,
            "simulated_offline": True,
            "simulated_unsigned_approval": True,
            "pause_downtime_counted_in_wall_time": False,
        }
    )
    _print_json(payload)
    return 0 if matched else 1


def _print_json(value: Mapping[str, Any]) -> None:
    snapshot = json.loads(canonical_json_bytes(value))
    print(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2))


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="about-llm-agent",
        description="Offline safe-agent execution and SQLite reconciliation lab",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run an offline JSONL scenario")
    run.add_argument("--scenario", type=Path, required=True)
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--max-tool-calls", type=int, default=10)
    run.set_defaults(handler=_run_scenario_command)

    pending = subparsers.add_parser("pending", help="list unresolved pending calls")
    pending.add_argument("--ledger", type=Path, required=True)
    pending.add_argument("--older-than-seconds", type=float, default=0)
    pending.set_defaults(handler=_run_pending)

    resolve = subparsers.add_parser("resolve", help="record an operator reconciliation")
    resolve.add_argument("--ledger", type=Path, required=True)
    resolve.add_argument("--call-id", required=True)
    resolve.add_argument(
        "--resolution", choices=("external", "abandoned", "compensated"), required=True
    )
    resolve.add_argument("--note", required=True)
    resolve.add_argument("--value-json")
    resolve.set_defaults(handler=_run_resolve)

    inspect = subparsers.add_parser("inspect", help="inspect one ledger entry and audit history")
    inspect.add_argument("--ledger", type=Path, required=True)
    inspect.add_argument("--call-id", required=True)
    inspect.set_defaults(handler=_run_inspect)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate recorded trajectory/effect JSONL"
    )
    evaluate.add_argument("--traces", type=Path, required=True)
    evaluate.set_defaults(handler=_run_evaluate)

    loop = subparsers.add_parser(
        "loop", help="run deterministic typed planner-loop JSONL cases"
    )
    loop.add_argument("--cases", type=Path, required=True)
    loop.set_defaults(handler=_run_loop)

    pause_loop = subparsers.add_parser(
        "pause-loop", help="persist one offline approval-pause checkpoint"
    )
    pause_loop.add_argument("--cases", type=Path, required=True)
    pause_loop.add_argument("--case-id", required=True)
    pause_loop.add_argument("--ledger", type=Path, required=True)
    pause_loop.add_argument("--checkpoint", type=Path, required=True)
    pause_loop.set_defaults(handler=_run_pause_loop)

    resume_loop = subparsers.add_parser(
        "resume-loop", help="resume one persisted offline approval checkpoint"
    )
    resume_loop.add_argument("--cases", type=Path, required=True)
    resume_loop.add_argument("--case-id", required=True)
    resume_loop.add_argument("--ledger", type=Path, required=True)
    resume_loop.add_argument("--checkpoint", type=Path, required=True)
    resume_loop.set_defaults(handler=_run_resume_loop)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Callable[[argparse.Namespace], int], args.handler)
    try:
        return handler(args)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
