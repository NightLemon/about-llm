from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

import pytest

from about_llm.agents import (
    AgentLoopState,
    AgentRuntime,
    ApprovalGrant,
    CapabilityPolicy,
    CompletionVerifier,
    EscalationProposal,
    ExecutionContext,
    FinishProposal,
    LoopBudget,
    LoopTermination,
    PlannerDecision,
    ResourceRef,
    ScriptedPlanner,
    SideEffect,
    Tool,
    ToolCall,
    ToolProposal,
    ToolRegistry,
    VerificationResult,
    VerificationStatus,
    load_agent_loop_checkpoint,
    resume_agent_loop,
    run_agent_loop,
)

CONTEXT = ExecutionContext(
    "loop-task", "loop-user", "tenant-a", frozenset({"fixture:read", "fixture:write"})
)


def validate_key(arguments: Mapping[str, Any]) -> None:
    if set(arguments) != {"key"} or not isinstance(arguments["key"], str):
        raise ValueError("expected string key")


def resolve_key(arguments: Mapping[str, Any]) -> ResourceRef:
    return ResourceRef(
        "tenant-a", "fixture_key", str(arguments["key"]), "fixture@v1"
    )


def make_runtime(
    effects: list[str],
    *,
    side_effect: SideEffect = SideEffect.READ_ONLY,
    max_tool_calls: int = 10,
    initial_executed_tool_calls: int = 0,
) -> AgentRuntime:
    capability = "fixture:read" if side_effect is SideEffect.READ_ONLY else "fixture:write"
    tool = Tool(
        "fixture_tool",
        "fixture-tool@v1",
        "Return one deterministic local observation.",
        side_effect,
        validate_key,
        lambda arguments: effects.append(str(arguments["key"]))
        or {"key": arguments["key"], "simulated": True},
        required_capability=capability,
        resolve_resource=resolve_key,
    )
    return AgentRuntime(
        ToolRegistry([tool]),
        max_tool_calls=max_tool_calls,
        initial_executed_tool_calls=initial_executed_tool_calls,
        policy=CapabilityPolicy("loop-policy@v1"),
    )


def tool_decision(index: int, key: str) -> PlannerDecision:
    return PlannerDecision(
        f"decision-{index}",
        "scripted-planner@v1",
        ToolProposal(ToolCall(f"call-{key}", "fixture_tool", {"key": key})),
        input_tokens=2,
        output_tokens=1,
        cost_units=0.1,
    )


def finish_decision(index: int, answer: str) -> PlannerDecision:
    return PlannerDecision(
        f"decision-{index}",
        "scripted-planner@v1",
        FinishProposal(answer, ("fixture-evidence",)),
        input_tokens=2,
        output_tokens=1,
        cost_units=0.1,
    )


def budget(**overrides: object) -> LoopBudget:
    values: dict[str, object] = {
        "max_steps": 10,
        "max_model_tokens": 100,
        "max_cost_units": 10.0,
        "max_wall_time_seconds": 100.0,
        "repeated_action_limit": 3,
        "repeated_error_limit": 3,
    }
    values.update(overrides)
    return LoopBudget(**values)  # type: ignore[arg-type]


class ExactAnswerVerifier:
    def __init__(self, expected: str) -> None:
        self.expected = expected

    def verify(
        self, state: AgentLoopState, proposal: FinishProposal
    ) -> VerificationResult:
        passed = proposal.answer == self.expected and any(
            event.status == "completed" for event in state.events
        )
        return VerificationResult(
            VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            "exact-answer@v1",
            "answer_and_observation_match" if passed else "missing_verified_state",
        )


def run(
    decisions: list[PlannerDecision],
    runtime: AgentRuntime,
    verifier: CompletionVerifier,
    *,
    loop_budget: LoopBudget | None = None,
    context: ExecutionContext = CONTEXT,
    clock: Any = lambda: 0.0,
):
    return run_agent_loop(
        runtime=runtime,
        planner=ScriptedPlanner(decisions),
        verifier=verifier,
        context=context,
        budget=budget() if loop_budget is None else loop_budget,
        clock=clock,
    )


def test_loop_requires_verifier_pass_before_completion() -> None:
    effects: list[str] = []
    report = run(
        [
            tool_decision(1, "evidence"),
            finish_decision(2, "wrong"),
            finish_decision(3, "verified answer"),
        ],
        make_runtime(effects),
        ExactAnswerVerifier("verified answer"),
    )

    assert report.termination is LoopTermination.COMPLETED
    assert report.final_answer == "verified answer"
    assert report.steps_used == 3
    assert report.model_tokens_used == 9
    assert report.cost_units_used == 0.3
    assert report.handler_attempts == 1
    assert report.message is None
    assert report.state.events[1].status == "failed"
    assert report.state.events[2].verification is not None
    assert effects == ["evidence"]


def test_side_effect_without_external_approval_pauses_before_handler() -> None:
    effects: list[str] = []
    report = run(
        [tool_decision(1, "write")],
        make_runtime(effects, side_effect=SideEffect.IRREVERSIBLE),
        ExactAnswerVerifier("unused"),
    )

    assert report.termination is LoopTermination.NEEDS_APPROVAL
    assert report.pending_call_id == "call-write"
    assert report.state.events[0].call_id == "call-write"
    assert report.pending_execution_fingerprint is not None
    assert report.handler_attempts == 0
    assert effects == []


def test_repeated_action_stops_cached_loop() -> None:
    effects: list[str] = []
    report = run(
        [tool_decision(1, "same"), tool_decision(2, "same"), tool_decision(3, "same")],
        make_runtime(effects),
        ExactAnswerVerifier("unused"),
    )

    assert report.termination is LoopTermination.REPEATED_ACTION
    assert [event.status for event in report.state.events] == [
        "completed",
        "cached",
        "cached",
    ]
    assert report.handler_attempts == 1
    assert effects == ["same"]


def test_alternating_action_cycle_is_detected() -> None:
    effects: list[str] = []
    report = run(
        [
            tool_decision(1, "a"),
            tool_decision(2, "b"),
            tool_decision(3, "a"),
            tool_decision(4, "b"),
        ],
        make_runtime(effects),
        ExactAnswerVerifier("unused"),
        loop_budget=budget(repeated_action_limit=5),
    )

    assert report.termination is LoopTermination.ACTION_CYCLE
    assert report.handler_attempts == 2
    assert effects == ["a", "b"]


def test_repeated_policy_error_uses_error_budget_even_for_distinct_actions() -> None:
    effects: list[str] = []
    unauthorized = ExecutionContext("loop-task", "loop-user", "tenant-a", frozenset())
    report = run(
        [tool_decision(1, "a"), tool_decision(2, "b")],
        make_runtime(effects),
        ExactAnswerVerifier("unused"),
        loop_budget=budget(repeated_error_limit=2),
        context=unauthorized,
    )

    assert report.termination is LoopTermination.REPEATED_ERROR
    assert [event.status for event in report.state.events] == [
        "policy_denied",
        "policy_denied",
    ]
    assert report.handler_attempts == 0
    assert effects == []


def test_step_budget_stops_before_requesting_another_planner_decision() -> None:
    effects: list[str] = []
    report = run(
        [tool_decision(1, "a"), tool_decision(2, "b"), tool_decision(3, "c")],
        make_runtime(effects),
        ExactAnswerVerifier("unused"),
        loop_budget=budget(max_steps=2),
    )

    assert report.termination is LoopTermination.STEP_BUDGET
    assert report.steps_used == 2
    assert report.handler_attempts == 2
    assert effects == ["a", "b"]


def test_model_token_and_cost_overruns_do_not_execute_planned_action() -> None:
    effects: list[str] = []
    token_report = run(
        [tool_decision(1, "token")],
        make_runtime(effects),
        ExactAnswerVerifier("unused"),
        loop_budget=budget(max_model_tokens=2),
    )
    cost_report = run(
        [tool_decision(1, "cost")],
        make_runtime(effects),
        ExactAnswerVerifier("unused"),
        loop_budget=budget(max_cost_units=0.05),
    )

    assert token_report.termination is LoopTermination.MODEL_TOKEN_BUDGET
    assert cost_report.termination is LoopTermination.COST_BUDGET
    assert token_report.state.events[0].status == "budget_rejected"
    assert cost_report.state.events[0].status == "budget_rejected"
    assert effects == []


class IncrementingClock:
    def __init__(self) -> None:
        self.value = -1.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def test_wall_time_budget_stops_between_steps() -> None:
    effects: list[str] = []
    report = run(
        [tool_decision(1, "a"), tool_decision(2, "b")],
        make_runtime(effects),
        ExactAnswerVerifier("unused"),
        loop_budget=budget(max_wall_time_seconds=2.5),
        clock=IncrementingClock(),
    )

    assert report.termination is LoopTermination.WALL_TIME_BUDGET
    assert report.steps_used == 1
    assert effects == ["a"]


class InvalidPlanner:
    def decide(self, state: AgentLoopState, remaining: Any) -> PlannerDecision:
        return cast(PlannerDecision, {"not": "a decision"})


class InvalidVerifier:
    def verify(
        self, state: AgentLoopState, proposal: FinishProposal
    ) -> VerificationResult:
        return cast(VerificationResult, {"not": "a verification"})


def test_invalid_extension_results_fail_closed_without_overloading_answer() -> None:
    effects: list[str] = []
    runtime = make_runtime(effects)
    planner_report = run_agent_loop(
        runtime=runtime,
        planner=InvalidPlanner(),
        verifier=ExactAnswerVerifier("unused"),
        context=CONTEXT,
        budget=budget(),
        clock=lambda: 0.0,
    )
    verifier_report = run(
        [finish_decision(1, "answer")], runtime, InvalidVerifier()
    )

    assert planner_report.termination is LoopTermination.PLANNER_ERROR
    assert verifier_report.termination is LoopTermination.VERIFIER_ERROR
    assert planner_report.final_answer is None
    assert verifier_report.final_answer is None
    assert planner_report.message is not None
    assert verifier_report.message is not None


def test_runtime_boundary_exception_becomes_typed_terminal_report() -> None:
    effects: list[str] = []
    decision = PlannerDecision(
        "unknown-tool",
        "scripted-planner@v1",
        ToolProposal(ToolCall("missing-1", "missing_tool", {})),
        input_tokens=0,
        output_tokens=0,
        cost_units=0,
    )

    report = run([decision], make_runtime(effects), ExactAnswerVerifier("unused"))

    assert report.termination is LoopTermination.RUNTIME_ERROR
    assert report.state.events[0].status == "runtime_error"
    assert report.final_answer is None
    assert report.message is not None


def approval_for(report: Any) -> ApprovalGrant:
    assert report.pending_call_id is not None
    assert report.pending_execution_fingerprint is not None
    return ApprovalGrant(
        "approval-1",
        "fixture-operator",
        CONTEXT.subject_id,
        CONTEXT.task_id,
        report.pending_call_id,
        report.pending_execution_fingerprint,
        4_000_000_000.0,
    )


def write_then_finish_decisions() -> list[PlannerDecision]:
    return [
        tool_decision(1, "write"),
        finish_decision(2, "verified answer"),
    ]


def test_checkpoint_round_trip_restart_and_resume_without_double_usage() -> None:
    effects: list[str] = []
    decisions = write_then_finish_decisions()
    paused = run(
        decisions,
        make_runtime(effects, side_effect=SideEffect.IRREVERSIBLE),
        ExactAnswerVerifier("verified answer"),
    )

    assert paused.checkpoint is not None
    checkpoint_payload = paused.checkpoint.to_dict()
    json.dumps(checkpoint_payload, allow_nan=False)
    checkpoint = load_agent_loop_checkpoint(
        json.loads(json.dumps(checkpoint_payload))
    )
    restarted = make_runtime(
        effects,
        side_effect=SideEffect.IRREVERSIBLE,
        initial_executed_tool_calls=checkpoint.runtime_executed_tool_calls,
    )
    resumed = resume_agent_loop(
        checkpoint=checkpoint,
        runtime=restarted,
        planner=ScriptedPlanner(decisions, start_index=len(checkpoint.state.events)),
        verifier=ExactAnswerVerifier("verified answer"),
        context=CONTEXT,
        approval=approval_for(paused),
        clock=lambda: 0.0,
    )

    assert resumed.termination is LoopTermination.COMPLETED
    assert resumed.steps_used == 2
    assert resumed.model_tokens_used == 6
    assert resumed.cost_units_used == 0.2
    assert resumed.handler_attempts == 1
    assert [event.status for event in resumed.state.events] == ["completed", "passed"]
    assert effects == ["write"]


def test_resume_rejects_identity_drift_and_unrestored_runtime_counter() -> None:
    effects: list[str] = []
    side_effect_runtime = make_runtime(effects, side_effect=SideEffect.IRREVERSIBLE)
    approval_pause = run(
        [tool_decision(1, "write")],
        side_effect_runtime,
        ExactAnswerVerifier("unused"),
    )
    checkpoint = approval_pause.checkpoint
    assert checkpoint is not None
    wrong_context = ExecutionContext(
        CONTEXT.task_id, "other-user", CONTEXT.tenant_id, CONTEXT.capabilities
    )
    with pytest.raises(ValueError, match="identity"):
        resume_agent_loop(
            checkpoint=checkpoint,
            runtime=side_effect_runtime,
            planner=ScriptedPlanner([finish_decision(2, "unused")]),
            verifier=ExactAnswerVerifier("unused"),
            context=wrong_context,
            approval=approval_for(approval_pause),
            clock=lambda: 0.0,
        )

    prior_effects: list[str] = []
    mixed_runtime = make_runtime(prior_effects)
    read_report = run(
        [tool_decision(1, "read")],
        mixed_runtime,
        ExactAnswerVerifier("unused"),
        loop_budget=budget(max_steps=1),
    )
    assert read_report.handler_attempts == 1
    pending_runtime = make_runtime(
        prior_effects,
        side_effect=SideEffect.IRREVERSIBLE,
        initial_executed_tool_calls=1,
    )
    pending_report = run(
        [tool_decision(1, "write")],
        pending_runtime,
        ExactAnswerVerifier("unused"),
    )
    pending_checkpoint = pending_report.checkpoint
    assert pending_checkpoint is not None
    with pytest.raises(ValueError, match="counter"):
        resume_agent_loop(
            checkpoint=pending_checkpoint,
            runtime=make_runtime(prior_effects, side_effect=SideEffect.IRREVERSIBLE),
            planner=ScriptedPlanner([finish_decision(2, "unused")]),
            verifier=ExactAnswerVerifier("unused"),
            context=CONTEXT,
            approval=approval_for(pending_report),
            clock=lambda: 0.0,
        )
    with pytest.raises(ValueError, match="max_tool_calls"):
        resume_agent_loop(
            checkpoint=pending_checkpoint,
            runtime=make_runtime(
                prior_effects,
                side_effect=SideEffect.IRREVERSIBLE,
                max_tool_calls=11,
                initial_executed_tool_calls=1,
            ),
            planner=ScriptedPlanner([finish_decision(2, "unused")]),
            verifier=ExactAnswerVerifier("unused"),
            context=CONTEXT,
            approval=approval_for(pending_report),
            clock=lambda: 0.0,
        )


def test_expired_resume_approval_is_typed_terminal_and_retryable() -> None:
    effects: list[str] = []
    paused = run(
        write_then_finish_decisions(),
        make_runtime(effects, side_effect=SideEffect.IRREVERSIBLE),
        ExactAnswerVerifier("verified answer"),
    )
    checkpoint = paused.checkpoint
    assert checkpoint is not None
    pending_action = checkpoint.pending_decision.action
    assert isinstance(pending_action, ToolProposal)
    expired = ApprovalGrant(
        "expired",
        "fixture-operator",
        CONTEXT.subject_id,
        CONTEXT.task_id,
        pending_action.call.call_id,
        checkpoint.pending_execution_fingerprint,
        0.0,
    )
    rejected = resume_agent_loop(
        checkpoint=checkpoint,
        runtime=make_runtime(effects, side_effect=SideEffect.IRREVERSIBLE),
        planner=ScriptedPlanner([finish_decision(2, "verified answer")]),
        verifier=ExactAnswerVerifier("verified answer"),
        context=CONTEXT,
        approval=expired,
        clock=lambda: 0.0,
    )

    assert rejected.termination is LoopTermination.APPROVAL_REJECTED
    assert rejected.checkpoint is checkpoint
    assert rejected.model_tokens_used == paused.model_tokens_used
    assert rejected.handler_attempts == 0
    assert effects == []


def test_resume_reauthorizes_and_preserves_exhausted_handler_budget() -> None:
    effects: list[str] = []
    runtime = make_runtime(
        effects,
        side_effect=SideEffect.IRREVERSIBLE,
        max_tool_calls=1,
        initial_executed_tool_calls=1,
    )
    paused = run(
        [tool_decision(1, "write")],
        runtime,
        ExactAnswerVerifier("unused"),
        loop_budget=budget(max_steps=1),
    )
    checkpoint = paused.checkpoint
    assert checkpoint is not None
    restored = make_runtime(
        effects,
        side_effect=SideEffect.IRREVERSIBLE,
        max_tool_calls=1,
        initial_executed_tool_calls=1,
    )
    exhausted = resume_agent_loop(
        checkpoint=checkpoint,
        runtime=restored,
        planner=ScriptedPlanner([finish_decision(2, "unused")]),
        verifier=ExactAnswerVerifier("unused"),
        context=CONTEXT,
        approval=approval_for(paused),
        clock=lambda: 0.0,
    )
    assert exhausted.termination is LoopTermination.STEP_BUDGET
    assert exhausted.state.events[0].status == "failed"
    assert exhausted.handler_attempts == 0
    assert effects == []

    revoked = ExecutionContext(
        CONTEXT.task_id, CONTEXT.subject_id, CONTEXT.tenant_id, frozenset()
    )
    escalated = resume_agent_loop(
        checkpoint=checkpoint,
        runtime=make_runtime(
            effects,
            side_effect=SideEffect.IRREVERSIBLE,
            max_tool_calls=1,
            initial_executed_tool_calls=1,
        ),
        planner=ScriptedPlanner(
            [
                PlannerDecision(
                    "escalate-after-revocation",
                    "scripted-planner@v1",
                    EscalationProposal("authorization_changed", "operator review"),
                    0,
                    0,
                    0,
                )
            ]
        ),
        verifier=ExactAnswerVerifier("unused"),
        context=revoked,
        approval=approval_for(paused),
        clock=lambda: 0.0,
    )
    assert escalated.termination is LoopTermination.STEP_BUDGET
    assert escalated.state.events[0].status == "policy_denied"
    assert effects == []


def test_checkpoint_loader_rejects_budget_or_pending_decision_tampering() -> None:
    effects: list[str] = []
    paused = run(
        write_then_finish_decisions(),
        make_runtime(effects, side_effect=SideEffect.IRREVERSIBLE),
        ExactAnswerVerifier("verified answer"),
    )
    assert paused.checkpoint is not None
    payload = paused.checkpoint.to_dict()
    payload["budget"]["max_steps"] = 999
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_agent_loop_checkpoint(payload)

    payload = paused.checkpoint.to_dict()
    payload["pending_decision"]["action"]["arguments"]["key"] = "changed"
    with pytest.raises(ValueError):
        load_agent_loop_checkpoint(payload)

    payload = paused.checkpoint.to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="field mismatch"):
        load_agent_loop_checkpoint(payload)


def test_resume_accumulates_active_wall_time_and_stops_before_handler() -> None:
    effects: list[str] = []
    loop_budget = budget(max_wall_time_seconds=1.0)
    paused = run_agent_loop(
        runtime=make_runtime(effects, side_effect=SideEffect.IRREVERSIBLE),
        planner=ScriptedPlanner(write_then_finish_decisions()),
        verifier=ExactAnswerVerifier("verified answer"),
        context=CONTEXT,
        budget=loop_budget,
        clock=SequenceClock([0.0, 0.1, 0.2, 0.3]),
    )
    checkpoint = paused.checkpoint
    assert checkpoint is not None
    assert checkpoint.active_wall_time_seconds == pytest.approx(0.3)

    report = resume_agent_loop(
        checkpoint=checkpoint,
        runtime=make_runtime(effects, side_effect=SideEffect.IRREVERSIBLE),
        planner=ScriptedPlanner([finish_decision(2, "verified answer")]),
        verifier=ExactAnswerVerifier("verified answer"),
        context=CONTEXT,
        approval=approval_for(paused),
        clock=SequenceClock([10.0, 10.8]),
    )

    assert report.termination is LoopTermination.WALL_TIME_BUDGET
    assert report.wall_time_seconds == pytest.approx(1.1)
    assert report.handler_attempts == 0
    assert effects == []
