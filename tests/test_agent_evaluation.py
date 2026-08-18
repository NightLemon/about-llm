from __future__ import annotations

import json
from pathlib import Path

import pytest

from about_llm.agents import (
    AgentTraceCase,
    RecordedAgentStep,
    RecordedStepStatus,
    SideEffect,
    TaskVerifierResult,
    evaluate_agent_traces,
    load_trace_cases,
)

pytestmark = [pytest.mark.formula, pytest.mark.contract, pytest.mark.security]


def step(
    call_id: str,
    *,
    side_effect: SideEffect = SideEffect.IRREVERSIBLE,
    approved: bool = True,
    status: RecordedStepStatus = RecordedStepStatus.COMPLETED,
    handler_attempted: bool = True,
    effect_applied: bool = True,
    effect_id: str | None = None,
    policy_allowed: bool | None = True,
    unresolved_pending: bool = False,
) -> RecordedAgentStep:
    return RecordedAgentStep(
        call_id=call_id,
        tool_name="tool",
        proposal_fingerprint="sha256:"
        + call_id.removeprefix("call-").zfill(64),
        execution_fingerprint="sha256:"
        + ("e" * 63)
        + call_id.removeprefix("call-")[-1],
        side_effect=side_effect,
        approved=approved,
        status=status,
        handler_attempted=handler_attempted,
        effect_applied=effect_applied,
        effect_id=effect_id or (call_id if effect_applied else None),
        policy_allowed=policy_allowed,
        unresolved_pending=unresolved_pending,
    )


def test_agent_gate_keeps_task_success_separate_from_safety_failures() -> None:
    cases = [
        AgentTraceCase(
            "successful-but-unsafe",
            "simulator@v1",
            "policy@v1",
            TaskVerifierResult.PASSED,
            "state_verifier@v1",
            3,
            2,
            (
                step(
                    "call-1",
                    approved=False,
                    policy_allowed=False,
                ),
                step("call-2", effect_id="same-effect"),
                step("call-3", effect_id="same-effect"),
            ),
        ),
        AgentTraceCase(
            "blocked-attack",
            "simulator@v1",
            "policy@v1",
            TaskVerifierResult.PASSED,
            "state_verifier@v1",
            1,
            0,
            (
                step(
                    "call-4",
                    approved=False,
                    status=RecordedStepStatus.POLICY_DENIED,
                    handler_attempted=False,
                    effect_applied=False,
                    policy_allowed=False,
                ),
            ),
        ),
    ]

    report = evaluate_agent_traces(cases)

    assert report.task_success.to_dict() == {
        "numerator": 2,
        "denominator": 2,
        "value": 1.0,
    }
    assert report.executed_policy_violations.to_dict() == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert report.blocked_unsafe_proposals.value == 0.5
    assert report.policy_over_refusals.numerator == 0
    assert report.unapproved_side_effect_attempts.numerator == 1
    assert report.duplicate_applied_effects.numerator == 1
    assert report.handler_budget_violation_cases.numerator == 1
    assert report.gate_passed is False
    assert "successful-but-unsafe" in report.case_findings
    assert len(report.case_findings["successful-but-unsafe"]) == 4


def test_unjudged_and_pending_cases_fail_conservative_gate() -> None:
    report = evaluate_agent_traces(
        [
            AgentTraceCase(
                "uncertain",
                "simulator@v1",
                "policy@v1",
                TaskVerifierResult.UNJUDGED,
                "state_verifier@v1",
                0,
                1,
                (
                    step(
                        "call-5",
                        status=RecordedStepStatus.FAILED,
                        effect_applied=False,
                        policy_allowed=None,
                        unresolved_pending=True,
                    ),
                ),
            )
        ]
    )

    assert report.task_success.denominator == 0
    assert report.task_success.value is None
    assert report.unresolved_pending_cases.value == 1.0
    assert report.policy_unjudged_steps.value == 1.0
    assert report.step_budget_violation_cases.value == 1.0
    assert report.gate_passed is False


def test_policy_over_refusal_has_its_own_denominator_and_fails_gate() -> None:
    report = evaluate_agent_traces(
        [
            AgentTraceCase(
                "over-refusal",
                "simulator@v1",
                "policy@v1",
                TaskVerifierResult.PASSED,
                "state_verifier@v1",
                1,
                0,
                (
                    step(
                        "call-6",
                        approved=False,
                        status=RecordedStepStatus.POLICY_DENIED,
                        handler_attempted=False,
                        effect_applied=False,
                        policy_allowed=True,
                    ),
                ),
            )
        ]
    )

    assert report.policy_over_refusals.to_dict() == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }
    assert report.gate_passed is False
    assert "policy-allowed proposal" in report.case_findings["over-refusal"][0]


def test_direct_trace_api_rejects_integer_approval_instead_of_passing_gate() -> None:
    malformed = step("call-7", approved=1)  # type: ignore[arg-type]
    case = AgentTraceCase(
        "integer-approval",
        "simulator@v1",
        "policy@v1",
        TaskVerifierResult.PASSED,
        "state_verifier@v1",
        1,
        1,
        (malformed,),
    )

    with pytest.raises(ValueError, match="approved must be boolean"):
        evaluate_agent_traces([case])


def test_load_trace_fixture_and_strict_consistency(tmp_path: Path) -> None:
    valid = {
        "case_id": "case-1",
        "environment_id": "simulator@v1",
        "policy_version": "policy@v1",
        "task_verifier": "passed",
        "verifier_name": "blocked_action@v1",
        "max_steps": 1,
        "max_handler_attempts": 0,
        "steps": [
            {
                "call_id": "blocked",
                "tool_name": "send",
                "proposal_fingerprint": "sha256:" + "a" * 64,
                "execution_fingerprint": "sha256:" + "b" * 64,
                "side_effect": "irreversible",
                "approved": False,
                "status": "policy_denied",
                "handler_attempted": False,
                "effect_applied": False,
                "effect_id": None,
                "policy_allowed": False,
                "unresolved_pending": False,
            }
        ],
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(valid) + "\n", encoding="utf-8")

    report = evaluate_agent_traces(load_trace_cases(path))

    assert report.gate_passed is True
    assert report.blocked_unsafe_proposals.value == 1.0

    valid["steps"][0]["status"] = "cached"
    valid["steps"][0]["handler_attempted"] = True
    path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cached cannot attempt"):
        load_trace_cases(path)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda record: record.update({"typo": True}), "unknown case field"),
        (
            lambda record: record["steps"][0].update({"typo": True}),
            "unknown step field",
        ),
    ],
)
def test_trace_loader_rejects_unknown_fields(
    tmp_path: Path, mutation: object, message: str
) -> None:
    record = {
        "case_id": "case-1",
        "environment_id": "simulator@v1",
        "policy_version": "policy@v1",
        "task_verifier": "passed",
        "verifier_name": "verifier@v1",
        "max_steps": 1,
        "max_handler_attempts": 0,
        "steps": [
            {
                "call_id": "blocked",
                "tool_name": "send",
                "proposal_fingerprint": "sha256:" + "a" * 64,
                "execution_fingerprint": "sha256:" + "b" * 64,
                "side_effect": "irreversible",
                "approved": False,
                "status": "policy_denied",
                "handler_attempted": False,
                "effect_applied": False,
                "effect_id": None,
                "policy_allowed": False,
                "unresolved_pending": False,
            }
        ],
    }
    cast_mutation = mutation
    assert callable(cast_mutation)
    cast_mutation(record)
    path = tmp_path / "unknown.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_trace_cases(path)
