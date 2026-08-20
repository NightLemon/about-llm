"""Trace one refund task across proposal, authorization, execution, and recovery."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from about_llm.agents import (
    DRAFT_2020_12_URI,
    AgentRuntime,
    ApprovalGrant,
    ExecutionContext,
    ExecutionOutcome,
    ExecutionStatus,
    JSONSchemaToolContract,
    PolicyDecision,
    PolicyEffect,
    PolicyRequest,
    ResourceRef,
    SideEffect,
    SQLiteLedger,
    ToolArgumentValidationError,
    ToolCall,
    ToolRegistry,
)

WALKTHROUGH_VERSION = "about-llm.agent-refund-lifecycle.v1"
TASK_ID = "after-sale-20260820-001"
CALL_ID = "refund-order-1001-attempt-1"
NOW = 1_787_155_200.0


class OrderRecord(TypedDict):
    tenant_id: str
    owner_id: str
    paid_cents: int
    status: str
    version: str


ORDERS: dict[str, OrderRecord] = {
    "order-1001": {
        "tenant_id": "tenant-shop-a",
        "owner_id": "user-42",
        "paid_cents": 30_000,
        "status": "delivered",
        "version": "order@7",
    },
    "order-9001": {
        "tenant_id": "tenant-shop-b",
        "owner_id": "user-99",
        "paid_cents": 30_000,
        "status": "delivered",
        "version": "order@3",
    },
}

REFUND_CONTRACT = JSONSchemaToolContract(
    name="request_refund",
    description="Request a refund for one server-resolved order.",
    schema_revision="refund-arguments@v1",
    arguments_schema={
        "$schema": DRAFT_2020_12_URI,
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "pattern": "^order-[0-9]{4}$",
            },
            "amount_cents": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30_000,
            },
            "reason": {
                "type": "string",
                "enum": ["item_damaged", "not_as_described"],
            },
        },
        "required": ["order_id", "amount_cents", "reason"],
        "additionalProperties": False,
    },
)


class RefundACLPolicy:
    """Resolve tenant, ownership, and exact capability outside model arguments."""

    VERSION = "refund-acl@v1"

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        order = ORDERS.get(request.resource.resource_id)
        if order is None:
            return self._deny("resource_not_found")
        if request.context.tenant_id != request.resource.tenant_id:
            return self._deny("tenant_mismatch")
        if request.context.subject_id != order["owner_id"]:
            return self._deny("owner_mismatch")
        if request.required_capability not in request.context.capabilities:
            return self._deny("missing_capability")
        return PolicyDecision(
            effect=PolicyEffect.ALLOW,
            policy_version=self.VERSION,
            reason_code="refund_capability_and_ownership_granted",
            matched_capability=request.required_capability,
        )

    @classmethod
    def _deny(cls, reason_code: str) -> PolicyDecision:
        return PolicyDecision(
            effect=PolicyEffect.DENY,
            policy_version=cls.VERSION,
            reason_code=reason_code,
        )


@dataclass
class SimulatedRefundProvider:
    """Apply one remote effect, then lose the first response after acceptance."""

    receipts: dict[str, dict[str, Any]] = field(default_factory=dict)
    request_attempts: int = 0
    effect_count: int = 0
    lose_first_response: bool = True

    def request_refund(
        self,
        *,
        idempotency_key: str,
        order_id: str,
        amount_cents: int,
        reason: str,
    ) -> dict[str, Any]:
        self.request_attempts += 1
        receipt = self.receipts.get(idempotency_key)
        if receipt is None:
            receipt = {
                "provider_refund_id": "refund-provider-7001",
                "idempotency_key": idempotency_key,
                "order_id": order_id,
                "amount_cents": amount_cents,
                "reason": reason,
                "provider_status": "accepted",
            }
            self.receipts[idempotency_key] = receipt
            self.effect_count += 1
        if self.lose_first_response:
            self.lose_first_response = False
            raise TimeoutError("provider accepted the refund but the response was lost")
        return dict(receipt)

    def query_refund(self, idempotency_key: str) -> dict[str, Any] | None:
        receipt = self.receipts.get(idempotency_key)
        return dict(receipt) if receipt is not None else None


def _resolve_order(arguments: Mapping[str, Any]) -> ResourceRef:
    order_id = str(arguments["order_id"])
    try:
        order = ORDERS[order_id]
    except KeyError as error:
        raise ValueError("order does not exist in the trusted order store") from error
    return ResourceRef(
        tenant_id=order["tenant_id"],
        resource_type="order",
        resource_id=order_id,
        version=order["version"],
    )


def _make_registry(provider: SimulatedRefundProvider) -> ToolRegistry:
    def handler(arguments: Mapping[str, Any]) -> dict[str, Any]:
        # CALL_ID is assigned by the trusted orchestrator. It is deliberately
        # absent from the model-authored arguments.
        return provider.request_refund(
            idempotency_key=CALL_ID,
            order_id=str(arguments["order_id"]),
            amount_cents=int(arguments["amount_cents"]),
            reason=str(arguments["reason"]),
        )

    tool = REFUND_CONTRACT.build_tool(
        tool_version="refund-tool@v1",
        side_effect=SideEffect.IRREVERSIBLE,
        handler=handler,
        required_capability="refund:request",
        resolve_resource=_resolve_order,
    )
    return ToolRegistry([tool])


def _make_runtime(registry: ToolRegistry, database: Path) -> AgentRuntime:
    return AgentRuntime(
        registry,
        max_tool_calls=2,
        ledger=SQLiteLedger(database),
        policy=RefundACLPolicy(),
        clock=lambda: NOW,
    )


def _outcome_summary(outcome: ExecutionOutcome) -> dict[str, Any]:
    return {
        "status": outcome.status.value,
        "policy_effect": outcome.policy_decision.effect.value,
        "policy_reason": outcome.policy_decision.reason_code,
        "resolved_resource": {
            "tenant_id": outcome.resource.tenant_id,
            "resource_type": outcome.resource.resource_type,
            "resource_id": outcome.resource.resource_id,
            "version": outcome.resource.version,
        },
        "proposal_fingerprint": outcome.call.fingerprint(),
        "execution_fingerprint": outcome.execution_fingerprint,
        "message": outcome.message,
    }


def _verify_provider_effect(receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    expected = {
        "idempotency_key": CALL_ID,
        "order_id": "order-1001",
        "amount_cents": 30_000,
        "reason": "item_damaged",
        "provider_status": "accepted",
    }
    provider_refund_id = receipt.get("provider_refund_id") if receipt else None
    fields_match = receipt is not None and all(
        receipt.get(key) == value for key, value in expected.items()
    )
    receipt_has_identity = isinstance(provider_refund_id, str) and bool(provider_refund_id.strip())
    passed = fields_match and receipt_has_identity
    if passed:
        status = "passed"
        reason = "provider_receipt_matches"
    elif receipt is None:
        status = "indeterminate"
        reason = "provider_state_not_observed"
    else:
        status = "failed"
        reason = "provider_receipt_mismatch"
    return {
        "status": status,
        "verifier_version": "refund-provider-query@v1",
        "reason": reason,
        "expected": expected,
        "provider_refund_id_required": True,
        "observed_receipt": dict(receipt) if receipt is not None else None,
    }


def build_walkthrough(database: Path) -> dict[str, Any]:
    """Run the deterministic lifecycle and return its inspectable evidence."""

    if database.exists():
        raise ValueError("walkthrough database path must not already exist")

    context = ExecutionContext(
        task_id=TASK_ID,
        subject_id="user-42",
        tenant_id="tenant-shop-a",
        capabilities=frozenset({"refund:request"}),
    )
    proposal_arguments = {
        "order_id": "order-1001",
        "amount_cents": 30_000,
        "reason": "item_damaged",
    }
    call = ToolCall(CALL_ID, REFUND_CONTRACT.name, proposal_arguments)

    REFUND_CONTRACT.validate(proposal_arguments)
    injected_identity_rejection: dict[str, Any]
    try:
        REFUND_CONTRACT.validate({**proposal_arguments, "tenant_id": "tenant-shop-b"})
    except ToolArgumentValidationError as error:
        injected_identity_rejection = {
            "rejected": True,
            "code": error.code,
            "keyword": error.keyword,
        }
    else:
        raise AssertionError("closed schema accepted a model-authored tenant_id")

    provider = SimulatedRefundProvider()
    registry = _make_registry(provider)
    runtime = _make_runtime(registry, database)

    preview = runtime.execute(call, context=context)
    if preview.status is not ExecutionStatus.NEEDS_APPROVAL:
        raise AssertionError("authorized irreversible action did not pause for approval")

    cross_tenant = runtime.execute(
        ToolCall(
            "refund-order-9001-attempt-1",
            REFUND_CONTRACT.name,
            {**proposal_arguments, "order_id": "order-9001"},
        ),
        context=context,
    )
    if cross_tenant.status is not ExecutionStatus.POLICY_DENIED:
        raise AssertionError("cross-tenant order was not denied")
    if provider.request_attempts != 0:
        raise AssertionError("provider was called before ACL and approval completed")

    approval = ApprovalGrant(
        approval_id="approval-refund-1001-v1",
        approver_id="user-42-via-confirmation-ui",
        authorized_subject_id=context.subject_id,
        task_id=context.task_id,
        call_id=call.call_id,
        execution_fingerprint=preview.execution_fingerprint,
        expires_at_epoch_seconds=NOW + 300,
    )

    drifted_call = ToolCall(
        call.call_id,
        REFUND_CONTRACT.name,
        {**proposal_arguments, "amount_cents": 29_900},
    )
    drifted_approval = runtime.execute(
        drifted_call,
        context=context,
        approval=approval,
    )
    if drifted_approval.status is not ExecutionStatus.APPROVAL_REJECTED:
        raise AssertionError("changed refund amount reused the old approval")
    if provider.request_attempts != 0:
        raise AssertionError("approval drift reached the provider")

    execution = runtime.execute(call, context=context, approval=approval)
    if execution.status is not ExecutionStatus.FAILED:
        raise AssertionError("fixture did not expose the intended uncertain outcome")
    if provider.request_attempts != 1 or provider.effect_count != 1:
        raise AssertionError("first handler attempt did not create exactly one provider effect")
    attempts_after_execution = provider.request_attempts
    effects_after_execution = provider.effect_count

    replay = _make_runtime(registry, database).execute(call, context=context)
    if replay.status is not ExecutionStatus.FAILED or "pending" not in replay.message:
        raise AssertionError("pending call was not fenced before replay")
    if provider.request_attempts != 1 or provider.effect_count != 1:
        raise AssertionError("pending replay called the provider again")
    attempts_after_pending_replay = provider.request_attempts
    effects_after_pending_replay = provider.effect_count

    observed_receipt = provider.query_refund(CALL_ID)
    verification = _verify_provider_effect(observed_receipt)
    if verification["status"] != "passed" or observed_receipt is None:
        raise AssertionError("provider query did not prove the expected refund")
    mismatched_receipt = {**observed_receipt, "amount_cents": 29_900}
    mismatched_verification = _verify_provider_effect(mismatched_receipt)
    if mismatched_verification["status"] != "failed":
        raise AssertionError("verifier accepted a receipt for a different amount")

    ledger = SQLiteLedger(database)
    ledger.resolve_external_completion(
        call.call_id,
        observed_receipt,
        note="provider audit query confirmed the accepted refund",
    )
    revoked_context = ExecutionContext(
        task_id=context.task_id,
        subject_id=context.subject_id,
        tenant_id=context.tenant_id,
        capabilities=frozenset(),
    )
    revoked_replay = _make_runtime(registry, database).execute(
        call,
        context=revoked_context,
    )
    if revoked_replay.status is not ExecutionStatus.POLICY_DENIED:
        raise AssertionError("reconciled cache bypassed current authorization")
    recovered = _make_runtime(registry, database).execute(call, context=context)
    if recovered.status is not ExecutionStatus.CACHED:
        raise AssertionError("reconciled result was not reused after restart")
    if provider.request_attempts != 1 or provider.effect_count != 1:
        raise AssertionError("recovery duplicated the provider request or effect")
    attempts_after_recovery = provider.request_attempts
    effects_after_recovery = provider.effect_count

    history = ledger.reconciliation_history(call.call_id)
    return {
        "walkthrough_version": WALKTHROUGH_VERSION,
        "task": {
            "user_request": "商品坏了, 帮我退 300 元。",
            "desired_outcome": "refund order-1001 for CNY 300",
        },
        "stages": {
            "observation": {
                "untrusted_user_text": "商品坏了, 帮我退 300 元。",
                "trusted_context": {
                    "task_id": context.task_id,
                    "subject_id": context.subject_id,
                    "tenant_id": context.tenant_id,
                    "capabilities": sorted(context.capabilities),
                },
                "trusted_order_snapshot": dict(ORDERS["order-1001"]),
            },
            "proposal": {
                "planner_mode": "authored_offline_fixture",
                "tool_name": call.tool_name,
                "arguments": dict(call.arguments),
                "model_did_not_supply": [
                    "subject_id",
                    "tenant_id",
                    "capabilities",
                    "call_id",
                    "approval",
                ],
            },
            "schema": {
                "valid_proposal_accepted": True,
                "schema_revision": REFUND_CONTRACT.schema_revision,
                "validator_revision": REFUND_CONTRACT.validator_revision,
                "closed_schema_negative_control": injected_identity_rejection,
            },
            "acl": {
                "authorized_proposal": _outcome_summary(preview),
                "cross_tenant_negative_control": _outcome_summary(cross_tenant),
                "provider_attempts_after_acl": 0,
            },
            "approval": {
                "approval_id": approval.approval_id,
                "call_id": approval.call_id,
                "authorized_subject_id": approval.authorized_subject_id,
                "execution_fingerprint": approval.execution_fingerprint,
                "expires_at_epoch_seconds": approval.expires_at_epoch_seconds,
                "arguments_copied_into_approval": False,
                "drifted_amount_negative_control": _outcome_summary(drifted_approval),
                "provider_attempts_after_drift": 0,
            },
            "execution": {
                **_outcome_summary(execution),
                "handler_attempted": True,
                "provider_request_attempts": attempts_after_execution,
                "provider_effect_count": effects_after_execution,
                "local_ledger_state": "pending",
            },
            "idempotency": {
                **_outcome_summary(replay),
                "handler_attempted_on_replay": False,
                "provider_request_attempts": attempts_after_pending_replay,
                "provider_effect_count": effects_after_pending_replay,
            },
            "verifier": {
                **verification,
                "mismatched_receipt_negative_control": mismatched_verification,
            },
            "recovery": {
                "resolution": history[0].resolution,
                "note": history[0].note,
                "revoked_replay_negative_control": _outcome_summary(revoked_replay),
                "replay_after_reconciliation": _outcome_summary(recovered),
                "provider_request_attempts": attempts_after_recovery,
                "provider_effect_count": effects_after_recovery,
                "safe_final_answer": "退款已由支付服务确认受理, 退款单号 refund-provider-7001。",
            },
        },
        "scope": {
            "real_llm_or_provider_network_called": False,
            "planner_output_is_authored_fixture": True,
            "draft_2020_12_schema_executed": True,
            "resource_acl_and_approval_executed": True,
            "sqlite_claim_and_reconciliation_executed": True,
            "simulated_remote_effect_verified_by_separate_query": True,
            "exactly_once_or_production_safety_proved": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="about-llm-agent-") as temporary:
        payload = build_walkthrough(Path(temporary) / "refund-lifecycle.db")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    sys.stdout.buffer.write(rendered.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
