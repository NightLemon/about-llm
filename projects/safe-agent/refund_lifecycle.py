"""让一次退款任务走过 Agent 的提案、授权、审批、执行、验证与恢复。

用户说“商品坏了，退 300 元”后，实验把自然语言转成工具参数，但 subject、tenant、capability、
call ID 与 approval 都来自可信控制面。远端 provider 首次成功后丢失响应，迫使系统用幂等、
持久化 pending 状态和独立查询恢复，而不是盲目重试副作用。
"""

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
    """可信订单库中的最小记录；这些字段不会由模型自行声明。"""

    tenant_id: str
    owner_id: str
    paid_cents: int
    status: str
    version: str


# 两个订单金额相同但属于不同租户，用来验证资源 ID 不能绕过 tenant ACL。
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

# 工具 schema 只接受业务参数，不允许模型夹带 tenant、subject 或 approval。
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
    """在模型参数之外检查租户、所有权和精确 capability。"""

    VERSION = "refund-acl@v1"

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        """默认拒绝，只有四项可信条件全部满足才允许退款。"""

        # order 来自服务端存储，不能信任模型在 arguments 中声称的归属。
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
        """生成带稳定 reason code 的拒绝决定。"""

        return PolicyDecision(
            effect=PolicyEffect.DENY,
            policy_version=cls.VERSION,
            reason_code=reason_code,
        )


@dataclass
class SimulatedRefundProvider:
    """先接受远端退款副作用，再故意丢失第一次响应。"""

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
        """按 idempotency key 创建或复用退款，并模拟 success-before-timeout。"""

        self.request_attempts += 1
        receipt = self.receipts.get(idempotency_key)
        if receipt is None:
            # 只有新 key 才产生新退款；相同 key 的重试始终返回原 receipt。
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
        # 副作用已持久存在后才抛超时，复现最危险的“不知道成功还是失败”窗口。
        if self.lose_first_response:
            self.lose_first_response = False
            raise TimeoutError("provider accepted the refund but the response was lost")
        return dict(receipt)

    def query_refund(self, idempotency_key: str) -> dict[str, Any] | None:
        """独立查询 provider 状态，不重新提交退款。"""

        receipt = self.receipts.get(idempotency_key)
        return dict(receipt) if receipt is not None else None


def _resolve_order(arguments: Mapping[str, Any]) -> ResourceRef:
    """用模型提供的 order_id 在可信订单库解析真实资源边界。"""

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
    """把退款 contract、资源解析器、权限和 handler 组装成工具注册表。"""

    def handler(arguments: Mapping[str, Any]) -> dict[str, Any]:
        """调用模拟 provider；此时 schema、ACL 与 approval 均已通过。"""

        # CALL_ID 由可信 orchestrator 分配，故意不接受模型在 arguments 中自报。
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
    """创建使用 SQLite ledger、固定策略和固定时钟的 AgentRuntime。"""

    return AgentRuntime(
        registry,
        max_tool_calls=2,
        ledger=SQLiteLedger(database),
        policy=RefundACLPolicy(),
        clock=lambda: NOW,
    )


def _outcome_summary(outcome: ExecutionOutcome) -> dict[str, Any]:
    """提取贯穿各阶段对比所需的执行状态、资源和指纹。"""

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
    """独立核对 provider receipt 是否与原批准退款完全一致。"""

    # 验证器不只看 status，还绑定 call、order、金额、原因和 provider 侧退款 ID。
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
    """运行完整退款生命周期，并返回按阶段组织的可检查证据。"""

    if database.exists():
        raise ValueError("walkthrough database path must not already exist")

    # observation 阶段将不可信用户文本与可信登录上下文、订单快照分开。
    context = ExecutionContext(
        task_id=TASK_ID,
        subject_id="user-42",
        tenant_id="tenant-shop-a",
        capabilities=frozenset({"refund:request"}),
    )
    # proposal 只表达模型建议做什么，不携带它无权决定的身份或授权字段。
    proposal_arguments = {
        "order_id": "order-1001",
        "amount_cents": 30_000,
        "reason": "item_damaged",
    }
    call = ToolCall(CALL_ID, REFUND_CONTRACT.name, proposal_arguments)

    # 先验证正常 proposal，再尝试注入 tenant_id，确认 closed schema 会拒绝未知字段。
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

    # irreversible 工具即使 ACL 允许，也必须先停在 NEEDS_APPROVAL，不能调用 provider。
    preview = runtime.execute(call, context=context)
    if preview.status is not ExecutionStatus.NEEDS_APPROVAL:
        raise AssertionError("authorized irreversible action did not pause for approval")

    # 把 order_id 换成另一租户订单，验证服务端资源解析后的 ACL 会拒绝。
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

    # approval 绑定 execution_fingerprint，而非只绑定容易复用的 tool name。
    approval = ApprovalGrant(
        approval_id="approval-refund-1001-v1",
        approver_id="user-42-via-confirmation-ui",
        authorized_subject_id=context.subject_id,
        task_id=context.task_id,
        call_id=call.call_id,
        execution_fingerprint=preview.execution_fingerprint,
        expires_at_epoch_seconds=NOW + 300,
    )

    # 金额改动会改变 execution fingerprint，旧批准必须失效且不能触达 provider。
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

    # 正式执行时 provider 实际接受退款，但丢失响应；本地只能得到 FAILED/pending。
    execution = runtime.execute(call, context=context, approval=approval)
    if execution.status is not ExecutionStatus.FAILED:
        raise AssertionError("fixture did not expose the intended uncertain outcome")
    if provider.request_attempts != 1 or provider.effect_count != 1:
        raise AssertionError("first handler attempt did not create exactly one provider effect")
    attempts_after_execution = provider.request_attempts
    effects_after_execution = provider.effect_count

    # 新建 runtime 模拟进程重启。pending fence 阻止它盲目再次调用不可逆工具。
    replay = _make_runtime(registry, database).execute(call, context=context)
    if replay.status is not ExecutionStatus.FAILED or "pending" not in replay.message:
        raise AssertionError("pending call was not fenced before replay")
    if provider.request_attempts != 1 or provider.effect_count != 1:
        raise AssertionError("pending replay called the provider again")
    attempts_after_pending_replay = provider.request_attempts
    effects_after_pending_replay = provider.effect_count

    # 恢复流程先走只读查询，再用独立 verifier 核对远端事实。
    observed_receipt = provider.query_refund(CALL_ID)
    verification = _verify_provider_effect(observed_receipt)
    if verification["status"] != "passed" or observed_receipt is None:
        raise AssertionError("provider query did not prove the expected refund")
    mismatched_receipt = {**observed_receipt, "amount_cents": 29_900}
    mismatched_verification = _verify_provider_effect(mismatched_receipt)
    if mismatched_verification["status"] != "failed":
        raise AssertionError("verifier accepted a receipt for a different amount")

    # verifier 通过后才把 pending 记录人工/自动 reconciliation 为外部已完成。
    ledger = SQLiteLedger(database)
    ledger.resolve_external_completion(
        call.call_id,
        observed_receipt,
        note="provider audit query confirmed the accepted refund",
    )
    # 即使 ledger 已缓存成功结果，当前权限撤销后也不能直接泄露 receipt。
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
    # 权限仍有效时，重启后的 runtime 可安全返回 reconciled cache，不重复远端 effect。
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
    """在临时 SQLite 数据库运行退款故事并输出 UTF-8 trace。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="about-llm-agent-") as temporary:
        payload = build_walkthrough(Path(temporary) / "refund-lifecycle.db")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    sys.stdout.buffer.write(rendered.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
