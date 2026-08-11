from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from about_llm.agents import (
    AgentRuntime,
    ApprovalGrant,
    CapabilityPolicy,
    ExecutionContext,
    ExecutionStatus,
    InMemoryLedger,
    PolicyDecision,
    PolicyEffect,
    PolicyRequest,
    ResourceRef,
    SideEffect,
    Tool,
    ToolCall,
    ToolRegistry,
    execution_fingerprint,
)


def validate_resource(arguments: Mapping[str, Any]) -> None:
    if set(arguments) != {"resource_id", "claimed_tenant"} or not all(
        isinstance(arguments[key], str) for key in arguments
    ):
        raise ValueError("expected resource_id and claimed_tenant")


def server_resolve(arguments: Mapping[str, Any]) -> ResourceRef:
    resource_id = arguments["resource_id"]
    assert isinstance(resource_id, str)
    owners = {"doc-a": "tenant-a", "doc-b": "tenant-b"}
    if resource_id not in owners:
        raise ValueError("unknown resource")
    return ResourceRef(
        owners[resource_id], "document", resource_id, "resource-table@v1"
    )


def make_tool(effects: list[str]) -> Tool:
    return Tool(
        "read_document",
        "document-reader@v1",
        "Read one server-resolved document.",
        SideEffect.READ_ONLY,
        validate_resource,
        lambda arguments: effects.append(str(arguments["resource_id"])) or "content",
        required_capability="document:read",
        resolve_resource=server_resolve,
    )


def make_write_tool(
    effects: list[str],
    *,
    tool_version: str = "document-writer@v1",
    resource_version: list[str] | None = None,
) -> Tool:
    version = ["resource-table@v1"] if resource_version is None else resource_version

    def resolve(arguments: Mapping[str, Any]) -> ResourceRef:
        resolved = server_resolve(arguments)
        return ResourceRef(
            resolved.tenant_id,
            resolved.resource_type,
            resolved.resource_id,
            version[0],
        )

    return Tool(
        "read_document",
        tool_version,
        "Simulate one write to a server-resolved document.",
        SideEffect.IRREVERSIBLE,
        validate_resource,
        lambda arguments: effects.append(str(arguments["resource_id"])) or {"ok": True},
        required_capability="document:read",
        resolve_resource=resolve,
    )


def approval_for(
    preview: Any,
    *,
    authorized_context: ExecutionContext,
    expires_at: float = 200.0,
) -> ApprovalGrant:
    return ApprovalGrant(
        "approval-1",
        "approver-1",
        authorized_context.subject_id,
        authorized_context.task_id,
        preview.call.call_id,
        preview.execution_fingerprint,
        expires_at,
    )


def context(
    *, tenant_id: str = "tenant-a", capabilities: frozenset[str] | None = None
) -> ExecutionContext:
    return ExecutionContext(
        "task-1",
        "user-1",
        tenant_id,
        frozenset({"document:read"}) if capabilities is None else capabilities,
    )


def call(resource_id: str = "doc-a", claimed_tenant: str = "tenant-a") -> ToolCall:
    return ToolCall(
        f"call-{resource_id}",
        "read_document",
        {"resource_id": resource_id, "claimed_tenant": claimed_tenant},
    )


def test_runtime_is_default_deny_even_for_read_only_tool() -> None:
    effects: list[str] = []
    runtime = AgentRuntime(ToolRegistry([make_tool(effects)]))

    outcome = runtime.execute(call(), context=context())

    assert outcome.status is ExecutionStatus.POLICY_DENIED
    assert outcome.policy_decision.effect is PolicyEffect.DENY
    assert outcome.policy_decision.reason_code == "policy_not_configured"
    assert runtime.executed_tool_calls == 0
    assert effects == []


def test_capability_and_server_resolved_tenant_are_both_required() -> None:
    effects: list[str] = []
    runtime = AgentRuntime(
        ToolRegistry([make_tool(effects)]),
        policy=CapabilityPolicy("document-policy@v1"),
    )

    missing_capability = runtime.execute(
        call(), context=context(capabilities=frozenset())
    )
    cross_tenant = runtime.execute(
        call("doc-b", claimed_tenant="tenant-a"), context=context()
    )

    assert missing_capability.status is ExecutionStatus.POLICY_DENIED
    assert missing_capability.policy_decision.reason_code == "missing_capability"
    assert cross_tenant.status is ExecutionStatus.POLICY_DENIED
    assert cross_tenant.policy_decision.reason_code == "tenant_mismatch"
    assert cross_tenant.resource.tenant_id == "tenant-b"
    assert effects == []


def test_cached_replay_is_reauthorized_after_capability_revocation() -> None:
    effects: list[str] = []
    runtime = AgentRuntime(
        ToolRegistry([make_tool(effects)]),
        policy=CapabilityPolicy("document-policy@v1"),
    )
    proposal = call()

    first = runtime.execute(proposal, context=context())
    revoked = runtime.execute(
        proposal, context=context(capabilities=frozenset())
    )
    restored = runtime.execute(proposal, context=context())

    assert first.status is ExecutionStatus.COMPLETED
    assert first.execution_fingerprint != proposal.fingerprint()
    assert revoked.status is ExecutionStatus.POLICY_DENIED
    assert restored.status is ExecutionStatus.CACHED
    assert effects == ["doc-a"]


def test_execution_identity_binds_subject_resource_tool_and_policy_versions() -> None:
    effects: list[str] = []
    base_tool = make_tool(effects)
    proposal = call()
    base_context = context()
    base_resource = server_resolve(proposal.arguments)
    base_policy = CapabilityPolicy("document-policy@v1")
    def request(
        resource: ResourceRef, subject_context: ExecutionContext
    ) -> PolicyRequest:
        return PolicyRequest(
            context=subject_context,
            call_id=proposal.call_id,
            call_fingerprint=proposal.fingerprint(),
            tool_name=base_tool.name,
            required_capability=base_tool.required_capability,
            side_effect=base_tool.side_effect.value,
            resource=resource,
        )

    base_decision = base_policy.evaluate(request(base_resource, base_context))

    def identity(
        tool: Tool,
        subject_context: ExecutionContext,
        resource: ResourceRef,
        policy: CapabilityPolicy,
    ) -> str:
        decision = policy.evaluate(
            request(resource, subject_context)
        )
        return execution_fingerprint(
            call=proposal,
            tool=tool,
            context=subject_context,
            policy_decision=decision,
            resource=resource,
        )

    changed_tool = Tool(
        base_tool.name,
        "document-reader@v2",
        base_tool.description,
        base_tool.side_effect,
        base_tool.validate,
        base_tool.handler,
        required_capability=base_tool.required_capability,
        resolve_resource=base_tool.resolve_resource,
    )
    changed_subject = ExecutionContext(
        base_context.task_id,
        "user-2",
        base_context.tenant_id,
        base_context.capabilities,
    )
    changed_resource = ResourceRef(
        base_resource.tenant_id,
        base_resource.resource_type,
        base_resource.resource_id,
        "resource-table@v2",
    )
    base_identity = execution_fingerprint(
        call=proposal,
        tool=base_tool,
        context=base_context,
        policy_decision=base_decision,
        resource=base_resource,
    )
    identities = {
        base_identity,
        identity(changed_tool, base_context, base_resource, base_policy),
        identity(base_tool, changed_subject, base_resource, base_policy),
        identity(base_tool, base_context, changed_resource, base_policy),
        identity(
            base_tool,
            base_context,
            base_resource,
            CapabilityPolicy("document-policy@v2"),
        ),
    }

    assert len(identities) == 5


def test_same_call_id_cannot_cross_tool_contract_version() -> None:
    effects: list[str] = []
    first_tool = make_tool(effects)
    second_tool = Tool(
        first_tool.name,
        "document-reader@v2",
        first_tool.description,
        first_tool.side_effect,
        first_tool.validate,
        first_tool.handler,
        required_capability=first_tool.required_capability,
        resolve_resource=first_tool.resolve_resource,
    )
    ledger = InMemoryLedger()
    first_runtime = AgentRuntime(
        ToolRegistry([first_tool]),
        ledger=ledger,
        policy=CapabilityPolicy("document-policy@v1"),
    )
    second_runtime = AgentRuntime(
        ToolRegistry([second_tool]),
        ledger=ledger,
        policy=CapabilityPolicy("document-policy@v1"),
    )

    assert first_runtime.execute(call(), context=context()).status is ExecutionStatus.COMPLETED
    with pytest.raises(RuntimeError, match="different execution identity"):
        second_runtime.execute(call(), context=context())
    assert effects == ["doc-a"]


def test_execution_context_snapshots_capabilities() -> None:
    capabilities = {"document:read"}
    trusted_context = ExecutionContext(
        "task-1",
        "user-1",
        "tenant-a",
        capabilities,  # type: ignore[arg-type]
    )
    capabilities.clear()

    assert trusted_context.capabilities == frozenset({"document:read"})


def test_denied_decision_cannot_claim_matched_capability() -> None:
    with pytest.raises(ValueError, match="non-allow decision"):
        PolicyDecision(
            PolicyEffect.DENY,
            "policy@v1",
            "denied",
            matched_capability="document:read",
        )


def test_indeterminate_policy_fails_closed_without_handler_attempt() -> None:
    class IndeterminatePolicy:
        def evaluate(self, request: PolicyRequest) -> PolicyDecision:
            return PolicyDecision(
                PolicyEffect.INDETERMINATE,
                "unavailable-policy@v1",
                "policy_backend_unavailable",
            )

    effects: list[str] = []
    runtime = AgentRuntime(
        ToolRegistry([make_tool(effects)]), policy=IndeterminatePolicy()
    )

    outcome = runtime.execute(call(), context=context())

    assert outcome.status is ExecutionStatus.POLICY_DENIED
    assert outcome.policy_decision.effect is PolicyEffect.INDETERMINATE
    assert outcome.policy_decision.reason_code == "policy_backend_unavailable"
    assert effects == []


def test_execution_context_rejects_string_as_capability_collection() -> None:
    with pytest.raises(ValueError, match="collection of capability strings"):
        ExecutionContext("task", "user", "tenant", "document:read")  # type: ignore[arg-type]


def test_runtime_rejects_malformed_resource_or_policy_outputs() -> None:
    class BadPolicy:
        def evaluate(self, request: PolicyRequest) -> object:
            return object()

    effects: list[str] = []
    valid_tool = make_tool(effects)
    invalid_resource_tool = Tool(
        valid_tool.name,
        valid_tool.version,
        valid_tool.description,
        valid_tool.side_effect,
        valid_tool.validate,
        valid_tool.handler,
        required_capability=valid_tool.required_capability,
        resolve_resource=lambda arguments: "not-a-resource",  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(TypeError, match="ResourceRef"):
        AgentRuntime(
            ToolRegistry([invalid_resource_tool]),
            policy=CapabilityPolicy("policy@v1"),
        ).execute(call(), context=context())
    with pytest.raises(TypeError, match="PolicyDecision"):
        AgentRuntime(
            ToolRegistry([valid_tool]),
            policy=BadPolicy(),  # type: ignore[arg-type]
        ).execute(call(), context=context())
    assert effects == []


def test_approval_rejects_expiry_subject_and_argument_drift() -> None:
    effects: list[str] = []
    trusted_context = context()
    runtime = AgentRuntime(
        ToolRegistry([make_write_tool(effects)]),
        policy=CapabilityPolicy("document-policy@v1"),
        clock=lambda: 100.0,
    )
    proposal = call()
    preview = runtime.execute(proposal, context=trusted_context)
    assert preview.status is ExecutionStatus.NEEDS_APPROVAL

    expired = runtime.execute(
        proposal,
        context=trusted_context,
        approval=approval_for(
            preview, authorized_context=trusted_context, expires_at=100.0
        ),
    )
    changed_subject = ExecutionContext(
        trusted_context.task_id,
        "user-2",
        trusted_context.tenant_id,
        trusted_context.capabilities,
    )
    wrong_subject = runtime.execute(
        proposal,
        context=changed_subject,
        approval=approval_for(preview, authorized_context=trusted_context),
    )
    changed_arguments = runtime.execute(
        call(claimed_tenant="model-claimed-other-tenant"),
        context=trusted_context,
        approval=approval_for(preview, authorized_context=trusted_context),
    )

    assert expired.status is ExecutionStatus.APPROVAL_REJECTED
    assert "approval_expired" in expired.message
    assert wrong_subject.status is ExecutionStatus.APPROVAL_REJECTED
    assert "approval_subject_mismatch" in wrong_subject.message
    assert changed_arguments.status is ExecutionStatus.APPROVAL_REJECTED
    assert "approval_execution_mismatch" in changed_arguments.message
    assert effects == []


def test_approval_binds_resource_tool_and_policy_versions() -> None:
    effects: list[str] = []
    resource_version = ["resource-table@v1"]
    trusted_context = context()
    tool = make_write_tool(effects, resource_version=resource_version)
    runtime = AgentRuntime(
        ToolRegistry([tool]),
        policy=CapabilityPolicy("document-policy@v1"),
        clock=lambda: 100.0,
    )
    proposal = call()
    preview = runtime.execute(proposal, context=trusted_context)
    grant = approval_for(preview, authorized_context=trusted_context)

    resource_version[0] = "resource-table@v2"
    resource_drift = runtime.execute(
        proposal, context=trusted_context, approval=grant
    )
    resource_version[0] = "resource-table@v1"
    tool_drift = AgentRuntime(
        ToolRegistry(
            [
                make_write_tool(
                    effects,
                    tool_version="document-writer@v2",
                    resource_version=resource_version,
                )
            ]
        ),
        policy=CapabilityPolicy("document-policy@v1"),
        clock=lambda: 100.0,
    ).execute(proposal, context=trusted_context, approval=grant)
    policy_drift = AgentRuntime(
        ToolRegistry([tool]),
        policy=CapabilityPolicy("document-policy@v2"),
        clock=lambda: 100.0,
    ).execute(proposal, context=trusted_context, approval=grant)
    completed = runtime.execute(
        proposal, context=trusted_context, approval=grant
    )

    assert resource_drift.status is ExecutionStatus.APPROVAL_REJECTED
    assert tool_drift.status is ExecutionStatus.APPROVAL_REJECTED
    assert policy_drift.status is ExecutionStatus.APPROVAL_REJECTED
    assert completed.status is ExecutionStatus.COMPLETED
    assert effects == ["doc-a"]


@pytest.mark.parametrize("expiry", [float("nan"), float("inf"), True])
def test_approval_rejects_non_finite_or_boolean_expiry(expiry: object) -> None:
    with pytest.raises(ValueError, match="approval expiry"):
        ApprovalGrant(
            "approval",
            "approver",
            "user",
            "task",
            "call",
            "sha256:" + "a" * 64,
            expiry,  # type: ignore[arg-type]
        )
