"""Typed, model-independent authorization policy for Agent tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class PolicyEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ExecutionContext:
    """Trusted caller identity and capabilities, never model-authored fields."""

    task_id: str
    subject_id: str
    tenant_id: str
    capabilities: frozenset[str]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("task_id", self.task_id),
            ("subject_id", self.subject_id),
            ("tenant_id", self.tenant_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if isinstance(self.capabilities, (str, bytes, bytearray)):
            raise ValueError("capabilities must be a collection of capability strings")
        capabilities = frozenset(self.capabilities)
        if any(
            not isinstance(capability, str) or not capability.strip()
            for capability in capabilities
        ):
            raise ValueError("capabilities must contain only non-empty strings")
        object.__setattr__(self, "capabilities", capabilities)


@dataclass(frozen=True)
class ResourceRef:
    """Server-resolved resource ownership and revision used for authorization."""

    tenant_id: str
    resource_type: str
    resource_id: str
    version: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("tenant_id", self.tenant_id),
            ("resource_type", self.resource_type),
            ("resource_id", self.resource_id),
            ("version", self.version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"resource {field_name} must be a non-empty string")


@dataclass(frozen=True)
class PolicyRequest:
    context: ExecutionContext
    call_id: str
    call_fingerprint: str
    tool_name: str
    required_capability: str
    side_effect: str
    resource: ResourceRef


@dataclass(frozen=True)
class PolicyDecision:
    effect: PolicyEffect
    policy_version: str
    reason_code: str
    matched_capability: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.effect, PolicyEffect):
            raise ValueError("policy effect must be a PolicyEffect")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version cannot be empty")
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError("policy reason_code cannot be empty")
        if self.matched_capability is not None and (
            not isinstance(self.matched_capability, str)
            or not self.matched_capability.strip()
        ):
            raise ValueError("matched_capability cannot be empty")
        if self.effect is not PolicyEffect.ALLOW and self.matched_capability is not None:
            raise ValueError("non-allow decision cannot claim a matched capability")


class PolicyEvaluator(Protocol):
    def evaluate(self, request: PolicyRequest) -> PolicyDecision: ...


class DefaultDenyPolicy:
    """Fail closed when the caller did not install an application policy."""

    VERSION = "builtin-default-deny@v1"

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        return PolicyDecision(
            effect=PolicyEffect.DENY,
            policy_version=self.VERSION,
            reason_code="policy_not_configured",
        )


class CapabilityPolicy:
    """Exact-capability and same-tenant reference policy.

    This deliberately has no wildcard or role inheritance. Production systems
    may replace it, but must keep identity and resource ownership outside model
    arguments and return a typed decision for every proposal and replay.
    """

    def __init__(self, policy_version: str) -> None:
        if not isinstance(policy_version, str) or not policy_version.strip():
            raise ValueError("policy_version cannot be empty")
        self.policy_version = policy_version

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        if request.context.tenant_id != request.resource.tenant_id:
            return PolicyDecision(
                effect=PolicyEffect.DENY,
                policy_version=self.policy_version,
                reason_code="tenant_mismatch",
            )
        if request.required_capability not in request.context.capabilities:
            return PolicyDecision(
                effect=PolicyEffect.DENY,
                policy_version=self.policy_version,
                reason_code="missing_capability",
            )
        return PolicyDecision(
            effect=PolicyEffect.ALLOW,
            policy_version=self.policy_version,
            reason_code="capability_granted",
            matched_capability=request.required_capability,
        )
