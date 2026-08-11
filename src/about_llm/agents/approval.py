"""Typed approval grants bound to one authorized execution identity."""

from __future__ import annotations

import math
from dataclasses import dataclass

from about_llm.agents.policy import ExecutionContext


@dataclass(frozen=True)
class ApprovalGrant:
    """Trusted-service approval artifact; not a signature or bearer-token format."""

    approval_id: str
    approver_id: str
    authorized_subject_id: str
    task_id: str
    call_id: str
    execution_fingerprint: str
    expires_at_epoch_seconds: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("approval_id", self.approval_id),
            ("approver_id", self.approver_id),
            ("authorized_subject_id", self.authorized_subject_id),
            ("task_id", self.task_id),
            ("call_id", self.call_id),
            ("execution_fingerprint", self.execution_fingerprint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"approval {field_name} must be a non-empty string")
        if not self.execution_fingerprint.startswith("sha256:") or len(
            self.execution_fingerprint
        ) != len("sha256:") + 64:
            raise ValueError("approval execution_fingerprint must be sha256:<64 hex>")
        try:
            int(self.execution_fingerprint.removeprefix("sha256:"), 16)
        except ValueError as error:
            raise ValueError("approval fingerprint digest must be hexadecimal") from error
        if isinstance(self.expires_at_epoch_seconds, bool) or not isinstance(
            self.expires_at_epoch_seconds, (int, float)
        ):
            raise ValueError("approval expiry must be a numeric timestamp")
        if not math.isfinite(self.expires_at_epoch_seconds):
            raise ValueError("approval expiry must be finite")


def approval_rejection_reason(
    grant: ApprovalGrant,
    *,
    context: ExecutionContext,
    call_id: str,
    execution_fingerprint: str,
    now_epoch_seconds: float,
) -> str | None:
    """Return a stable rejection code, or None when the grant matches."""

    if isinstance(now_epoch_seconds, bool) or not isinstance(
        now_epoch_seconds, (int, float)
    ) or not math.isfinite(now_epoch_seconds):
        raise ValueError("trusted approval clock must return a finite timestamp")
    if now_epoch_seconds >= grant.expires_at_epoch_seconds:
        return "approval_expired"
    if grant.task_id != context.task_id:
        return "approval_task_mismatch"
    if grant.authorized_subject_id != context.subject_id:
        return "approval_subject_mismatch"
    if grant.call_id != call_id:
        return "approval_call_mismatch"
    if grant.execution_fingerprint != execution_fingerprint:
        return "approval_execution_mismatch"
    return None
