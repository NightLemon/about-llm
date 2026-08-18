from __future__ import annotations

from dataclasses import replace

import pytest

from about_llm.integrations.reasoning_artifact import (
    InMemoryConsumptionLedger,
    InMemoryNonceLedger,
    ReasoningArtifactClaims,
    ReasoningArtifactError,
    ReasoningEnvelope,
    ReasoningReplayContext,
    consume_reasoning_envelope,
    issue_reasoning_envelope,
)

pytestmark = [pytest.mark.contract, pytest.mark.security]

KEY = bytes(range(32))
KEY_ID = "fixture-key-2026-08"
PREDECESSOR = "1" * 64
PLAINTEXT = b"authored reasoning bytes with no real user data"


def claims() -> ReasoningArtifactClaims:
    return ReasoningArtifactClaims(
        artifact_id="artifact-001",
        provider="fixture-provider",
        key_id=KEY_ID,
        subject_id="subject-a",
        tenant_id="tenant-a",
        session_id="session-a",
        branch_id="main",
        predecessor_digest=PREDECESSOR,
        model_audience=("model-strong",),
        issued_at_epoch_seconds=100,
        expires_at_epoch_seconds=200,
    )


def context() -> ReasoningReplayContext:
    return ReasoningReplayContext(
        provider="fixture-provider",
        subject_id="subject-a",
        tenant_id="tenant-a",
        session_id="session-a",
        branch_id="main",
        predecessor_digest=PREDECESSOR,
        model_id="model-strong",
        now_epoch_seconds=150,
    )


def issue(binding_mode: str, *, nonce_byte: int = 1) -> ReasoningEnvelope:
    assert binding_mode in {"content-only", "context-bound"}
    return issue_reasoning_envelope(
        key=KEY,
        claims=claims(),
        plaintext=PLAINTEXT,
        binding_mode=binding_mode,
        nonce=bytes([nonce_byte]) * 12,
        nonce_ledger=InMemoryNonceLedger(),
    )


def consume(
    envelope: ReasoningEnvelope,
    replay_context: ReasoningReplayContext,
    *,
    ledger: InMemoryConsumptionLedger | None = None,
    retired: frozenset[str] = frozenset(),
) -> bytes:
    return consume_reasoning_envelope(
        envelope,
        keys={KEY_ID: KEY},
        retired_key_ids=retired,
        context=replay_context,
        consumption_ledger=ledger,
    )


@pytest.mark.parametrize(
    "changed_context",
    [
        replace(context(), subject_id="subject-b"),
        replace(context(), tenant_id="tenant-b"),
        replace(context(), session_id="session-b"),
        replace(context(), branch_id="fork-b"),
        replace(context(), predecessor_digest="2" * 64),
        replace(context(), model_id="model-weak"),
        replace(context(), now_epoch_seconds=250),
    ],
)
def test_content_only_authentication_does_not_stop_context_replay(
    changed_context: ReasoningReplayContext,
) -> None:
    envelope = issue("content-only")

    assert consume(envelope, changed_context) == PLAINTEXT


@pytest.mark.parametrize(
    ("changed_context", "reason"),
    [
        (replace(context(), subject_id="subject-b"), "subject_mismatch"),
        (replace(context(), tenant_id="tenant-b"), "tenant_mismatch"),
        (replace(context(), session_id="session-b"), "session_mismatch"),
        (replace(context(), branch_id="fork-b"), "branch_mismatch"),
        (replace(context(), predecessor_digest="2" * 64), "predecessor_mismatch"),
        (replace(context(), model_id="model-weak"), "model_not_allowed"),
        (replace(context(), now_epoch_seconds=99), "not_yet_valid"),
        (replace(context(), now_epoch_seconds=200), "expired"),
    ],
)
def test_context_bound_envelope_rejects_scope_drift(
    changed_context: ReasoningReplayContext, reason: str
) -> None:
    envelope = issue("context-bound")

    with pytest.raises(ReasoningArtifactError, match=reason):
        consume(envelope, changed_context)


def test_context_bound_envelope_accepts_exact_context_once() -> None:
    envelope = issue("context-bound")
    ledger = InMemoryConsumptionLedger()

    assert consume(envelope, context(), ledger=ledger) == PLAINTEXT
    with pytest.raises(ReasoningArtifactError, match="replay_detected"):
        consume(envelope, context(), ledger=ledger)


def test_bound_claim_or_ciphertext_tampering_fails_authentication() -> None:
    envelope = issue("context-bound")
    changed_claims = replace(envelope.claims, subject_id="subject-b")
    changed_ciphertext = envelope.ciphertext[:-1] + bytes([envelope.ciphertext[-1] ^ 1])

    for tampered in (
        replace(envelope, claims=changed_claims),
        replace(envelope, ciphertext=changed_ciphertext),
    ):
        with pytest.raises(ReasoningArtifactError, match="authentication_failed"):
            consume(tampered, context())


def test_nonce_reuse_and_retired_key_fail_closed() -> None:
    nonce_ledger = InMemoryNonceLedger()
    nonce = b"n" * 12
    issue_reasoning_envelope(
        key=KEY,
        claims=claims(),
        plaintext=PLAINTEXT,
        binding_mode="context-bound",
        nonce=nonce,
        nonce_ledger=nonce_ledger,
    )
    with pytest.raises(ReasoningArtifactError, match="nonce_reused"):
        issue_reasoning_envelope(
            key=KEY,
            claims=replace(claims(), artifact_id="artifact-002"),
            plaintext=PLAINTEXT,
            binding_mode="context-bound",
            nonce=nonce,
            nonce_ledger=nonce_ledger,
        )

    envelope = issue("context-bound", nonce_byte=2)
    with pytest.raises(ReasoningArtifactError, match="retired_key"):
        consume(envelope, context(), retired=frozenset({KEY_ID}))
