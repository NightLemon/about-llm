from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from about_llm.conversation import (
    ConversationMemoryLedger,
    MemoryKind,
    MemoryScope,
    MemoryStatus,
)

pytestmark = [pytest.mark.contract, pytest.mark.security]

NOW = datetime(2026, 8, 6, 4, tzinfo=timezone.utc)


def add_session_fact(
    ledger: ConversationMemoryLedger,
    *,
    fact_id: str = "fact-1",
    tenant_id: str = "tenant-a",
    subject_id: str = "user-1",
    key: str = "preferred_language",
    value: object = "zh-CN",
    expires_at: datetime | None = None,
):
    return ledger.add_fact(
        fact_id=fact_id,
        tenant_id=tenant_id,
        subject_id=subject_id,
        key=key,
        value=value,
        kind=MemoryKind.WORKING,
        scope=MemoryScope.SESSION,
        source_event_id=f"source-{fact_id}",
        created_at=NOW,
        confidence=1.0,
        policy_version="memory-policy-v1",
        expires_at=expires_at,
    )


def test_add_and_read_active_fact() -> None:
    ledger = ConversationMemoryLedger()
    fact = add_session_fact(ledger)

    assert ledger.active_facts(tenant_id="tenant-a", subject_id="user-1", now=NOW) == (
        fact,
    )
    assert fact.value == "zh-CN"
    assert ledger.status(
        fact_id="fact-1", tenant_id="tenant-a", subject_id="user-1", now=NOW
    ) is MemoryStatus.ACTIVE


def test_value_is_a_defensive_canonical_json_snapshot() -> None:
    ledger = ConversationMemoryLedger()
    source = {"languages": ["zh-CN", "en"]}
    fact = add_session_fact(ledger, value=source)
    source["languages"].append("ja")
    first_read = fact.value
    assert isinstance(first_read, dict)
    assert first_read == {"languages": ["zh-CN", "en"]}
    first_read["languages"] = []
    assert fact.value == {"languages": ["zh-CN", "en"]}


def test_profile_memory_requires_consent_reference() -> None:
    ledger = ConversationMemoryLedger()
    with pytest.raises(ValueError, match="consent_reference"):
        ledger.add_fact(
            fact_id="profile-1",
            tenant_id="tenant-a",
            subject_id="user-1",
            key="preferred_language",
            value="zh-CN",
            kind=MemoryKind.SEMANTIC,
            scope=MemoryScope.PROFILE,
            source_event_id="message-1",
            created_at=NOW,
            confidence=1.0,
            policy_version="memory-policy-v1",
        )


def test_profile_memory_with_consent_is_active() -> None:
    ledger = ConversationMemoryLedger()
    fact = ledger.add_fact(
        fact_id="profile-1",
        tenant_id="tenant-a",
        subject_id="user-1",
        key="preferred_language",
        value="zh-CN",
        kind=MemoryKind.SEMANTIC,
        scope=MemoryScope.PROFILE,
        source_event_id="message-1",
        created_at=NOW,
        confidence=1.0,
        policy_version="memory-policy-v1",
        consent_reference="consent-event-1",
    )

    assert fact.value == "zh-CN"


def test_correction_supersedes_previous_fact() -> None:
    ledger = ConversationMemoryLedger()
    previous = add_session_fact(ledger)
    corrected = ledger.correct_fact(
        previous_fact_id=previous.fact_id,
        new_fact_id="fact-2",
        tenant_id="tenant-a",
        subject_id="user-1",
        value="en",
        source_event_id="correction-1",
        created_at=NOW + timedelta(minutes=1),
        confidence=1.0,
    )

    assert ledger.active_facts(
        tenant_id="tenant-a", subject_id="user-1", now=NOW + timedelta(minutes=1)
    ) == (corrected,)
    assert ledger.status(
        fact_id=previous.fact_id,
        tenant_id="tenant-a",
        subject_id="user-1",
        now=NOW + timedelta(minutes=1),
    ) is MemoryStatus.SUPERSEDED
    assert ledger.history(tenant_id="tenant-a", subject_id="user-1") == (
        previous,
        corrected,
    )


def test_active_key_must_be_explicitly_superseded() -> None:
    ledger = ConversationMemoryLedger()
    add_session_fact(ledger)

    with pytest.raises(ValueError, match="supersede"):
        add_session_fact(ledger, fact_id="fact-2", value="en")


def test_cannot_supersede_across_tenants() -> None:
    ledger = ConversationMemoryLedger()
    previous = add_session_fact(ledger)

    with pytest.raises(KeyError, match="tenant and subject scope"):
        ledger.add_fact(
            fact_id="fact-2",
            tenant_id="tenant-b",
            subject_id="user-1",
            key=previous.key,
            value="en",
            kind=MemoryKind.WORKING,
            scope=MemoryScope.SESSION,
            source_event_id="source-2",
            created_at=NOW + timedelta(minutes=1),
            confidence=1.0,
            policy_version="memory-policy-v1",
            supersedes_fact_id=previous.fact_id,
        )


def test_retraction_removes_fact_from_active_view_but_keeps_history() -> None:
    ledger = ConversationMemoryLedger()
    fact = add_session_fact(ledger)
    event = ledger.retract_fact(
        retraction_id="retract-1",
        fact_id=fact.fact_id,
        tenant_id="tenant-a",
        subject_id="user-1",
        source_event_id="user-delete-1",
        reason="user withdrew the preference",
        created_at=NOW + timedelta(minutes=1),
    )

    assert event.fact_id == fact.fact_id
    assert ledger.active_facts(
        tenant_id="tenant-a", subject_id="user-1", now=NOW + timedelta(minutes=1)
    ) == ()
    assert ledger.history(tenant_id="tenant-a", subject_id="user-1") == (fact,)
    assert ledger.status(
        fact_id=fact.fact_id,
        tenant_id="tenant-a",
        subject_id="user-1",
        now=NOW + timedelta(minutes=1),
    ) is MemoryStatus.RETRACTED


def test_expiry_boundary_is_inactive() -> None:
    ledger = ConversationMemoryLedger()
    expiry = NOW + timedelta(hours=1)
    fact = add_session_fact(ledger, expires_at=expiry)

    assert ledger.status(
        fact_id=fact.fact_id,
        tenant_id="tenant-a",
        subject_id="user-1",
        now=expiry,
    ) is MemoryStatus.EXPIRED
    assert ledger.active_facts(
        tenant_id="tenant-a", subject_id="user-1", now=expiry
    ) == ()


def test_temporal_view_does_not_apply_future_correction_early() -> None:
    ledger = ConversationMemoryLedger()
    previous = add_session_fact(ledger)
    corrected_at = NOW + timedelta(minutes=10)
    corrected = ledger.correct_fact(
        previous_fact_id=previous.fact_id,
        new_fact_id="fact-2",
        tenant_id="tenant-a",
        subject_id="user-1",
        value="en",
        source_event_id="correction-1",
        created_at=corrected_at,
        confidence=1.0,
    )

    before = corrected_at - timedelta(seconds=1)
    assert ledger.active_facts(
        tenant_id="tenant-a", subject_id="user-1", now=before
    ) == (previous,)
    assert ledger.status(
        fact_id=corrected.fact_id,
        tenant_id="tenant-a",
        subject_id="user-1",
        now=before,
    ) is MemoryStatus.NOT_YET_EFFECTIVE


def test_temporal_view_does_not_apply_future_retraction_early() -> None:
    ledger = ConversationMemoryLedger()
    fact = add_session_fact(ledger)
    retracted_at = NOW + timedelta(minutes=10)
    ledger.retract_fact(
        retraction_id="retract-1",
        fact_id=fact.fact_id,
        tenant_id="tenant-a",
        subject_id="user-1",
        source_event_id="delete-1",
        reason="user request",
        created_at=retracted_at,
    )

    assert ledger.active_facts(
        tenant_id="tenant-a",
        subject_id="user-1",
        now=retracted_at - timedelta(seconds=1),
    ) == (fact,)
    assert ledger.active_facts(
        tenant_id="tenant-a", subject_id="user-1", now=retracted_at
    ) == ()


def test_correction_and_retraction_cannot_predate_fact() -> None:
    ledger = ConversationMemoryLedger()
    fact = add_session_fact(ledger)
    before = NOW - timedelta(seconds=1)

    with pytest.raises(ValueError, match="cannot predate"):
        ledger.correct_fact(
            previous_fact_id=fact.fact_id,
            new_fact_id="fact-2",
            tenant_id="tenant-a",
            subject_id="user-1",
            value="en",
            source_event_id="correction-1",
            created_at=before,
            confidence=1.0,
        )
    with pytest.raises(ValueError, match="cannot predate"):
        ledger.retract_fact(
            retraction_id="retract-1",
            fact_id=fact.fact_id,
            tenant_id="tenant-a",
            subject_id="user-1",
            source_event_id="delete-1",
            reason="user request",
            created_at=before,
        )


def test_reads_are_tenant_and_subject_scoped() -> None:
    ledger = ConversationMemoryLedger()
    first = add_session_fact(ledger)
    second = add_session_fact(
        ledger,
        fact_id="fact-2",
        tenant_id="tenant-b",
        subject_id="user-2",
        value="en",
    )

    assert ledger.active_facts(
        tenant_id="tenant-a", subject_id="user-1", now=NOW
    ) == (first,)
    assert ledger.active_facts(
        tenant_id="tenant-b", subject_id="user-2", now=NOW
    ) == (second,)
    with pytest.raises(KeyError, match="tenant and subject scope"):
        ledger.status(
            fact_id=first.fact_id,
            tenant_id="tenant-b",
            subject_id="user-1",
            now=NOW,
        )


def test_delete_subject_does_not_delete_other_scope() -> None:
    ledger = ConversationMemoryLedger()
    add_session_fact(ledger)
    other = add_session_fact(
        ledger,
        fact_id="fact-2",
        tenant_id="tenant-a",
        subject_id="user-2",
        value="en",
    )

    assert ledger.delete_subject(tenant_id="tenant-a", subject_id="user-1") == 1
    assert ledger.history(tenant_id="tenant-a", subject_id="user-1") == ()
    assert ledger.active_facts(
        tenant_id="tenant-a", subject_id="user-2", now=NOW
    ) == (other,)


def test_duplicate_fact_id_is_rejected() -> None:
    ledger = ConversationMemoryLedger()
    add_session_fact(ledger)

    with pytest.raises(ValueError, match="fact_id already exists"):
        add_session_fact(ledger)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_confidence_is_rejected(confidence: float) -> None:
    ledger = ConversationMemoryLedger()
    with pytest.raises(ValueError, match="confidence"):
        ledger.add_fact(
            fact_id="fact-1",
            tenant_id="tenant-a",
            subject_id="user-1",
            key="preferred_language",
            value="zh-CN",
            kind=MemoryKind.WORKING,
            scope=MemoryScope.SESSION,
            source_event_id="message-1",
            created_at=NOW,
            confidence=confidence,
            policy_version="memory-policy-v1",
        )


def test_naive_timestamps_are_rejected() -> None:
    ledger = ConversationMemoryLedger()
    with pytest.raises(ValueError, match="timezone-aware"):
        ledger.add_fact(
            fact_id="fact-1",
            tenant_id="tenant-a",
            subject_id="user-1",
            key="preferred_language",
            value="zh-CN",
            kind=MemoryKind.WORKING,
            scope=MemoryScope.SESSION,
            source_event_id="message-1",
            created_at=datetime(2026, 8, 6, 4),
            confidence=1.0,
            policy_version="memory-policy-v1",
        )


def test_non_json_memory_value_is_rejected() -> None:
    ledger = ConversationMemoryLedger()
    with pytest.raises(TypeError, match="non-JSON"):
        add_session_fact(ledger, value={1, 2, 3})
