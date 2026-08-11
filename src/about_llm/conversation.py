"""Typed, source-linked conversation memory reference primitives.

This module is deliberately an in-memory reference core. It demonstrates state
invariants that a persistent service must preserve, but it does not provide a
database, encryption, distributed consistency, retention jobs, or production
authorization by itself.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import cast

from about_llm.llmops import JsonValue, canonical_json_bytes


class MemoryKind(str, Enum):
    """Why a memory exists, not how long it is stored."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryScope(str, Enum):
    """Where a memory may be reused."""

    SESSION = "session"
    PROFILE = "profile"


class MemoryStatus(str, Enum):
    """Derived state of an append-only fact at a particular time."""

    ACTIVE = "active"
    NOT_YET_EFFECTIVE = "not_yet_effective"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    EXPIRED = "expired"


def _require_nonempty(name: str, value: str) -> None:
    if not value or value.isspace():
        raise ValueError(f"{name} must not be empty")


def _utc(name: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MemoryFact:
    """Immutable memory event with a canonical JSON value snapshot."""

    fact_id: str
    tenant_id: str
    subject_id: str
    key: str
    kind: MemoryKind
    scope: MemoryScope
    source_event_id: str
    created_at: datetime
    confidence: float
    policy_version: str
    expires_at: datetime | None = None
    consent_reference: str | None = None
    supersedes_fact_id: str | None = None
    _canonical_value: bytes = field(default=b"", repr=False)

    def __post_init__(self) -> None:
        for name in (
            "fact_id",
            "tenant_id",
            "subject_id",
            "key",
            "source_event_id",
            "policy_version",
        ):
            _require_nonempty(name, cast(str, getattr(self, name)))
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be a real number")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        created_at = _utc("created_at", self.created_at)
        object.__setattr__(self, "created_at", created_at)
        if self.expires_at is not None:
            expires_at = _utc("expires_at", self.expires_at)
            if expires_at <= created_at:
                raise ValueError("expires_at must be later than created_at")
            object.__setattr__(self, "expires_at", expires_at)
        if self.scope is MemoryScope.PROFILE:
            if self.consent_reference is None:
                raise ValueError("profile memory requires a consent_reference")
            _require_nonempty("consent_reference", self.consent_reference)
        if self.supersedes_fact_id is not None:
            _require_nonempty("supersedes_fact_id", self.supersedes_fact_id)
            if self.supersedes_fact_id == self.fact_id:
                raise ValueError("a fact cannot supersede itself")
        try:
            decoded = cast(JsonValue, json.loads(self._canonical_value))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("_canonical_value must contain valid UTF-8 JSON") from error
        if canonical_json_bytes(decoded) != self._canonical_value:
            raise ValueError("_canonical_value must use canonical JSON serialization")

    @classmethod
    def create(
        cls,
        *,
        fact_id: str,
        tenant_id: str,
        subject_id: str,
        key: str,
        value: object,
        kind: MemoryKind,
        scope: MemoryScope,
        source_event_id: str,
        created_at: datetime,
        confidence: float,
        policy_version: str,
        expires_at: datetime | None = None,
        consent_reference: str | None = None,
        supersedes_fact_id: str | None = None,
    ) -> MemoryFact:
        return cls(
            fact_id=fact_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            key=key,
            kind=kind,
            scope=scope,
            source_event_id=source_event_id,
            created_at=created_at,
            confidence=confidence,
            policy_version=policy_version,
            expires_at=expires_at,
            consent_reference=consent_reference,
            supersedes_fact_id=supersedes_fact_id,
            _canonical_value=canonical_json_bytes(value),
        )

    @property
    def value(self) -> JsonValue:
        """Return a defensive JSON copy of the recorded value."""

        return cast(JsonValue, json.loads(self._canonical_value))


@dataclass(frozen=True)
class MemoryRetraction:
    """An auditable event that makes one fact inactive."""

    retraction_id: str
    fact_id: str
    tenant_id: str
    subject_id: str
    source_event_id: str
    reason: str
    created_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "retraction_id",
            "fact_id",
            "tenant_id",
            "subject_id",
            "source_event_id",
            "reason",
        ):
            _require_nonempty(name, cast(str, getattr(self, name)))
        object.__setattr__(self, "created_at", _utc("created_at", self.created_at))


class ConversationMemoryLedger:
    """In-memory reference ledger with tenant-scoped reads and corrections."""

    def __init__(self) -> None:
        self._facts: dict[str, MemoryFact] = {}
        self._retractions: dict[str, MemoryRetraction] = {}

    def _fact_for_subject(
        self, *, fact_id: str, tenant_id: str, subject_id: str
    ) -> MemoryFact:
        fact = self._facts.get(fact_id)
        if fact is None or fact.tenant_id != tenant_id or fact.subject_id != subject_id:
            raise KeyError("fact does not exist in this tenant and subject scope")
        return fact

    def _superseded_ids(self, *, now: datetime) -> set[str]:
        return {
            fact.supersedes_fact_id
            for fact in self._facts.values()
            if fact.supersedes_fact_id is not None and fact.created_at <= now
        }

    def _retracted_ids(self, *, now: datetime) -> set[str]:
        return {
            event.fact_id
            for event in self._retractions.values()
            if event.created_at <= now
        }

    def status(
        self,
        *,
        fact_id: str,
        tenant_id: str,
        subject_id: str,
        now: datetime,
    ) -> MemoryStatus:
        fact = self._fact_for_subject(
            fact_id=fact_id, tenant_id=tenant_id, subject_id=subject_id
        )
        observed_at = _utc("now", now)
        if fact.created_at > observed_at:
            return MemoryStatus.NOT_YET_EFFECTIVE
        if fact_id in self._retracted_ids(now=observed_at):
            return MemoryStatus.RETRACTED
        if fact_id in self._superseded_ids(now=observed_at):
            return MemoryStatus.SUPERSEDED
        if fact.expires_at is not None and fact.expires_at <= observed_at:
            return MemoryStatus.EXPIRED
        return MemoryStatus.ACTIVE

    def add_fact(
        self,
        *,
        fact_id: str,
        tenant_id: str,
        subject_id: str,
        key: str,
        value: object,
        kind: MemoryKind,
        scope: MemoryScope,
        source_event_id: str,
        created_at: datetime,
        confidence: float,
        policy_version: str,
        expires_at: datetime | None = None,
        consent_reference: str | None = None,
        supersedes_fact_id: str | None = None,
    ) -> MemoryFact:
        if fact_id in self._facts:
            raise ValueError("fact_id already exists")
        created_at = _utc("created_at", created_at)
        if supersedes_fact_id is not None:
            previous = self._fact_for_subject(
                fact_id=supersedes_fact_id,
                tenant_id=tenant_id,
                subject_id=subject_id,
            )
            if previous.key != key:
                raise ValueError("a correction must preserve the memory key")
            if created_at < previous.created_at:
                raise ValueError("a correction cannot predate the fact it supersedes")
            if self.status(
                fact_id=previous.fact_id,
                tenant_id=tenant_id,
                subject_id=subject_id,
                now=created_at,
            ) is not MemoryStatus.ACTIVE:
                raise ValueError("only an active fact can be superseded")
        for active in self.active_facts(
            tenant_id=tenant_id, subject_id=subject_id, now=created_at
        ):
            if active.key == key and active.fact_id != supersedes_fact_id:
                raise ValueError("an active value for this key already exists; supersede it")
        fact = MemoryFact.create(
            fact_id=fact_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            key=key,
            value=value,
            kind=kind,
            scope=scope,
            source_event_id=source_event_id,
            created_at=created_at,
            confidence=confidence,
            policy_version=policy_version,
            expires_at=expires_at,
            consent_reference=consent_reference,
            supersedes_fact_id=supersedes_fact_id,
        )
        self._facts[fact.fact_id] = fact
        return fact

    def correct_fact(
        self,
        *,
        previous_fact_id: str,
        new_fact_id: str,
        tenant_id: str,
        subject_id: str,
        value: object,
        source_event_id: str,
        created_at: datetime,
        confidence: float,
        expires_at: datetime | None = None,
        consent_reference: str | None = None,
    ) -> MemoryFact:
        previous = self._fact_for_subject(
            fact_id=previous_fact_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
        )
        return self.add_fact(
            fact_id=new_fact_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            key=previous.key,
            value=value,
            kind=previous.kind,
            scope=previous.scope,
            source_event_id=source_event_id,
            created_at=created_at,
            confidence=confidence,
            policy_version=previous.policy_version,
            expires_at=expires_at,
            consent_reference=consent_reference,
            supersedes_fact_id=previous_fact_id,
        )

    def retract_fact(
        self,
        *,
        retraction_id: str,
        fact_id: str,
        tenant_id: str,
        subject_id: str,
        source_event_id: str,
        reason: str,
        created_at: datetime,
    ) -> MemoryRetraction:
        if retraction_id in self._retractions:
            raise ValueError("retraction_id already exists")
        fact = self._fact_for_subject(
            fact_id=fact_id, tenant_id=tenant_id, subject_id=subject_id
        )
        created_at = _utc("created_at", created_at)
        if created_at < fact.created_at:
            raise ValueError("a retraction cannot predate its fact")
        if self.status(
            fact_id=fact.fact_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            now=created_at,
        ) is not MemoryStatus.ACTIVE:
            raise ValueError("only an active fact can be retracted")
        event = MemoryRetraction(
            retraction_id=retraction_id,
            fact_id=fact_id,
            tenant_id=tenant_id,
            subject_id=subject_id,
            source_event_id=source_event_id,
            reason=reason,
            created_at=created_at,
        )
        self._retractions[event.retraction_id] = event
        return event

    def active_facts(
        self, *, tenant_id: str, subject_id: str, now: datetime
    ) -> tuple[MemoryFact, ...]:
        observed_at = _utc("now", now)
        facts = (
            fact
            for fact in self._facts.values()
            if fact.tenant_id == tenant_id and fact.subject_id == subject_id
        )
        active = [
            fact
            for fact in facts
            if self.status(
                fact_id=fact.fact_id,
                tenant_id=tenant_id,
                subject_id=subject_id,
                now=observed_at,
            )
            is MemoryStatus.ACTIVE
        ]
        return tuple(sorted(active, key=lambda fact: (fact.created_at, fact.fact_id)))

    def history(self, *, tenant_id: str, subject_id: str) -> tuple[MemoryFact, ...]:
        """Return immutable facts for one scope, including inactive facts."""

        facts = (
            fact
            for fact in self._facts.values()
            if fact.tenant_id == tenant_id and fact.subject_id == subject_id
        )
        return tuple(sorted(facts, key=lambda fact: (fact.created_at, fact.fact_id)))

    def delete_subject(self, *, tenant_id: str, subject_id: str) -> int:
        """Physically remove one subject from this in-memory reference store."""

        fact_ids = {
            fact.fact_id
            for fact in self._facts.values()
            if fact.tenant_id == tenant_id and fact.subject_id == subject_id
        }
        for fact_id in fact_ids:
            del self._facts[fact_id]
        retraction_ids = [
            event_id
            for event_id, event in self._retractions.items()
            if event.tenant_id == tenant_id and event.subject_id == subject_id
        ]
        for event_id in retraction_ids:
            del self._retractions[event_id]
        return len(fact_ids)


__all__ = [
    "ConversationMemoryLedger",
    "MemoryFact",
    "MemoryKind",
    "MemoryRetraction",
    "MemoryScope",
    "MemoryStatus",
]
