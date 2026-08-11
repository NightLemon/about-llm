"""Typed domain models shared by RAG implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Document:
    """A retrievable unit with an explicit security boundary."""

    document_id: str
    text: str
    tenant_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    acl: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not self.text.strip():
            raise ValueError("text cannot be empty")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id cannot be empty")
        if any(not principal.strip() for principal in self.acl):
            raise ValueError("acl principals cannot be empty")
        if len(self.acl) != len(set(self.acl)):
            raise ValueError("acl contains duplicate principals")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
    rank: int
    source: str

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank is one-based and must be positive")
        if not self.source:
            raise ValueError("source cannot be empty")
