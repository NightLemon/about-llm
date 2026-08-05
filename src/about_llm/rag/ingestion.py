"""Deterministic Markdown ingestion and incremental index planning."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[\u3002\uff01\uff1f!?\uff1b;\.])\s+")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


@dataclass(frozen=True)
class SourceDocument:
    """One versioned source inside a tenant security boundary."""

    source_id: str
    tenant_id: str
    version: str
    text: str
    acl: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source_id", "tenant_id", "version", "text"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if len(self.acl) != len(set(self.acl)):
            raise ValueError("acl contains duplicate principals")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class SourceChunk:
    """An indexable chunk whose id survives unrelated insertions."""

    chunk_id: str
    content_hash: str
    source_id: str
    tenant_id: str
    source_version: str
    ordinal: int
    text: str
    heading_path: tuple[str, ...]
    acl: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not self.text.strip():
            raise ValueError("chunk text cannot be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class IngestionPlan:
    upsert: tuple[SourceChunk, ...]
    delete_chunk_ids: tuple[str, ...]
    unchanged_chunk_ids: tuple[str, ...]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split an oversized block without silently dropping any characters."""
    sentences = [part for part in _SENTENCE_BOUNDARY.split(text) if part]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                sentence[index : index + max_chars]
                for index in range(0, len(sentence), max_chars)
            )
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _markdown_blocks(text: str) -> list[tuple[tuple[str, ...], str]]:
    headings: list[str] = []
    blocks: list[tuple[tuple[str, ...], str]] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        block = _normalise("\n".join(buffer))
        if block:
            blocks.append((tuple(headings), block))
        buffer.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            in_fence = not in_fence
            buffer.append(line)
            continue
        heading = None if in_fence else _HEADING.match(line)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            headings[level - 1 :] = []
            while len(headings) < level - 1:
                headings.append("(untitled)")
            headings.append(title)
        elif not line.strip() and not in_fence:
            flush()
        else:
            buffer.append(line)
    flush()
    return blocks


def split_markdown(source: SourceDocument, *, max_chars: int = 1200) -> list[SourceChunk]:
    """Split by Markdown structure, then bound oversized blocks deterministically."""
    if max_chars < 64:
        raise ValueError("max_chars must be at least 64")
    raw: list[tuple[tuple[str, ...], str]] = []
    for heading_path, block in _markdown_blocks(source.text):
        raw.extend((heading_path, piece) for piece in _hard_split(block, max_chars))

    occurrences: dict[tuple[tuple[str, ...], str], int] = {}
    chunks: list[SourceChunk] = []
    for ordinal, (heading_path, text) in enumerate(raw):
        content_hash = _digest(text)
        duplicate_key = heading_path, content_hash
        occurrence = occurrences.get(duplicate_key, 0)
        occurrences[duplicate_key] = occurrence + 1
        identity = "\x1f".join(
            (
                source.tenant_id,
                source.source_id,
                " / ".join(heading_path),
                content_hash,
                str(occurrence),
            )
        )
        chunks.append(
            SourceChunk(
                chunk_id=f"chk_{_digest(identity)[:24]}",
                content_hash=content_hash,
                source_id=source.source_id,
                tenant_id=source.tenant_id,
                source_version=source.version,
                ordinal=ordinal,
                text=text,
                heading_path=heading_path,
                acl=source.acl,
                metadata=source.metadata,
            )
        )
    return chunks


def plan_incremental_update(
    existing: Iterable[SourceChunk], desired: Iterable[SourceChunk]
) -> IngestionPlan:
    """Return explicit index writes/deletes; never infer deletion from an empty batch."""
    existing_chunks = list(existing)
    desired_chunks = list(desired)
    old = {chunk.chunk_id: chunk for chunk in existing_chunks}
    new = {chunk.chunk_id: chunk for chunk in desired_chunks}
    if len(old) != len(existing_chunks) or len(new) != len(desired_chunks):
        raise ValueError("duplicate chunk_id in ingestion input")

    unchanged = tuple(
        sorted(
            chunk_id
            for chunk_id in old.keys() & new.keys()
            if old[chunk_id] == new[chunk_id]
        )
    )
    upsert = tuple(
        sorted(
            (chunk for chunk_id, chunk in new.items() if old.get(chunk_id) != chunk),
            key=lambda chunk: chunk.ordinal,
        )
    )
    deleted = tuple(sorted(old.keys() - new.keys()))
    return IngestionPlan(upsert=upsert, delete_chunk_ids=deleted, unchanged_chunk_ids=unchanged)
