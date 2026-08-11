"""Authorized context rendering and deliberately syntactic citation auditing."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from about_llm.rag.models import Document, SearchResult

_CITATION = re.compile(r"\[([A-Z][A-Z0-9_-]*\d+)\]")


@dataclass(frozen=True)
class CitationContext:
    rendered: str
    sources: Mapping[str, Document]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))


@dataclass(frozen=True)
class CitationAudit:
    cited_source_ids: tuple[str, ...]
    unknown_source_ids: tuple[str, ...]
    uncited_paragraphs: tuple[str, ...]

    @property
    def syntactically_valid(self) -> bool:
        return not self.unknown_source_ids and not self.uncited_paragraphs


def build_citation_context(
    results: Iterable[SearchResult],
    *,
    tenant_id: str,
    principals: Iterable[str] = (),
    prefix: str = "S",
) -> CitationContext:
    """Render only authorized results and assign compact canonical source ids."""
    if not tenant_id.strip():
        raise ValueError("tenant_id cannot be empty")
    if not prefix.isalpha() or not prefix.isupper():
        raise ValueError("prefix must contain uppercase letters only")
    principal_set = set(principals)
    if any(not principal.strip() for principal in principal_set):
        raise ValueError("principals cannot contain an empty value")

    sources: dict[str, Document] = {}
    seen_documents: set[str] = set()
    rendered: list[str] = []
    for result in results:
        document = result.document
        if document.tenant_id != tenant_id:
            raise PermissionError(
                f"result {document.document_id!r} belongs to tenant {document.tenant_id!r}"
            )
        if document.acl and principal_set.isdisjoint(document.acl):
            raise PermissionError(
                f"result {document.document_id!r} is not visible to caller principals"
            )
        if document.document_id in seen_documents:
            continue
        seen_documents.add(document.document_id)
        source_id = f"{prefix}{len(sources) + 1}"
        sources[source_id] = document
        rendered.append(
            f'<source id="{source_id}" document_id="{document.document_id}">\n'
            f"{document.text}\n</source>"
        )
    return CitationContext(rendered="\n\n".join(rendered), sources=sources)


def audit_citations(answer: str, valid_source_ids: Iterable[str]) -> CitationAudit:
    """Check citation presence/ids only; this does not prove claim entailment."""
    valid = set(valid_source_ids)
    cited = tuple(dict.fromkeys(_CITATION.findall(answer)))
    unknown = tuple(source_id for source_id in cited if source_id not in valid)
    uncited: list[str] = []
    for paragraph in re.split(r"\n\s*\n", answer.strip()):
        stripped = paragraph.strip()
        if not stripped or stripped.startswith("```") or stripped.startswith("~~~"):
            continue
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", stripped):
            continue
        if not _CITATION.search(stripped):
            uncited.append(stripped)
    return CitationAudit(cited, unknown, tuple(uncited))
