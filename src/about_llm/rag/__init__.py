"""Framework-independent RAG building blocks."""

from about_llm.rag.bm25 import BM25Index
from about_llm.rag.citations import (
    CitationAudit,
    CitationContext,
    audit_citations,
    build_citation_context,
)
from about_llm.rag.dense import DenseIndex, EmbeddingModel
from about_llm.rag.ingestion import (
    IngestionPlan,
    SourceChunk,
    SourceDocument,
    plan_incremental_update,
    split_markdown,
)
from about_llm.rag.models import Document, SearchResult
from about_llm.rag.rank_fusion import reciprocal_rank_fusion

__all__ = [
    "BM25Index",
    "CitationAudit",
    "CitationContext",
    "DenseIndex",
    "Document",
    "EmbeddingModel",
    "IngestionPlan",
    "SearchResult",
    "SourceChunk",
    "SourceDocument",
    "audit_citations",
    "build_citation_context",
    "plan_incremental_update",
    "reciprocal_rank_fusion",
    "split_markdown",
]
