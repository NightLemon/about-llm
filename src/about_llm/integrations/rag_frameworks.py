"""Strict adapters from canonical RAG results to framework retriever objects.

The canonical index remains the security and ranking authority.  Framework
objects are transport values: callers can validate that a framework round trip
did not reorder results or alter content, ACL context, scores, or metadata.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from about_llm.rag import BM25Index, SearchResult

_PROTECTED_METADATA_KEYS = frozenset(
    {
        "document_id",
        "tenant_id",
        "acl",
        "retrieval_score",
        "retrieval_rank",
        "retriever",
    }
)


def _canonical_results(results: Sequence[SearchResult]) -> tuple[SearchResult, ...]:
    canonical = tuple(results)
    expected_ranks = tuple(range(1, len(canonical) + 1))
    actual_ranks = tuple(result.rank for result in canonical)
    if actual_ranks != expected_ranks:
        raise ValueError("results must be ordered with contiguous one-based ranks")
    document_ids = tuple(result.document.document_id for result in canonical)
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("results contain duplicate document ids")
    for result in canonical:
        if isinstance(result.score, bool) or not isinstance(result.score, (int, float)):
            raise TypeError("retrieval scores must be real numbers")
        if not math.isfinite(result.score):
            raise ValueError("retrieval scores must be finite")
    return canonical


def _result_metadata(result: SearchResult) -> dict[str, Any]:
    metadata = dict(result.document.metadata)
    collision = _PROTECTED_METADATA_KEYS & set(metadata)
    if collision:
        raise ValueError(f"document metadata uses protected keys: {sorted(collision)}")
    metadata.update(
        {
            "document_id": result.document.document_id,
            "tenant_id": result.document.tenant_id,
            "acl": list(result.document.acl),
            "retrieval_score": result.score,
            "retrieval_rank": result.rank,
            "retriever": result.source,
        }
    )
    return metadata


def to_langchain_documents(results: Sequence[SearchResult]) -> list[Any]:
    """Convert results while preserving ids, ACL context, rank, and score."""
    try:
        from langchain_core.documents import Document as LangChainDocument
    except ImportError as error:
        raise ImportError("LangChain adapter requires: pip install -e '.[langchain]'") from error
    canonical = _canonical_results(results)
    return [
        LangChainDocument(
            id=result.document.document_id,
            page_content=result.document.text,
            metadata=_result_metadata(result),
        )
        for result in canonical
    ]


def to_llamaindex_nodes(results: Sequence[SearchResult]) -> list[Any]:
    """Convert results to NodeWithScore without rerunning retrieval."""
    try:
        from llama_index.core.schema import NodeWithScore, TextNode
    except ImportError as error:
        raise ImportError("LlamaIndex adapter requires: pip install -e '.[llamaindex]'") from error
    canonical = _canonical_results(results)
    excluded_metadata = sorted(_PROTECTED_METADATA_KEYS)
    return [
        NodeWithScore(
            node=TextNode(
                id_=result.document.document_id,
                text=result.document.text,
                metadata=_result_metadata(result),
                excluded_embed_metadata_keys=excluded_metadata,
                excluded_llm_metadata_keys=excluded_metadata,
            ),
            score=result.score,
        )
        for result in canonical
    ]


def validate_langchain_round_trip(
    documents: Sequence[Any],
    expected: Sequence[SearchResult],
) -> tuple[SearchResult, ...]:
    """Fail closed unless LangChain documents exactly match canonical results."""
    canonical = _canonical_results(expected)
    if len(documents) != len(canonical):
        raise ValueError("LangChain result count differs from canonical retrieval")
    for position, (document, result) in enumerate(zip(documents, canonical, strict=True), start=1):
        if document.id != result.document.document_id:
            raise ValueError(f"LangChain document id drift at result {position}")
        if document.page_content != result.document.text:
            raise ValueError(f"LangChain document text drift at result {position}")
        if document.metadata != _result_metadata(result):
            raise ValueError(f"LangChain metadata drift at result {position}")
    return canonical


def validate_llamaindex_round_trip(
    nodes: Sequence[Any],
    expected: Sequence[SearchResult],
) -> tuple[SearchResult, ...]:
    """Fail closed unless LlamaIndex nodes exactly match canonical results."""
    canonical = _canonical_results(expected)
    if len(nodes) != len(canonical):
        raise ValueError("LlamaIndex result count differs from canonical retrieval")
    excluded_metadata = sorted(_PROTECTED_METADATA_KEYS)
    for position, (item, result) in enumerate(zip(nodes, canonical, strict=True), start=1):
        if item.node.node_id != result.document.document_id:
            raise ValueError(f"LlamaIndex node id drift at result {position}")
        if item.node.text != result.document.text:
            raise ValueError(f"LlamaIndex node text drift at result {position}")
        if item.node.metadata != _result_metadata(result):
            raise ValueError(f"LlamaIndex metadata drift at result {position}")
        if item.score != result.score:
            raise ValueError(f"LlamaIndex score drift at result {position}")
        if item.node.excluded_embed_metadata_keys != excluded_metadata:
            raise ValueError("LlamaIndex embedding metadata exclusion drift")
        if item.node.excluded_llm_metadata_keys != excluded_metadata:
            raise ValueError("LlamaIndex LLM metadata exclusion drift")
    return canonical


def _security_context(
    tenant_id: str,
    principals: Iterable[str],
    top_k: int,
) -> tuple[str, tuple[str, ...], int]:
    if not tenant_id.strip():
        raise ValueError("tenant_id cannot be empty")
    principal_tuple = tuple(principals)
    if any(not principal.strip() for principal in principal_tuple):
        raise ValueError("principals cannot contain an empty value")
    if len(principal_tuple) != len(set(principal_tuple)):
        raise ValueError("principals cannot contain duplicates")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return tenant_id, principal_tuple, top_k


def build_langchain_retriever(
    index: BM25Index,
    *,
    tenant_id: str,
    principals: Iterable[str] = (),
    top_k: int = 5,
) -> Any:
    """Bind canonical authorization and ranking behind LangChain's Retriever API."""
    try:
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
        from langchain_core.retrievers import BaseRetriever
    except ImportError as error:
        raise ImportError("LangChain adapter requires: pip install -e '.[langchain]'") from error
    bound_tenant, bound_principals, bound_top_k = _security_context(
        tenant_id, principals, top_k
    )

    class CanonicalLangChainRetriever(BaseRetriever):
        canonical_index: Any
        bound_tenant_id: str
        bound_principals: tuple[str, ...]
        bound_top_k: int

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> list[Any]:
            del run_manager
            results = self.canonical_index.search(
                query,
                tenant_id=self.bound_tenant_id,
                principals=self.bound_principals,
                top_k=self.bound_top_k,
            )
            return to_langchain_documents(results)

    return CanonicalLangChainRetriever(
        canonical_index=index,
        bound_tenant_id=bound_tenant,
        bound_principals=bound_principals,
        bound_top_k=bound_top_k,
    )


def build_llamaindex_retriever(
    index: BM25Index,
    *,
    tenant_id: str,
    principals: Iterable[str] = (),
    top_k: int = 5,
) -> Any:
    """Bind canonical authorization and ranking behind LlamaIndex's Retriever API."""
    try:
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import QueryBundle
    except ImportError as error:
        raise ImportError("LlamaIndex adapter requires: pip install -e '.[llamaindex]'") from error
    bound_tenant, bound_principals, bound_top_k = _security_context(
        tenant_id, principals, top_k
    )

    class CanonicalLlamaIndexRetriever(BaseRetriever):
        def __init__(self) -> None:
            super().__init__()
            self._canonical_index = index
            self._bound_tenant_id = bound_tenant
            self._bound_principals = bound_principals
            self._bound_top_k = bound_top_k

        def _retrieve(self, query_bundle: QueryBundle) -> list[Any]:
            results = self._canonical_index.search(
                query_bundle.query_str,
                tenant_id=self._bound_tenant_id,
                principals=self._bound_principals,
                top_k=self._bound_top_k,
            )
            return to_llamaindex_nodes(results)

    return CanonicalLlamaIndexRetriever()
