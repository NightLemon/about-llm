"""Dense retrieval with an injected embedding model and explicit ACL filtering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from about_llm.rag.models import Document, SearchResult

FloatMatrix = NDArray[np.float32]


class EmbeddingModel(Protocol):
    """The minimal contract needed by the canonical dense index."""

    def encode(self, texts: Sequence[str]) -> FloatMatrix: ...


def _normalize_rows(matrix: FloatMatrix, *, label: str) -> FloatMatrix:
    if matrix.ndim != 2:
        raise ValueError(f"{label} embeddings must be a 2D matrix")
    if matrix.shape[1] == 0:
        raise ValueError(f"{label} embedding dimension cannot be zero")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} embeddings must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{label} embeddings cannot contain zero vectors")
    return np.asarray(matrix / norms, dtype=np.float32)


class DenseIndex:
    """Small in-memory cosine index for teaching and deterministic tests."""

    def __init__(
        self,
        documents: Iterable[Document],
        embedding_model: EmbeddingModel,
    ) -> None:
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("at least one document is required")
        ids = [document.document_id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document_id values must be unique")
        self.embedding_model = embedding_model
        embeddings = np.asarray(
            embedding_model.encode([document.text for document in self.documents]),
            dtype=np.float32,
        )
        if embeddings.shape[0] != len(self.documents):
            raise ValueError(
                f"embedding model returned {embeddings.shape[0]} rows "
                f"for {len(self.documents)} documents"
            )
        self.embeddings = _normalize_rows(embeddings, label="document")

    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        principals: Iterable[str] = (),
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not query.strip():
            return []
        if not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        principal_set = set(principals)
        if any(not principal.strip() for principal in principal_set):
            raise ValueError("principals cannot contain an empty value")

        query_embedding = np.asarray(self.embedding_model.encode([query]), dtype=np.float32)
        query_embedding = _normalize_rows(query_embedding, label="query")
        if query_embedding.shape != (1, self.embeddings.shape[1]):
            raise ValueError(
                f"query embedding shape {query_embedding.shape} does not match "
                f"document dimension {self.embeddings.shape[1]}"
            )

        visible_indices = [
            index
            for index, document in enumerate(self.documents)
            if document.tenant_id == tenant_id
            and (not document.acl or not principal_set.isdisjoint(document.acl))
        ]
        if not visible_indices:
            return []
        visible_embeddings = self.embeddings[visible_indices]
        scores = visible_embeddings @ query_embedding[0]
        ranked = sorted(
            zip(scores.tolist(), visible_indices, strict=True),
            key=lambda item: (-item[0], self.documents[item[1]].document_id),
        )
        return [
            SearchResult(
                document=self.documents[index],
                score=float(score),
                rank=rank,
                source="dense",
            )
            for rank, (score, index) in enumerate(ranked[:top_k], start=1)
        ]
