"""Adapters for learned dense retrieval and reranking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from about_llm.rag import (
    RerankReport,
    SearchResult,
    rerank_authorized_candidates,
)


class SentenceTransformerEmbedder:
    """Lazy optional adapter implementing the canonical EmbeddingModel protocol."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        device: str | None = None,
    ) -> None:
        if not model_name_or_path.strip():
            raise ValueError("model_name_or_path cannot be empty")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("dense model adapter requires: pip install -e '.[rag]'") from error
        self.model: Any = SentenceTransformer(
            model_name_or_path,
            revision=revision,
            device=device,
        )

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        values = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(values, dtype=np.float32)


class CrossEncoderReranker:
    """Rerank authorized candidates while preserving canonical documents."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise ImportError("reranker requires: pip install -e '.[rag]'") from error
        self.model: Any = CrossEncoder(
            model_name_or_path,
            revision=revision,
            device=device,
        )
        self.scorer_identity = (
            "sentence-transformers-cross-encoder:"
            f"{model_name_or_path}@{revision or 'unresolved-revision'}"
        )

    def score(
        self,
        query: str,
        candidates: Sequence[SearchResult],
    ) -> Sequence[float]:
        pairs = [(query, result.document.text) for result in candidates]
        values = np.asarray(self.model.predict(pairs)).reshape(-1)
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.bool_
        ):
            raise ValueError("cross-encoder scores must be real numbers, not booleans")
        return [float(value) for value in values]

    def rerank_with_report(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        tenant_id: str,
        principals: Iterable[str] = (),
        top_k: int,
    ) -> RerankReport:
        return rerank_authorized_candidates(
            query,
            candidates,
            self,
            tenant_id=tenant_id,
            principals=principals,
            top_k=top_k,
            scorer_identity=self.scorer_identity,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        tenant_id: str,
        principals: Iterable[str] = (),
        top_k: int,
    ) -> list[SearchResult]:
        return list(
            self.rerank_with_report(
                query,
                candidates,
                tenant_id=tenant_id,
                principals=principals,
                top_k=top_k,
            ).results
        )
