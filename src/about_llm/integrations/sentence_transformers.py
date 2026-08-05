"""Adapters for learned dense retrieval and reranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from about_llm.rag import SearchResult


class SentenceTransformerEmbedder:
    """Lazy optional adapter implementing the canonical EmbeddingModel protocol."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
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

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []
        pairs = [(query, result.document.text) for result in candidates]
        scores = np.asarray(self.model.predict(pairs), dtype=np.float64).reshape(-1)
        if scores.shape[0] != len(candidates) or not np.all(np.isfinite(scores)):
            raise ValueError("reranker returned invalid scores")
        ordered = sorted(
            zip(scores.tolist(), candidates, strict=True),
            key=lambda item: (-item[0], item[1].document.document_id),
        )
        return [
            SearchResult(
                document=result.document,
                score=float(score),
                rank=rank,
                source=f"rerank:{result.source}",
            )
            for rank, (score, result) in enumerate(ordered[:top_k], start=1)
        ]
