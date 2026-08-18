"""A small BM25 implementation with tenant filtering before ranking."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from about_llm.rag.models import Document, SearchResult
from about_llm.rag.tokenization import lexical_tokens

BM25_AUTHORIZED_STATISTICS_IMPLEMENTATION = (
    "about_llm.rag.BM25Index/authorized-statistics-v2"
)
BM25_LEGACY_GLOBAL_STATISTICS_IMPLEMENTATION = "about_llm.rag.BM25Index"


class BM25Index:
    """In-memory BM25 whose query statistics use only authorized documents."""

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        _legacy_global_statistics: bool = False,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("at least one document is required")
        ids = [document.document_id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("document_id values must be unique")

        self.k1 = k1
        self.b = b
        self._legacy_global_statistics = _legacy_global_statistics
        self.term_frequencies = tuple(
            Counter(lexical_tokens(document.text)) for document in self.documents
        )
        self.lengths = tuple(sum(frequencies.values()) for frequencies in self.term_frequencies)
        self.average_length, self.inverse_document_frequency = self._statistics(
            tuple(range(len(self.documents)))
        )

    @classmethod
    def _for_legacy_global_statistics(
        cls,
        documents: Iterable[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        """Reconstruct pre-v2 recorded artifacts; never use for new retrieval."""

        return cls(documents, k1=k1, b=b, _legacy_global_statistics=True)

    def _statistics(
        self, indices: tuple[int, ...]
    ) -> tuple[float, dict[str, float]]:
        if not indices:
            raise ValueError("BM25 statistics require at least one visible document")
        average_length = sum(self.lengths[index] for index in indices) / len(indices)
        document_frequency: Counter[str] = Counter()
        for index in indices:
            frequencies = self.term_frequencies[index]
            document_frequency.update(frequencies.keys())
        number_of_documents = len(indices)
        inverse_document_frequency = {
            term: math.log(1 + (number_of_documents - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return average_length, inverse_document_frequency

    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        principals: Iterable[str] = (),
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rank only documents visible to tenant_id and caller principals.

        Filtering after retrieval can leak document existence through scores,
        timing, caches, or generated answers. The baseline therefore applies
        tenant and principal boundaries before computing IDF, average length,
        or per-document scores. An empty document ACL means public inside the
        tenant; a non-empty ACL requires a matching caller principal.
        """
        if not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        principal_set = set(principals)
        if any(not principal.strip() for principal in principal_set):
            raise ValueError("principals cannot contain an empty value")
        query_terms = lexical_tokens(query)
        if not query_terms:
            return []

        visible_indices = tuple(
            index
            for index, document in enumerate(self.documents)
            if document.tenant_id == tenant_id
            and (not document.acl or not principal_set.isdisjoint(document.acl))
        )
        if not visible_indices:
            return []
        if self._legacy_global_statistics:
            average_length = self.average_length
            inverse_document_frequency = self.inverse_document_frequency
        else:
            average_length, inverse_document_frequency = self._statistics(visible_indices)

        scored: list[tuple[float, Document]] = []
        for index in visible_indices:
            document = self.documents[index]
            score = self._score(
                query_terms,
                self.term_frequencies[index],
                self.lengths[index],
                average_length=average_length,
                inverse_document_frequency=inverse_document_frequency,
            )
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].document_id))
        return [
            SearchResult(document=document, score=score, rank=rank, source="bm25")
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]

    def _score(
        self,
        query_terms: list[str],
        frequencies: Counter[str],
        length: int,
        *,
        average_length: float,
        inverse_document_frequency: dict[str, float],
    ) -> float:
        score = 0.0
        length_normalization = 1 - self.b + self.b * length / average_length
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * length_normalization
            score += inverse_document_frequency.get(term, 0.0) * numerator / denominator
        return score
