from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from about_llm.rag import DenseIndex, Document

pytestmark = pytest.mark.security


class ControlledEmbedder:
    def __init__(self, values: dict[str, tuple[float, ...]]) -> None:
        self.values = values

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray([self.values[text] for text in texts], dtype=np.float32)


def test_dense_index_normalizes_and_filters_tenant_before_ranking() -> None:
    documents = [
        Document("visible-best", "visible best", "a"),
        Document("visible-second", "visible second", "a"),
        Document("secret", "secret closest", "b"),
    ]
    embedder = ControlledEmbedder(
        {
            "visible best": (8, 2),
            "visible second": (1, 1),
            "secret closest": (100, 0),
            "query": (1, 0),
        }
    )
    index = DenseIndex(documents, embedder)
    results = index.search("query", tenant_id="a")

    assert [result.document.document_id for result in results] == [
        "visible-best",
        "visible-second",
    ]
    assert all(result.document.tenant_id == "a" for result in results)
    assert results[0].score <= 1


def test_dense_index_rejects_zero_and_mismatched_embeddings() -> None:
    document = Document("d", "doc", "a")
    with pytest.raises(ValueError, match="zero vectors"):
        DenseIndex([document], ControlledEmbedder({"doc": (0, 0)}))

    index = DenseIndex([document], ControlledEmbedder({"doc": (1, 0), "q": (1, 0, 0)}))
    with pytest.raises(ValueError, match="does not match"):
        index.search("q", tenant_id="a")


def test_dense_index_filters_principal_acl_before_ranking() -> None:
    documents = [
        Document("public", "public", "a"),
        Document("restricted", "restricted", "a", acl=("eng",)),
    ]
    embedder = ControlledEmbedder(
        {"public": (1, 0), "restricted": (2, 0), "query": (1, 0)}
    )
    index = DenseIndex(documents, embedder)

    anonymous = index.search("query", tenant_id="a")
    engineer = index.search("query", tenant_id="a", principals=("eng",))

    assert [result.document.document_id for result in anonymous] == ["public"]
    assert {result.document.document_id for result in engineer} == {
        "public",
        "restricted",
    }
