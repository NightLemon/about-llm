from __future__ import annotations

import numpy as np
import pytest

from about_llm.rag import (
    contrastive_retrieval_loss,
    late_interaction_scores,
    single_positive_info_nce,
    splade_max_pool,
)


def test_contrastive_gradients_match_finite_difference() -> None:
    queries = np.array([[1.0, 0.2], [-0.3, 0.8]], dtype=np.float64)
    documents = np.array(
        [[0.9, 0.1], [0.0, 1.0], [-0.5, -0.4]], dtype=np.float64
    )
    report = single_positive_info_nce(
        queries,
        documents,
        [0, 1],
        temperature=0.7,
    )
    epsilon = 1e-6
    query_finite_difference = np.empty_like(queries)
    document_finite_difference = np.empty_like(documents)

    for row, column in np.ndindex(queries.shape):
        positive = queries.copy()
        negative = queries.copy()
        positive[row, column] += epsilon
        negative[row, column] -= epsilon
        query_finite_difference[row, column] = (
            single_positive_info_nce(
                positive, documents, [0, 1], temperature=0.7
            ).mean_loss
            - single_positive_info_nce(
                negative, documents, [0, 1], temperature=0.7
            ).mean_loss
        ) / (2 * epsilon)

    for row, column in np.ndindex(documents.shape):
        positive = documents.copy()
        negative = documents.copy()
        positive[row, column] += epsilon
        negative[row, column] -= epsilon
        document_finite_difference[row, column] = (
            single_positive_info_nce(
                queries, positive, [0, 1], temperature=0.7
            ).mean_loss
            - single_positive_info_nce(
                queries, negative, [0, 1], temperature=0.7
            ).mean_loss
        ) / (2 * epsilon)

    np.testing.assert_allclose(report.query_gradients, query_finite_difference, atol=1e-9)
    np.testing.assert_allclose(
        report.document_gradients, document_finite_difference, atol=1e-9
    )
    np.testing.assert_allclose(np.sum(report.probabilities, axis=1), np.ones(2))
    np.testing.assert_allclose(
        np.sum(report.logit_gradients, axis=1), np.zeros(2), atol=1e-15
    )


def test_hard_negative_and_temperature_change_the_objective() -> None:
    query = [[1.0, 0.0]]
    positive = [1.0, 0.0]
    easy_negative = [-1.0, 0.0]
    hard_negative = [0.9, 0.1]

    easy = single_positive_info_nce(query, [positive, easy_negative], [0])
    hard = single_positive_info_nce(query, [positive, hard_negative], [0])
    colder = single_positive_info_nce(
        query,
        [positive, hard_negative],
        [0],
        temperature=0.25,
    )

    assert hard.mean_loss > easy.mean_loss
    assert colder.mean_loss < hard.mean_loss
    assert hard.probabilities[0, 1] > easy.probabilities[0, 1]


def test_multi_positive_mask_exposes_false_negative_gradient() -> None:
    query = [[1.0, 0.0]]
    documents = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    single = single_positive_info_nce(query, documents, [0])
    multi = contrastive_retrieval_loss(
        query,
        documents,
        [[True, True, False]],
    )

    assert single.logit_gradients[0, 1] > 0
    assert multi.logit_gradients[0, 1] < 0
    assert multi.mean_loss < single.mean_loss
    np.testing.assert_allclose(
        multi.positive_conditional_probabilities,
        [[0.5, 0.5, 0.0]],
    )


def test_late_interaction_applies_maxsim_and_masks() -> None:
    query_tokens = [[1.0, 0.0], [0.0, 1.0]]
    document_tokens = [
        [[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]],
        [[0.8, 0.8], [0.0, 0.0], [0.0, 0.0]],
    ]
    document_mask = [[True, True, False], [True, False, False]]

    scores = late_interaction_scores(
        query_tokens,
        document_tokens,
        document_mask=document_mask,
    )
    first_query_token_only = late_interaction_scores(
        query_tokens,
        document_tokens,
        query_mask=[True, False],
        document_mask=document_mask,
    )

    np.testing.assert_allclose(scores, [2.0, 1.6])
    np.testing.assert_allclose(first_query_token_only, [1.0, 0.8])


def test_splade_pooling_is_non_negative_masked_vocabulary_max() -> None:
    logits = np.array(
        [
            [[1.0, -2.0, 0.5], [100.0, 100.0, 100.0]],
            [[0.0, 2.0, -1.0], [3.0, 1.0, 4.0]],
        ]
    )
    pooled = splade_max_pool(logits, [[True, False], [True, True]])

    expected = np.array(
        [
            [np.log1p(1.0), 0.0, np.log1p(0.5)],
            [np.log1p(3.0), np.log1p(2.0), np.log1p(4.0)],
        ]
    )
    np.testing.assert_allclose(pooled, expected)
    assert np.all(pooled >= 0)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: single_positive_info_nce([[True, False]], [[1.0, 0.0]], [0]),
            "not booleans",
        ),
        (
            lambda: single_positive_info_nce([[1.0]], [[1.0]], [1]),
            "out-of-range",
        ),
        (
            lambda: single_positive_info_nce([[1.0]], [[1.0]], [0], temperature=0),
            "positive",
        ),
        (
            lambda: single_positive_info_nce([[1e308]], [[1e308]], [0]),
            "dot products must remain finite",
        ),
        (
            lambda: contrastive_retrieval_loss([[1.0]], [[1.0]], [[False]]),
            "at least one positive",
        ),
        (
            lambda: contrastive_retrieval_loss([[1.0]], [[1.0]], [[1]]),
            "boolean",
        ),
        (
            lambda: late_interaction_scores(
                [[1.0]], [[[1.0]]], document_mask=[[False]]
            ),
            "at least one token",
        ),
        (
            lambda: splade_max_pool([[[1.0]]], [[False]]),
            "at least one token",
        ),
    ],
)
def test_retriever_learning_contracts_reject_invalid_inputs(
    operation: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assert callable(operation)
        operation()
