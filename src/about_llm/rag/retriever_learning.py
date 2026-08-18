"""Exact NumPy controls for learned retrieval objectives and interactions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ContrastiveRetrievalReport:
    """A mean-reduced multi-positive contrastive retrieval objective.

    Gradients are with respect to the supplied, unnormalized embedding matrices.
    They are exact analytic gradients for this finite control, not gradients
    through an encoder, pooling operation, or embedding normalization layer.
    """

    scores: NDArray[np.float64]
    logits: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    positive_conditional_probabilities: NDArray[np.float64]
    positive_mask: NDArray[np.bool_]
    per_query_losses: NDArray[np.float64]
    mean_loss: float
    logit_gradients: NDArray[np.float64]
    query_gradients: NDArray[np.float64]
    document_gradients: NDArray[np.float64]
    temperature: float


def _finite_matrix(values: ArrayLike, label: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numeric values, not booleans")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{label} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _finite_tensor(values: ArrayLike, label: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numeric values, not booleans")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 3 or 0 in array.shape:
        raise ValueError(f"{label} must be a non-empty three-dimensional tensor")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def _positive_scalar(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _boolean_mask(
    values: ArrayLike,
    *,
    shape: tuple[int, ...],
    label: str,
) -> NDArray[np.bool_]:
    raw = np.asarray(values)
    if raw.dtype.kind != "b":
        raise ValueError(f"{label} must contain boolean values")
    mask = np.asarray(raw, dtype=np.bool_)
    if mask.shape != shape:
        raise ValueError(f"{label} shape {mask.shape} does not match expected {shape}")
    return mask


def _row_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    unnormalized = np.exp(shifted)
    return np.asarray(
        unnormalized / np.sum(unnormalized, axis=1, keepdims=True),
        dtype=np.float64,
    )


def contrastive_retrieval_loss(
    query_embeddings: ArrayLike,
    document_embeddings: ArrayLike,
    positive_mask: ArrayLike,
    *,
    temperature: float = 1.0,
) -> ContrastiveRetrievalReport:
    r"""Compute an exact multi-positive InfoNCE-style retrieval objective.

    For query ``i`` and its non-empty positive set ``P_i`` the per-query loss is

    ``log(sum_j exp(score_ij / temperature))
      - log(sum_{p in P_i} exp(score_ip / temperature))``.

    With one positive per query this is ordinary softmax cross-entropy over the
    supplied candidate documents. Every unmarked document is a negative for that
    query, so the candidate construction and positive mask are part of the
    training objective rather than incidental data-loader details.
    """

    queries = _finite_matrix(query_embeddings, "query_embeddings")
    documents = _finite_matrix(document_embeddings, "document_embeddings")
    if queries.shape[1] != documents.shape[1]:
        raise ValueError("query and document embedding dimensions must match")
    positives = _boolean_mask(
        positive_mask,
        shape=(queries.shape[0], documents.shape[0]),
        label="positive_mask",
    )
    if np.any(np.sum(positives, axis=1) == 0):
        raise ValueError("every query must have at least one positive document")
    temperature_value = _positive_scalar(temperature, "temperature")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scores = np.asarray(queries @ documents.T, dtype=np.float64)
        logits = np.asarray(scores / temperature_value, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("embedding dot products must remain finite")
    if not np.all(np.isfinite(logits)):
        raise ValueError("temperature-scaled retrieval logits must remain finite")
    probabilities = _row_softmax(logits)

    positive_logits = np.where(positives, logits, -np.inf)
    positive_row_max = np.max(positive_logits, axis=1, keepdims=True)
    positive_unnormalized = np.where(
        positives,
        np.exp(positive_logits - positive_row_max),
        0.0,
    )
    positive_denominator = np.sum(positive_unnormalized, axis=1, keepdims=True)
    positive_probabilities = positive_unnormalized / positive_denominator

    all_row_max = np.max(logits, axis=1)
    all_log_normalizer = all_row_max + np.log(
        np.sum(np.exp(logits - all_row_max[:, None]), axis=1)
    )
    positive_log_normalizer = positive_row_max[:, 0] + np.log(
        positive_denominator[:, 0]
    )
    per_query_losses = np.asarray(
        all_log_normalizer - positive_log_normalizer,
        dtype=np.float64,
    )

    query_count = queries.shape[0]
    logit_gradients = np.asarray(
        (probabilities - positive_probabilities) / query_count,
        dtype=np.float64,
    )
    query_gradients = np.asarray(
        logit_gradients @ documents / temperature_value,
        dtype=np.float64,
    )
    document_gradients = np.asarray(
        logit_gradients.T @ queries / temperature_value,
        dtype=np.float64,
    )
    return ContrastiveRetrievalReport(
        scores=scores,
        logits=logits,
        probabilities=probabilities,
        positive_conditional_probabilities=np.asarray(
            positive_probabilities, dtype=np.float64
        ),
        positive_mask=positives,
        per_query_losses=per_query_losses,
        mean_loss=float(np.mean(per_query_losses)),
        logit_gradients=logit_gradients,
        query_gradients=query_gradients,
        document_gradients=document_gradients,
        temperature=temperature_value,
    )


def single_positive_info_nce(
    query_embeddings: ArrayLike,
    document_embeddings: ArrayLike,
    positive_document_indices: ArrayLike,
    *,
    temperature: float = 1.0,
) -> ContrastiveRetrievalReport:
    """Build a one-positive mask and evaluate the contrastive objective."""

    queries = _finite_matrix(query_embeddings, "query_embeddings")
    documents = _finite_matrix(document_embeddings, "document_embeddings")
    raw_indices = np.asarray(positive_document_indices)
    if raw_indices.dtype.kind not in "iu":
        raise ValueError("positive_document_indices must contain integers")
    indices = np.asarray(raw_indices, dtype=np.int64)
    if indices.shape != (queries.shape[0],):
        raise ValueError(
            "positive_document_indices must contain exactly one index per query"
        )
    if np.any(indices < 0) or np.any(indices >= documents.shape[0]):
        raise ValueError("positive_document_indices contains an out-of-range index")
    positives = np.zeros((queries.shape[0], documents.shape[0]), dtype=np.bool_)
    positives[np.arange(queries.shape[0]), indices] = True
    return contrastive_retrieval_loss(
        queries,
        documents,
        positives,
        temperature=temperature,
    )


def late_interaction_scores(
    query_token_embeddings: ArrayLike,
    document_token_embeddings: ArrayLike,
    *,
    query_mask: ArrayLike | None = None,
    document_mask: ArrayLike | None = None,
) -> NDArray[np.float64]:
    r"""Compute ColBERT-style MaxSim scores for one query and many documents.

    The score is ``sum_i max_j dot(q_i, d_j)`` over unmasked query and document
    tokens. This isolates late interaction only; it does not run a Transformer,
    train a ColBERT model, compress token vectors, or build an ANN index.
    """

    query_tokens = _finite_matrix(query_token_embeddings, "query_token_embeddings")
    document_tokens = _finite_tensor(
        document_token_embeddings, "document_token_embeddings"
    )
    if query_tokens.shape[1] != document_tokens.shape[2]:
        raise ValueError("query and document token embedding dimensions must match")
    query_visible = (
        np.ones(query_tokens.shape[0], dtype=np.bool_)
        if query_mask is None
        else _boolean_mask(
            query_mask,
            shape=(query_tokens.shape[0],),
            label="query_mask",
        )
    )
    document_visible = (
        np.ones(document_tokens.shape[:2], dtype=np.bool_)
        if document_mask is None
        else _boolean_mask(
            document_mask,
            shape=document_tokens.shape[:2],
            label="document_mask",
        )
    )
    if not np.any(query_visible):
        raise ValueError("query_mask must expose at least one token")
    if np.any(np.sum(document_visible, axis=1) == 0):
        raise ValueError("document_mask must expose at least one token per document")

    with np.errstate(over="ignore", invalid="ignore"):
        token_scores = np.einsum(
            "qf,ndf->nqd",
            query_tokens,
            document_tokens,
            optimize=True,
        )
    if not np.all(np.isfinite(token_scores)):
        raise ValueError("late-interaction token dot products must remain finite")
    token_scores = np.where(document_visible[:, None, :], token_scores, -np.inf)
    max_scores = np.max(token_scores, axis=2)
    return np.asarray(
        np.sum(max_scores[:, query_visible], axis=1),
        dtype=np.float64,
    )


def splade_max_pool(
    token_logits: ArrayLike,
    attention_mask: ArrayLike,
) -> NDArray[np.float64]:
    r"""Apply the SPLADE ``max(log(1 + ReLU(logit)))`` vocabulary pooling rule.

    Inputs have shape ``[items, tokens, vocabulary]``. The result is a
    non-negative sparse-capable lexical vector before sparsity/FLOPS
    regularization. This function does not run an MLM encoder or prove that the
    authored logits produce an effective learned sparse retriever.
    """

    logits = _finite_tensor(token_logits, "token_logits")
    visible = _boolean_mask(
        attention_mask,
        shape=logits.shape[:2],
        label="attention_mask",
    )
    if np.any(np.sum(visible, axis=1) == 0):
        raise ValueError("attention_mask must expose at least one token per item")
    activations = np.log1p(np.maximum(logits, 0.0))
    masked = np.where(visible[:, :, None], activations, 0.0)
    return np.asarray(np.max(masked, axis=1), dtype=np.float64)
