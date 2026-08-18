"""Run finite controls for contrastive, late-interaction, and sparse retrieval."""

from __future__ import annotations

import json
from typing import Any

from about_llm.rag import (
    ContrastiveRetrievalReport,
    contrastive_retrieval_loss,
    late_interaction_scores,
    single_positive_info_nce,
    splade_max_pool,
)


def _summary(report: ContrastiveRetrievalReport) -> dict[str, Any]:
    return {
        "scores": report.scores.tolist(),
        "probabilities": report.probabilities.tolist(),
        "positive_mask": report.positive_mask.tolist(),
        "per_query_losses": report.per_query_losses.tolist(),
        "mean_loss": report.mean_loss,
        "logit_gradients": report.logit_gradients.tolist(),
        "temperature": report.temperature,
    }


def run_experiment() -> dict[str, Any]:
    query = [[1.0, 0.0]]
    positive = [1.0, 0.0]
    easy_negative = [-1.0, 0.0]
    hard_negative = [0.9, 0.1]
    duplicate_relevant = [1.0, 0.0]
    unrelated = [0.0, 1.0]

    easy = single_positive_info_nce(query, [positive, easy_negative], [0])
    hard = single_positive_info_nce(query, [positive, hard_negative], [0])
    hard_cold = single_positive_info_nce(
        query,
        [positive, hard_negative],
        [0],
        temperature=0.25,
    )
    false_negative = single_positive_info_nce(
        query,
        [positive, duplicate_relevant, unrelated],
        [0],
    )
    multi_positive = contrastive_retrieval_loss(
        query,
        [positive, duplicate_relevant, unrelated],
        [[True, True, False]],
    )

    maxsim = late_interaction_scores(
        [[1.0, 0.0], [0.0, 1.0]],
        [
            [[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]],
            [[0.8, 0.8], [0.0, 0.0], [0.0, 0.0]],
        ],
        document_mask=[[True, True, False], [True, False, False]],
    )
    sparse_vectors = splade_max_pool(
        [
            [[1.0, -2.0, 0.5], [100.0, 100.0, 100.0]],
            [[0.0, 2.0, -1.0], [3.0, 1.0, 4.0]],
        ],
        [[True, False], [True, True]],
    )

    return {
        "contrastive_objective": {
            "easy_negative": _summary(easy),
            "hard_negative": _summary(hard),
            "hard_negative_lower_temperature": _summary(hard_cold),
            "unlabeled_relevant_treated_as_negative": _summary(false_negative),
            "same_pair_with_multi_positive_labels": _summary(multi_positive),
        },
        "late_interaction": {
            "maxsim_scores": maxsim.tolist(),
        },
        "learned_sparse": {
            "splade_max_pooled_vectors": sparse_vectors.tolist(),
        },
        "scope": {
            "device": "CPU",
            "analytic_embedding_gradients_computed": True,
            "transformer_or_retrieval_model_executed": False,
            "encoder_parameters_trained": False,
            "ann_index_executed": False,
            "authored_labels_or_embeddings_representative": False,
            "retrieval_quality_or_production_performance_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
