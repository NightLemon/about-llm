"""用小向量串起对比学习、late interaction 和稀疏检索训练目标。

实验比较 easy/hard negative、false negative、multi-positive InfoNCE 和 temperature，随后手算
ColBERT 风格 MaxSim 与 SPLADE max pooling。所有输入都可手算，不训练真实 encoder。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from about_llm.rag import (
    ContrastiveRetrievalReport,
    contrastive_retrieval_loss,
    late_interaction_scores,
    single_positive_info_nce,
    splade_max_pool,
)


def _summary(report: ContrastiveRetrievalReport) -> dict[str, Any]:
    """抽取分数、概率、正例 mask、逐 query loss 与 logit 梯度。"""

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
    """运行 dense 对比损失、late interaction 和 sparse pooling 三组实验。"""

    # query 与 positive 完全同向；easy negative 反向，hard negative 几乎同向。
    query = [[1.0, 0.0]]
    positive = [1.0, 0.0]
    easy_negative = [-1.0, 0.0]
    hard_negative = [0.9, 0.1]
    duplicate_relevant = [1.0, 0.0]
    unrelated = [0.0, 1.0]

    # 先比较 negative 难度，再降低 temperature 观察概率分布变尖。
    easy = single_positive_info_nce(query, [positive, easy_negative], [0])
    hard = single_positive_info_nce(query, [positive, hard_negative], [0])
    hard_cold = single_positive_info_nce(
        query,
        [positive, hard_negative],
        [0],
        temperature=0.25,
    )
    # duplicate_relevant 若被误标为负例，会与真正 positive 竞争，形成 false negative。
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

    late_query_tokens = [[1.0, 0.0], [0.0, 1.0]]
    late_documents = [
        [[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]],
        [[0.8, 0.8], [0.0, 0.0], [0.0, 0.0]],
    ]
    late_document_mask = [[True, True, False], [True, False, False]]
    maxsim = late_interaction_scores(
        late_query_tokens,
        late_documents,
        document_mask=late_document_mask,
    )
    expected_maxsim = np.array([2.0, 1.6])
    sparse_token_activations = [
        [[1.0, -2.0, 0.5], [100.0, 100.0, 100.0]],
        [[0.0, 2.0, -1.0], [3.0, 1.0, 4.0]],
    ]
    sparse_token_mask = [[True, False], [True, True]]
    sparse_vectors = splade_max_pool(
        sparse_token_activations,
        sparse_token_mask,
    )
    expected_sparse_vectors = np.array(
        [
            [np.log(2.0), 0.0, np.log(1.5)],
            [np.log(4.0), np.log(3.0), np.log(5.0)],
        ]
    )

    return {
        "fixture": {
            "dense": {
                "query_vectors": query,
                "positive": positive,
                "easy_negative": easy_negative,
                "hard_negative": hard_negative,
                "duplicate_relevant": duplicate_relevant,
                "unrelated": unrelated,
            },
            "late_interaction": {
                "query_token_vectors": late_query_tokens,
                "document_token_vectors": late_documents,
                "document_mask": late_document_mask,
            },
            "learned_sparse": {
                "token_activations": sparse_token_activations,
                "token_mask": sparse_token_mask,
            },
        },
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
        "observations": {
            "hard_negative_has_higher_loss_than_easy_negative": (hard.mean_loss > easy.mean_loss),
            "multi_positive_labels_reduce_false_negative_loss": (
                multi_positive.mean_loss < false_negative.mean_loss
            ),
            "masked_large_padding_value_is_excluded_from_maxsim": bool(
                np.allclose(maxsim, expected_maxsim)
            ),
            "masked_large_padding_value_is_excluded_from_sparse_pooling": bool(
                np.allclose(sparse_vectors, expected_sparse_vectors)
            ),
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
