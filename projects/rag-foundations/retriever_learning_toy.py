"""用小向量串起对比学习、late interaction 和稀疏检索训练目标。

实验比较 easy/hard negative、false negative、multi-positive InfoNCE 和 temperature，随后手算
ColBERT 风格 MaxSim 与 SPLADE max pooling。所有输入都可手算，不训练真实 encoder。
"""

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
