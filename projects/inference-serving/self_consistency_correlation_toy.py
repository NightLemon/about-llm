"""用精确概率反例说明相关候选可能让 self-consistency 越采样越差。

两种场景的单次成功率相同。独立场景每票 IID，增加奇数票会改善多数决；相关场景先为整道题
抽一个 easy/hard 隐状态，同题候选因共享难度而相关，多采样反而更确信错误的 hard 答案。
"""

from __future__ import annotations

import json
from itertools import pairwise

from about_llm.inference import (
    BinaryVoteRegime,
    analyze_latent_regime_binary_majority,
)

# IID 场景只有一个 regime，每次候选成功概率恒为 3/5。
INDEPENDENT_REGIMES = (
    BinaryVoteRegime(
        regime_id="iid",
        regime_weight=1,
        success_weight=3,
        failure_weight=2,
    ),
)

# 相关场景先等概率选择 easy 或 hard，同一题的全部候选共享该 regime。
LATENT_CORRELATED_REGIMES = (
    BinaryVoteRegime(
        regime_id="easy",
        regime_weight=1,
        success_weight=9,
        failure_weight=1,
    ),
    BinaryVoteRegime(
        regime_id="hard",
        regime_weight=1,
        success_weight=3,
        failure_weight=7,
    ),
)


def run_toy() -> dict[str, object]:
    """精确计算 1/3/5/11 个候选的多数决成功率与票间相关性。"""

    sample_counts = (1, 3, 5, 11)
    independent = tuple(
        analyze_latent_regime_binary_majority(
            INDEPENDENT_REGIMES,
            sample_count=sample_count,
        )
        for sample_count in sample_counts
    )
    correlated = tuple(
        analyze_latent_regime_binary_majority(
            LATENT_CORRELATED_REGIMES,
            sample_count=sample_count,
        )
        for sample_count in sample_counts
    )
    # 分别抽出成功率序列，后面直接检查随 sample count 的单调方向。
    independent_majorities = [
        analysis.majority_success_probability for analysis in independent
    ]
    correlated_majorities = [
        analysis.majority_success_probability for analysis in correlated
    ]
    return {
        "implementation": "about-llm.self-consistency-correlation-toy.v1",
        "binary_answer_labels": ["target_success", "target_failure"],
        "scenarios": {
            "independent": [analysis.to_dict() for analysis in independent],
            "latent_correlated": [analysis.to_dict() for analysis in correlated],
        },
        "observations": {
            "same_single_sample_success_probability": all(
                analysis.single_sample_success_probability
                == independent[0].single_sample_success_probability
                for analysis in (*independent, *correlated)
            ),
            "independent_majority_strictly_increases": all(
                left < right
                for left, right in pairwise(independent_majorities)
            ),
            "correlated_majority_strictly_decreases": all(
                left > right
                for left, right in pairwise(correlated_majorities)
            ),
            "independent_pairwise_correlation_is_zero": all(
                analysis.pairwise_success_correlation == 0
                for analysis in independent
            ),
            "latent_pairwise_correlation_is_three_eighths": all(
                analysis.pairwise_success_correlation is not None
                and analysis.pairwise_success_correlation.numerator == 3
                and analysis.pairwise_success_correlation.denominator == 8
                for analysis in correlated
            ),
        },
        "scope": {
            "authored_binary_answer_distribution": True,
            "one_latent_regime_drawn_per_question": True,
            "candidate_correctness_conditionally_iid_within_regime": True,
            "exact_fraction_binomial_tail_executed": True,
            "binary_vote_sequence_enumeration_executed": False,
            "multiclass_or_open_text_canonicalization_modeled": False,
            "model_tokenizer_dataset_or_judge_executed": False,
            "latency_cost_provider_or_target_quality_measured": False,
        },
    }


def main() -> None:
    """输出独立与潜变量相关场景的精确分数结果。"""

    print(json.dumps(run_toy(), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
