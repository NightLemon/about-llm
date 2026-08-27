"""用精确反例说明 best-of-N 会放大 verifier 的可利用漏洞。

三个候选中，verifier_hack 实际错误却拿到最高 verifier 分。N 从 1 增至 16 时，选中样本的
平均 verifier 分继续上升，但真实成功率先升后降；oracle 上限则仍随覆盖率增加。
"""

from __future__ import annotations

import json

from about_llm.inference import (
    VerifierCandidate,
    analyze_verifier_guided_best_of_n,
)

# sampling_weight 表示模型采到候选的概率权重，target_success 是独立于 verifier 的真值。
AUTHORED_CANDIDATES = (
    VerifierCandidate(
        candidate_id="wrong",
        sampling_weight=5,
        verifier_score=20,
        target_success=False,
    ),
    VerifierCandidate(
        candidate_id="correct",
        sampling_weight=4,
        verifier_score=80,
        target_success=True,
    ),
    VerifierCandidate(
        candidate_id="verifier_hack",
        sampling_weight=1,
        verifier_score=99,
        target_success=False,
    ),
)


def run_toy() -> dict[str, object]:
    """精确计算 N=1/4/16 时 verifier 选择与 oracle 选择的成功率。"""

    analyses = tuple(
        analyze_verifier_guided_best_of_n(
            AUTHORED_CANDIDATES,
            sample_count=sample_count,
        )
        for sample_count in (1, 4, 16)
    )
    # 三条序列分别回答“分数是否更高”“实际是否更对”“候选集合是否覆盖正确答案”。
    expected_scores = [analysis.expected_selected_verifier_score for analysis in analyses]
    selected_success = [analysis.selected_success_probability for analysis in analyses]
    oracle_success = [analysis.oracle_success_probability for analysis in analyses]
    return {
        "implementation": "about-llm.verifier-best-of-n-toy.v1",
        "analyses": [analysis.to_dict() for analysis in analyses],
        "observations": {
            "expected_verifier_score_strictly_increases": (
                expected_scores[0] < expected_scores[1] < expected_scores[2]
            ),
            "selected_success_n4_above_n1": selected_success[1]
            > selected_success[0],
            "selected_success_n16_below_n1": selected_success[2]
            < selected_success[0],
            "oracle_success_strictly_increases": (
                oracle_success[0] < oracle_success[1] < oracle_success[2]
            ),
        },
        "scope": {
            "authored_finite_candidate_distribution": True,
            "iid_fixed_distribution_assumed": True,
            "closed_form_exact_fraction_analysis_executed": True,
            "candidate_sequence_enumeration_executed": False,
            "oracle_target_labels_authored": True,
            "model_tokenizer_or_prm_executed": False,
            "verifier_calibration_or_semantic_correctness_proved": False,
            "latency_cost_parallelism_or_target_quality_measured": False,
            "target_model_provider_or_gpu_behavior_proved": False,
        },
    }


def main() -> None:
    """打印不同 N 下的 verifier 分数与真实成功率。"""

    print(json.dumps(run_toy(), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
