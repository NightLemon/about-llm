"""精确计算重复偷看固定样本量检验带来的累计假阳性率。

在同一实验的 10/20/30/40/50 个 pair 处反复用 alpha=0.05 判断，会让“至少误报一次”的
概率超过 5%。实验再用预先分配的 Bonferroni 阈值对照，展示停止规则也是评测设计的一部分。
"""

from __future__ import annotations

import json
from fractions import Fraction

from about_llm.evaluation import analyze_repeated_two_sided_sign_tests


def run_toy() -> dict[str, object]:
    """对相同 look schedule 精确比较朴素阈值与 Bonferroni 阈值。"""

    # Fraction 保留精确有理数，避免边界 p-value 因浮点舍入改变是否拒绝。
    look_sample_counts = (10, 20, 30, 40, 50)
    familywise_alpha = Fraction(1, 20)
    # naive 在每次查看都花完整的 0.05，累计错误率会膨胀。
    naive = analyze_repeated_two_sided_sign_tests(
        look_sample_counts,
        per_look_alpha=familywise_alpha,
    )
    # Bonferroni 把总 alpha 平分给五次预先声明的查看。
    bonferroni = analyze_repeated_two_sided_sign_tests(
        look_sample_counts,
        per_look_alpha=familywise_alpha / len(look_sample_counts),
    )
    return {
        "implementation": "about-llm.sequential-peeking-toy.v1",
        "null_contract": {
            "informative_pair_outcomes": [
                "candidate_minus_baseline_positive",
                "candidate_minus_baseline_negative",
            ],
            "ties_modeled": False,
            "signs_iid_bernoulli_half": True,
            "two_sided_p_value": "twice_smaller_inclusive_binomial_tail_capped_at_one",
            "stopping_rule": "stop_at_first_p_lte_per_look_alpha",
        },
        "familywise_alpha": {
            "numerator": familywise_alpha.numerator,
            "denominator": familywise_alpha.denominator,
            "decimal": float(familywise_alpha),
        },
        "scenarios": {
            "naive_alpha_at_every_look": naive.to_dict(),
            "prespecified_bonferroni_alpha_split": bonferroni.to_dict(),
        },
        "observations": {
            "naive_familywise_error_exceeds_five_percent": (
                naive.familywise_null_rejection_probability > familywise_alpha
            ),
            "bonferroni_familywise_error_at_most_five_percent": (
                bonferroni.familywise_null_rejection_probability
                <= familywise_alpha
            ),
            "same_look_schedule": (
                naive.look_sample_counts == bonferroni.look_sample_counts
            ),
        },
        "scope": {
            "exact_fraction_dynamic_program_executed": True,
            "logical_sign_sequence_enumeration_executed": False,
            "look_schedule_and_thresholds_prespecified": True,
            "confidence_sequence_or_always_valid_p_value_implemented": False,
            "effect_size_power_or_sample_size_estimated": False,
            "case_sampling_labels_clusters_or_exchangeability_validated": False,
            "model_judge_provider_or_online_ab_test_executed": False,
        },
    }


def main() -> None:
    """输出两种停止策略的精确累计拒绝概率。"""

    print(
        json.dumps(
            run_toy(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
