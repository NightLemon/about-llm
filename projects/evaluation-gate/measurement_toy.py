"""用反例区分标注一致性与有效性，并演示精确 sign test 功效规划。

两位标注者彼此完全一致，却与给定 criterion 全部相反，说明高 reliability 不保证 validity。
后半段用二项分布精确计算给定样本量的功效、所需样本量和最小可检测效应。
"""

from __future__ import annotations

import json
from fractions import Fraction

from about_llm.evaluation import (
    cohen_kappa,
    criterion_validity,
    minimum_detectable_sign_effect,
    minimum_sign_test_sample_size,
    one_sided_sign_test_power,
)


def run_toy() -> dict[str, object]:
    """计算一致性反例和三种配对 sign test 规划问题。"""

    # 两位 rater 标签相同，所以 Cohen's kappa 很高；但它们都与 criterion 相反。
    criterion = ("correct", "correct", "incorrect", "incorrect")
    rater_a = ("incorrect", "incorrect", "correct", "correct")
    rater_b = rater_a
    reliability = cohen_kappa(rater_a, rater_b)
    validity_a = criterion_validity(rater_a, criterion)
    validity_b = criterion_validity(rater_b, criterion)

    # 问题一：只有五个非平局 pair，真实胜率若为 0.8，检验功效是多少？
    fixed_power = one_sided_sign_test_power(
        informative_pair_count=5,
        alternative_positive_probability=Fraction(4, 5),
    )
    # 问题二：真实胜率假设为 0.75，要达到 0.8 功效至少需要多少非平局 pair？
    sample_size = minimum_sign_test_sample_size(
        alternative_positive_probability=Fraction(3, 4),
        target_power=Fraction(4, 5),
    )
    # 问题三：固定 25 个 pair 和 0.8 功效，最小能检测多大的胜率偏离？
    detectable_effect = minimum_detectable_sign_effect(
        informative_pair_count=25,
        target_power=Fraction(4, 5),
        probability_grid_denominator=1_000,
    )
    return {
        "implementation": "about-llm.evaluation-measurement-toy.v1",
        "reliability_is_not_validity_counterexample": {
            "authored_criterion": list(criterion),
            "authored_rater_a": list(rater_a),
            "authored_rater_b": list(rater_b),
            "inter_rater_reliability": reliability.to_dict(),
            "rater_a_criterion_validity": validity_a.to_dict(),
            "rater_b_criterion_validity": validity_b.to_dict(),
            "observation": (
                "the raters agree perfectly with each other while both disagree "
                "with every supplied criterion label"
            ),
        },
        "paired_sign_test_planning": {
            "fixed_five_informative_pairs": fixed_power.to_dict(),
            "minimum_informative_pairs_for_declared_effect": sample_size.to_dict(),
            "minimum_detectable_effect_on_declared_grid": detectable_effect.to_dict(),
        },
        "scope": {
            "exact_fraction_agreement_confusion_and_binomial_power_executed": True,
            "labels_and_criterion_are_authored_not_human_annotations": True,
            "construct_content_and_criterion_quality_established": False,
            "criterion_independence_or_correctness_established": False,
            "case_sampling_or_population_representativeness_established": False,
            "total_case_count_from_informative_pair_count_inferred": False,
            "cluster_dependence_or_repeated_peeking_supported": False,
            "model_judge_provider_or_online_experiment_executed": False,
        },
    }


def main() -> None:
    """输出 reliability、validity 与精确功效规划结果。"""

    print(json.dumps(run_toy(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
