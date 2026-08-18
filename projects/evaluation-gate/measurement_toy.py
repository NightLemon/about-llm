"""Reliability/validity counterexample and exact sign-test planning toy."""

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
    criterion = ("correct", "correct", "incorrect", "incorrect")
    rater_a = ("incorrect", "incorrect", "correct", "correct")
    rater_b = rater_a
    reliability = cohen_kappa(rater_a, rater_b)
    validity_a = criterion_validity(rater_a, criterion)
    validity_b = criterion_validity(rater_b, criterion)

    fixed_power = one_sided_sign_test_power(
        informative_pair_count=5,
        alternative_positive_probability=Fraction(4, 5),
    )
    sample_size = minimum_sign_test_sample_size(
        alternative_positive_probability=Fraction(3, 4),
        target_power=Fraction(4, 5),
    )
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
    print(json.dumps(run_toy(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
