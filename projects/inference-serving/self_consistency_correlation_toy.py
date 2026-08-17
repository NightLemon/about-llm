"""Exact binary self-consistency counterexample with correlated candidates."""

from __future__ import annotations

import json
from itertools import pairwise

from about_llm.inference import (
    BinaryVoteRegime,
    analyze_latent_regime_binary_majority,
)

INDEPENDENT_REGIMES = (
    BinaryVoteRegime(
        regime_id="iid",
        regime_weight=1,
        success_weight=3,
        failure_weight=2,
    ),
)

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
    print(json.dumps(run_toy(), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
