"""Exact authored best-of-N verifier-selection counterexample."""

from __future__ import annotations

import json

from about_llm.inference import (
    VerifierCandidate,
    analyze_verifier_guided_best_of_n,
)

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
    analyses = tuple(
        analyze_verifier_guided_best_of_n(
            AUTHORED_CANDIDATES,
            sample_count=sample_count,
        )
        for sample_count in (1, 4, 16)
    )
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
    print(json.dumps(run_toy(), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
