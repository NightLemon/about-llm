"""Compare one paired score vector with bootstrap and sign-flip methods."""

from __future__ import annotations

import json
from dataclasses import asdict

from about_llm.evaluation import paired_bootstrap, paired_randomization_test


def main() -> None:
    baseline = [0, 0, 0, 0, 1]
    candidate = [1, 1, 1, 1, 1]
    bootstrap_samples = 10_000
    bootstrap_seed = 7
    bootstrap = paired_bootstrap(
        baseline,
        candidate,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    exact_greater = paired_randomization_test(
        baseline,
        candidate,
        alternative="greater",
    )
    exact_two_sided = paired_randomization_test(
        baseline,
        candidate,
        alternative="two-sided",
    )
    monte_carlo = paired_randomization_test(
        baseline,
        candidate,
        alternative="greater",
        exact_max_nonzero_pairs=2,
        monte_carlo_samples=10_000,
        seed=7,
    )
    artifact = {
        "authored_case_scores": {
            "baseline": baseline,
            "candidate": candidate,
        },
        "seeded_paired_bootstrap": {
            "confidence": 0.95,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            **asdict(bootstrap),
        },
        "exact_greater": exact_greater.to_dict(),
        "exact_two_sided": exact_two_sided.to_dict(),
        "seeded_monte_carlo_greater": monte_carlo.to_dict(),
        "scope": {
            "paired_case_sign_flip_distribution_executed": True,
            "paired_case_bootstrap_executed": True,
            "zero_difference_removed_from_sign_enumeration": True,
            "case_population_representativeness_established": False,
            "exchangeability_or_random_assignment_established": False,
            "cluster_dependence_modeled": False,
            "multiple_comparison_correction_applied": False,
            "causal_product_or_model_improvement_proved": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
