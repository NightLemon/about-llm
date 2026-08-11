from __future__ import annotations

import json

from about_llm.evaluation import (
    clustered_paired_randomization_test,
    paired_randomization_test,
)


def main() -> None:
    baseline = [0, 0, 0, 0, 0, 0]
    candidate = [1, 1, 1, 1, 1, -1]
    clusters = ["user-a"] * 5 + ["user-b"]
    artifact = {
        "authored_case_scores": {
            "baseline": baseline,
            "candidate": candidate,
            "cluster_ids": clusters,
        },
        "naive_case_sign_flip": paired_randomization_test(
            baseline,
            candidate,
            alternative="greater",
        ).to_dict(),
        "cluster_joint_case_weighted": clustered_paired_randomization_test(
            baseline,
            candidate,
            clusters,
            cluster_weighting="case",
            alternative="greater",
        ).to_dict(),
        "cluster_joint_equal_weighted": clustered_paired_randomization_test(
            baseline,
            candidate,
            clusters,
            cluster_weighting="equal",
            alternative="two-sided",
        ).to_dict(),
        "scope": {
            "causal_or_general_model_improvement_proved": False,
            "cluster_joint_sign_flip_executed": True,
            "cluster_level_exchangeability_or_independence_established": False,
            "estimand_or_cluster_definition_selected_without_outcome_looking": False,
            "within_cluster_case_independence_required": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
