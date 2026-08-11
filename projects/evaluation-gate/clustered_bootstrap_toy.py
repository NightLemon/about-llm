from __future__ import annotations

import json

from about_llm.evaluation import clustered_paired_bootstrap


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
        "case_weighted": clustered_paired_bootstrap(
            baseline,
            candidate,
            clusters,
            cluster_weighting="case",
        ).to_dict(),
        "equal_cluster": clustered_paired_bootstrap(
            baseline,
            candidate,
            clusters,
            cluster_weighting="equal",
        ).to_dict(),
        "scope": {
            "bca_or_small_cluster_coverage_guarantee": False,
            "case_and_equal_weighting_treated_as_same_estimand": False,
            "causal_or_general_model_improvement_proved": False,
            "ordered_cluster_resamples_enumerated": True,
            "representative_independent_clusters_established": False,
            "within_cluster_case_independence_required": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
