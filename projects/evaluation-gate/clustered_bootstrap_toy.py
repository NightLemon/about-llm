"""比较按 case 加权和按 cluster 等权的配对 bootstrap。

user-a 有五条改善样本，user-b 只有一条退化样本。两种加权回答不同问题：随机抽一条 case
的平均变化，或随机抽一个用户的平均变化。实验展示 cluster 大小不均时两者为何不能混用。
"""

from __future__ import annotations

import json

from about_llm.evaluation import clustered_paired_bootstrap


def main() -> None:
    """对同一组 clustered 差值运行两种 estimand 的完整枚举。"""

    # 前五条来自同一用户，不能误当作五个相互独立的用户证据。
    baseline = [0, 0, 0, 0, 0, 0]
    candidate = [1, 1, 1, 1, 1, -1]
    clusters = ["user-a"] * 5 + ["user-b"]
    # case weighting 让大 cluster 权重更高；equal weighting 让两个用户各占一半。
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
