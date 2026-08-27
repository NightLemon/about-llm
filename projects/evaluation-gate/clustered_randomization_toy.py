"""比较逐 case 翻符号与整 cluster 翻符号的配对随机化检验。

六条 case 实际只来自两个用户。朴素检验会独立翻转六个差值，cluster 检验则让同一用户的
全部差值共同翻转，从而保留组内依赖结构。实验还对比 case 加权和 cluster 等权结论。
"""

from __future__ import annotations

import json

from about_llm.evaluation import (
    clustered_paired_randomization_test,
    paired_randomization_test,
)


def main() -> None:
    """在同一差值向量上运行朴素与 clustered 三种检验。"""

    # user-a 重复五次改善，user-b 出现一次退化，故意制造 cluster 大小不平衡。
    baseline = [0, 0, 0, 0, 0, 0]
    candidate = [1, 1, 1, 1, 1, -1]
    clusters = ["user-a"] * 5 + ["user-b"]
    # 固定单侧备择，前两项只改变翻转单位，后两项只改变 cluster 权重。
    artifact = {
        "experiment": {
            "paired_case_unit": "one scored case",
            "cluster_unit": "one user",
            "naive_sign_flip": "flip each of six case differences independently",
            "cluster_sign_flip": "flip all differences from the same user together",
            "authored_cluster_sizes": {"user-a": 5, "user-b": 1},
            "alternative_for_all_tests": "greater",
            "comparison_design": {
                "sign_flip_unit": (
                    "compare naive_case_sign_flip with cluster_joint_case_weighted; "
                    "case weighting and the alternative stay fixed"
                ),
                "cluster_weighting": (
                    "compare cluster_joint_case_weighted with "
                    "cluster_joint_equal_weighted; the sign-flip unit and alternative "
                    "stay fixed"
                ),
            },
        },
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
            alternative="greater",
        ).to_dict(),
        "conclusion": (
            "joint cluster flips preserve the authored within-user dependence; changing "
            "from case weighting to equal-cluster weighting also changes the estimand"
        ),
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
