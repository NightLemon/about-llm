"""在同一组配对分数上比较 bootstrap 与 sign-flip 随机化检验。

每个位置都是同一 case 的 baseline/candidate 分数。实验保留配对关系，分别估计均值差区间、
单侧与双侧精确 p-value，并演示非零差值太多时如何切换到固定种子的 Monte Carlo。
"""

from __future__ import annotations

import json
from dataclasses import asdict

from about_llm.evaluation import paired_bootstrap, paired_randomization_test


def main() -> None:
    """运行配对 bootstrap、精确随机化和 Monte Carlo 三类计算。"""

    # candidate 改善四条、持平一条；平局不会进入 sign-flip 枚举。
    baseline = [0, 0, 0, 0, 1]
    candidate = [1, 1, 1, 1, 1]
    bootstrap_samples = 10_000
    bootstrap_seed = 7
    # bootstrap 重采样的是 case 对，而不是拆开重采样两组分数。
    bootstrap = paired_bootstrap(
        baseline,
        candidate,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    # 精确检验枚举所有非零差值的正负号组合，构造零假设分布。
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
    # 把精确阈值降到 2，故意触发随机抽样近似路径。
    monte_carlo = paired_randomization_test(
        baseline,
        candidate,
        alternative="greater",
        exact_max_nonzero_pairs=2,
        monte_carlo_samples=10_000,
        seed=7,
    )
    artifact = {
        "experiment": {
            "paired_unit": "the baseline and candidate values at one list position",
            "bootstrap_purpose": "estimate uncertainty in the paired mean difference",
            "exact_sign_flip_purpose": "enumerate the paired sharp-null sign distribution",
            "monte_carlo_purpose": (
                "approximate that sign distribution when exact enumeration is disabled"
            ),
            "nonzero_pairs": 4,
            "tied_pairs_removed_from_sign_flip": 1,
        },
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
        "conclusion": (
            "bootstrap intervals and sign-flip p-values answer different questions; "
            "the seeded Monte Carlo result approximates the exact sign-flip calculation"
        ),
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
