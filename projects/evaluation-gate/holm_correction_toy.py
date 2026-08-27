"""用四个 p-value 手算 Holm-Bonferroni 多重比较校正。

算法从最小 p-value 开始逐步收紧阈值，并用 running maximum 保证调整后的 p-value 单调。
它控制预先定义假设族的 FWER，但不会判断效应是否有业务意义。
"""

from __future__ import annotations

import json

from about_llm.evaluation import holm_bonferroni_correction


def main() -> None:
    """校正四个预先给定的 p-value，并输出排序与逐步决策。"""

    # 输入顺序代表四个假设的原始身份，算法内部排序后仍需映射回该顺序。
    p_values = [0.04, 0.01, 0.03, 0.20]
    result = holm_bonferroni_correction(p_values, alpha=0.05)
    artifact = {
        "input_order": {
            "interpretation": "four prespecified hypotheses in authored input order",
            "p_values": p_values,
        },
        "holm": result.to_dict(),
        "scope": {
            "arbitrary_dependence_fwer_control_requires_valid_input_p_values": True,
            "effect_size_or_practical_importance_estimated": False,
            "family_prespecified_or_selection_bias_repaired": False,
            "holm_rank_and_running_maximum_executed": True,
            "repeated_peeking_or_optional_stopping_repaired": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
