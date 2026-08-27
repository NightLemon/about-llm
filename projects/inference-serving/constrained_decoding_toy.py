"""用有限字符串集合演示 constrained decoding 如何逐 token 屏蔽非法路径。

允许的最终文本只有 ``{"x":1}`` 和 ``{"x":2}``。概率最高的局部 token ``1]`` 会使前缀
无法完成合法字符串，因此约束状态机在采样前将其屏蔽，并只在接受态允许 EOS。
"""

from __future__ import annotations

import json

from about_llm.inference import (
    LiteralSetConstraint,
    constrained_greedy_from_probabilities,
)


def main() -> None:
    """执行受约束贪心解码，并输出关键第三步的允许 token 集合。"""

    # literal set 会被编译成前缀状态机，每一步都知道哪些字符串仍可能完成。
    constraint = LiteralSetConstraint.from_literals(('{"x":1}', '{"x":2}'))
    token_texts = ('{"x"', ":", "1}", "1]", "2}", None, "garbage")
    # 第三步故意让非法的 "1]" 概率最高，验证 mask 发生在最终选择之前。
    probability_table = {
        (): [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        (0,): [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
        (0, 1): [0.0, 0.0, 0.25, 0.65, 0.10, 0.0, 0.0],
        (0, 1, 2): [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
    result = constrained_greedy_from_probabilities(
        probability_table,
        token_texts=token_texts,
        eos_token_id=5,
        constraint=constraint,
        max_new_tokens=4,
    )
    artifact = {
        "constraint": constraint.to_dict(),
        "token_texts": list(token_texts),
        "result": result.to_dict(),
        "critical_step": result.steps[2].to_dict(),
        "scope": {
            "complete_multi_character_token_transition_checked": True,
            "allowed_probability_mass_renormalized": True,
            "eos_requires_accepting_state": True,
            "finite_authored_literal_set_only": True,
            "tokenizer_byte_state_or_normalization_executed": False,
            "json_schema_cfg_or_provider_runtime_equivalence_proved": False,
            "model_kv_gpu_quality_or_performance_executed": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
