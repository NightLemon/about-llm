from __future__ import annotations

import json

from about_llm.inference import (
    LiteralSetConstraint,
    constrained_greedy_from_probabilities,
)


def main() -> None:
    constraint = LiteralSetConstraint.from_literals(('{"x":1}', '{"x":2}'))
    token_texts = ('{"x"', ":", "1}", "1]", "2}", None, "garbage")
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
