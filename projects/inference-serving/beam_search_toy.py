from __future__ import annotations

import json

from about_llm.inference import beam_search_from_probabilities


def main() -> None:
    pruning_table = {
        (): [0.6, 0.4, 0.0],
        (0,): [0.49, 0.0, 0.51],
        (1,): [0.0, 0.0, 1.0],
    }
    length_table = {
        (): [0.6, 0.4, 0.0, 0.0],
        (0,): [0.0, 0.0, 0.0, 1.0],
        (1,): [0.0, 0.0, 1.0, 0.0],
        (1, 2): [0.0, 0.0, 0.0, 1.0],
    }
    artifact = {
        "pruning_counterexample": {
            "token_legend": {"0": "A", "1": "B", "2": "EOS"},
            "beam_1": beam_search_from_probabilities(
                pruning_table,
                vocabulary_size=3,
                eos_token_id=2,
                beam_width=1,
                max_new_tokens=2,
            ).to_dict(),
            "beam_2": beam_search_from_probabilities(
                pruning_table,
                vocabulary_size=3,
                eos_token_id=2,
                beam_width=2,
                max_new_tokens=2,
            ).to_dict(),
        },
        "length_penalty_counterexample": {
            "token_legend": {"0": "A", "1": "B", "2": "C", "3": "EOS"},
            "alpha_0": beam_search_from_probabilities(
                length_table,
                vocabulary_size=4,
                eos_token_id=3,
                beam_width=2,
                max_new_tokens=3,
                length_penalty=0,
            ).to_dict(),
            "alpha_2": beam_search_from_probabilities(
                length_table,
                vocabulary_size=4,
                eos_token_id=3,
                beam_width=2,
                max_new_tokens=3,
                length_penalty=2,
            ).to_dict(),
        },
        "scope": {
            "beam_pruning_eos_and_length_finalization_executed": True,
            "global_sequence_optimality_proved": False,
            "length_penalty_includes_eos_and_excludes_prompt": True,
            "model_tokenizer_kv_or_gpu_executed": False,
            "runtime_or_provider_equivalence_claimed": False,
            "text_quality_or_performance_proved": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
