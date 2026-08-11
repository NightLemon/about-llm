"""Run exact one-step sampling and repetition-penalty fixtures."""

from __future__ import annotations

import json

import numpy as np

from about_llm.inference import SamplingConfig, sample_next_token


def main() -> None:
    step = sample_next_token(
        np.log(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)),
        config=SamplingConfig(temperature=1, top_k=3, top_p=0.7),
        uniform=0.6,
    )
    signed_penalty = sample_next_token(
        [2.0, -2.0, 0.5],
        config=SamplingConfig(repetition_penalty=2),
        prior_token_ids=(0, 1, 1),
        uniform=0,
    )
    artifact = {
        "exact_top_k_then_top_p_fixture": step.to_dict(),
        "signed_repetition_penalty_fixture": {
            "input_logits": signed_penalty.input_logits.tolist(),
            "prior_token_ids": signed_penalty.prior_token_ids,
            "adjusted_logits": signed_penalty.repetition_adjusted_logits.tolist(),
        },
        "scope": {
            "authored_finite_logits_processed": True,
            "fixed_uniform_inverse_cdf_executed": True,
            "processor_order_and_tie_break_fixed": True,
            "model_forward_or_tokenizer_executed": False,
            "multi_token_eos_stop_or_kv_modeled": False,
            "runtime_default_equivalence_proved": False,
            "generation_quality_latency_or_throughput_proved": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
