"""Run a deterministic continuous-batching scheduling fixture."""

from __future__ import annotations

import json
from typing import Any

from about_llm.inference import BatchingRequest, simulate_continuous_batching


def run_experiment() -> dict[str, Any]:
    requests = [
        BatchingRequest("request-a", arrival_step=0, prompt_tokens=4, output_tokens=3),
        BatchingRequest("request-b", arrival_step=1, prompt_tokens=2, output_tokens=2),
        BatchingRequest("request-c", arrival_step=1, prompt_tokens=1, output_tokens=1),
    ]
    report = simulate_continuous_batching(
        requests,
        max_batch_tokens=4,
        max_running_sequences=2,
        max_prefill_tokens_per_request=3,
    )
    return {
        "schema_version": 1,
        **report.to_dict(),
        "scope": {
            "deterministic_discrete_cpu_policy_simulated": True,
            "prefill_last_position_emits_first_token": True,
            "real_model_or_gpu_kernel_executed": False,
            "vllm_scheduler_equivalence_proved": False,
            "kv_capacity_preemption_or_prefix_cache_modeled": False,
            "wall_clock_latency_throughput_or_slo_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

