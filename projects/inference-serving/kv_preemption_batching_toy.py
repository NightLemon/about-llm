"""Run the deterministic KV-capacity/recompute-preemption scheduling fixture."""

from __future__ import annotations

import json

from about_llm.inference import BatchingRequest, simulate_kv_preemption_batching


def main() -> None:
    report = simulate_kv_preemption_batching(
        [
            BatchingRequest("a", 0, 4, 3),
            BatchingRequest("b", 1, 2, 2),
        ],
        total_blocks=3,
        block_size_tokens=2,
        max_batch_tokens=4,
        max_running_sequences=2,
        max_prefill_tokens_per_request=3,
    )
    artifact = {
        **report.to_dict(),
        "scope": {
            "metadata_only_paged_kv_and_scheduler_integrated": True,
            "recompute_preemption_and_rebuild_executed": True,
            "logical_and_executed_forward_work_separated": True,
            "real_kv_tensor_values_or_gpu_kernel_executed": False,
            "swap_prefix_cache_or_distributed_scheduler_modeled": False,
            "vllm_scheduler_equivalence_proved": False,
            "wall_clock_latency_throughput_vram_or_quality_proved": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
