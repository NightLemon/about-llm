"""模拟 KV block 不足时 continuous batching 的抢占与重新 prefill。

两个请求竞争三个 block。调度器无法同时保留全部 KV 时，会释放被抢占序列的 block，随后从
prompt 重新计算其状态。报告区分逻辑 token 与实际重复执行的 forward 工作。
"""

from __future__ import annotations

import json

from about_llm.inference import BatchingRequest, simulate_kv_preemption_batching


def main() -> None:
    """运行固定请求时间线并打印 block、抢占和重算事件。"""

    # block_size=2 且总 block=3，故意让并发序列在增长时触发容量压力。
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
