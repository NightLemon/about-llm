"""用离散时间模拟 continuous batching 的请求接纳、prefill 和 decode。

三个请求在不同 step 到达，调度器受 batch token 预算、并行序列数和单请求 prefill 上限约束。
输出时间线展示新请求如何插入正在 decode 的 batch，而不是等待整批请求同时结束。
"""

from __future__ import annotations

import json
from typing import Any

from about_llm.inference import BatchingRequest, simulate_continuous_batching


def run_experiment() -> dict[str, Any]:
    """构造三个请求并返回每个调度 step 的 token 工作量。"""

    # request-a 先到且 prompt 超过单次 prefill 上限，因此会被分块处理。
    requests = [
        BatchingRequest("request-a", arrival_step=0, prompt_tokens=4, output_tokens=3),
        BatchingRequest("request-b", arrival_step=1, prompt_tokens=2, output_tokens=2),
        BatchingRequest("request-c", arrival_step=1, prompt_tokens=1, output_tokens=1),
    ]
    # 每步最多处理 4 个 token、运行 2 条序列；模拟器会在预算内重新组成 batch。
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
    """打印调度时间线和每个请求的完成统计。"""

    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
