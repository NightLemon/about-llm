"""Run a deterministic top-k MoE capacity and sparse-dispatch fixture."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from about_llm.from_scratch import route_topk_capacity, routed_linear_expert_forward


def run_experiment() -> dict[str, Any]:
    logits = np.array(
        [
            [4.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 3.0, 0.0],
            [0.0, 4.0, 3.0],
        ]
    )
    routing = route_topk_capacity(logits, top_k=2, capacity_factor=0.75)
    hidden = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, -1.0], [3.0, 1.0]])
    expert_weights = np.stack(
        [np.eye(2), 2 * np.eye(2), np.array([[1.0, 1.0], [-1.0, 1.0]])]
    )
    output = routed_linear_expert_forward(hidden, expert_weights, routing)
    return {
        "schema_version": 1,
        "configuration": {
            "top_k": 2,
            "capacity_factor": 0.75,
            "capacity_formula": "ceil(capacity_factor * active_tokens * top_k / experts)",
            "capacity_priority": "router probability desc, token index asc, rank asc",
            "top_k_tie_break": "expert id asc",
            "renormalize_after_capacity": True,
        },
        "routing": routing.to_dict(),
        "linear_expert_output": output.tolist(),
        "scope": {
            "actual_topk_routing_and_capacity_drop_executed": True,
            "actual_linear_expert_dispatch_and_combine_executed": True,
            "trained_expert_mlp_or_router_used": False,
            "expert_parallel_all_to_all_or_gpu_kernel_executed": False,
            "deepseek_qwen_or_other_checkpoint_reproduced": False,
            "quality_throughput_or_memory_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

