"""用四个 token 演示 Top-k MoE 路由、专家容量和稀疏合并。

每行 router logits 表示一个 token 对三个专家的偏好。实验先选出概率最高的两个专家，
再按容量丢弃超额分配，最后只运行被接收的线性专家并按路由权重合并结果。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from about_llm.from_scratch import route_topk_capacity, routed_linear_expert_forward


def run_experiment() -> dict[str, Any]:
    """执行一次容量受限路由，并返回每个 token 的去向和专家输出。"""

    # 前三个 token 都强烈偏好专家 0，故意制造超过专家容量的竞争。
    logits = np.array(
        [
            [4.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
            [4.0, 3.0, 0.0],
            [0.0, 4.0, 3.0],
        ]
    )
    # top_k=2 表示每个 token 最多选择两个专家；capacity_factor 控制每个专家的槽位数。
    routing = route_topk_capacity(logits, top_k=2, capacity_factor=0.75)

    # hidden 是四个 token 的二维表示，三个矩阵分别充当三个可辨认的线性专家。
    hidden = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, -1.0], [3.0, 1.0]])
    expert_weights = np.stack(
        [np.eye(2), 2 * np.eye(2), np.array([[1.0, 1.0], [-1.0, 1.0]])]
    )
    # 只把已接收的 token 发给对应专家，再将多个专家的结果加权合并到原 token 顺序。
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
    """打印路由、容量丢弃和稀疏专家输出。"""

    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
