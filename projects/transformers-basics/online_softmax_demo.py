"""比较分块 online softmax Attention 与完整矩阵实现。

普通 Attention 会一次性构造完整分数矩阵；online softmax 则逐块读取 K/V，并用递推状态
维护当前最大值、归一化分母和加权和。本实验验证两条路径在实数算术上等价（float64 下最大
误差约 1 个 ULP），但不主张 bitwise 相同，也不测 GPU 性能。
"""

from __future__ import annotations

import json

import numpy as np

from about_llm.from_scratch import (
    blockwise_online_attention,
    causal_mask,
    scaled_dot_product_attention,
)


def build_report() -> dict[str, object]:
    """用同一组 Q/K/V 比较分块递推与稠密 NumPy 参考实现。"""

    # 固定随机输入，同时让 query_length 与 key_length 不同，覆盖非方形 Attention。
    rng = np.random.default_rng(47)
    query = rng.normal(size=(5, 4))
    key = rng.normal(size=(7, 4))
    value = rng.normal(size=(7, 3))
    mask = causal_mask(query_length=5, key_length=7)
    # 稠密路径会物化完整分数矩阵，作为独立而直观的参考答案。
    dense_output, _ = scaled_dot_product_attention(query, key, value, mask=mask)

    # 分块路径每次只处理三个 key，跨块递推 softmax 统计量。
    online = blockwise_online_attention(
        query,
        key,
        value,
        block_size=3,
        mask=mask,
    )
    return {
        "implementation": "about-llm.online-softmax-oracle.v1",
        "fixture": {
            "query_length": 5,
            "key_length": 7,
            "head_dim": 4,
            "value_dim": 3,
            "key_block_size": 3,
        },
        "observations": {
            "key_block_count": online.key_block_count,
            "logical_peak_score_elements": online.logical_peak_score_elements,
            "full_score_elements": online.full_score_elements,
            # 最大绝对误差直接回答两种计算顺序是否得到足够接近的输出。
            "max_abs_error_vs_dense": float(
                np.max(np.abs(online.output - dense_output))
            ),
            "all_outputs_finite": bool(np.all(np.isfinite(online.output))),
        },
        "scope": {
            "online_path_materialized_complete_score_or_probability": False,
            "dense_reference_materialized_for_comparison": True,
            "float64_online_accumulation": True,
            "real_arithmetic_equivalence_claimed": True,
            "bitwise_equivalence_claimed": False,
            "cuda_or_gpu_kernel_executed": False,
            "flashattention_backend_executed": False,
            "hbm_traffic_peak_memory_or_performance_measured": False,
        },
    }


def main() -> None:
    """输出形状、逻辑中间量规模和两条路径的最大误差。"""

    print(json.dumps(build_report(), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
