"""量化一组真实 K/V 张量，并观察 INT8 KV Cache 对 GQA 输出的误差。

每个 (batch, KV head, token) 位置上的 head_dim 向量单独取一个 absmax scale。实验实际生成 INT8 code 与
FP32 scale，反量化后分别比较 K、V、Attention 概率和最终输出，同时统计存储量。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

import numpy as np

from about_llm.from_scratch.attention_numpy import causal_mask, grouped_query_attention
from about_llm.inference import (
    quantization_error,
    quantize_kv_cache_int8,
    quantized_kv_grouped_query_attention,
)


def run_experiment(
    *,
    seed: int,
    batch_size: int,
    query_heads: int,
    key_value_heads: int,
    cached_tokens: int,
    query_tokens: int,
    key_head_dim: int,
    value_head_dim: int,
) -> dict[str, Any]:
    """生成合成 Q/K/V，执行 FP32 与量化 KV 两条 GQA 路径并比较。"""

    # GQA 要求每个 KV head 能被整数个 query head 共享。
    if query_heads % key_value_heads:
        raise ValueError("query_heads must be divisible by key_value_heads")
    if query_tokens > cached_tokens:
        raise ValueError("query_tokens cannot exceed cached_tokens")
    # 固定 seed 生成形状可调的 FP32 张量，便于重复误差实验。
    rng = np.random.default_rng(seed)
    query = rng.normal(
        size=(batch_size, query_heads, query_tokens, key_head_dim)
    ).astype(np.float32)
    key = rng.normal(
        size=(batch_size, key_value_heads, cached_tokens, key_head_dim)
    ).astype(np.float32)
    value = rng.normal(
        size=(batch_size, key_value_heads, cached_tokens, value_head_dim)
    ).astype(np.float32)
    # 放入全零向量，专门覆盖 absmax=0 时 scale 的边界处理。
    key[0, 0, 0] = 0
    value[0, 0, 0] = 0

    # cache 中保留真正的 int8 codes 和每向量 float32 scale，而非只估算大小。
    cache = quantize_kv_cache_int8(key, value)
    restored_key, restored_value = cache.dequantize()
    mask = causal_mask(query_tokens, cached_tokens)
    # FP32 GQA 是参考路径；量化路径会先反量化 KV 再执行同一 Attention 语义。
    baseline_output, baseline_probabilities = grouped_query_attention(
        query,
        key,
        value,
        mask=mask,
    )
    quantized_output, quantized_probabilities = quantized_kv_grouped_query_attention(
        query, cache
    )
    # 因果 mask 后的未来位置概率必须为零，量化不应破坏该结构约束。
    future_mask = np.broadcast_to(~mask, quantized_probabilities.shape)
    future_values = quantized_probabilities[future_mask]
    maximum_future_mass = float(np.max(future_values)) if future_values.size else 0.0
    return {
        "schema_version": 1,
        "configuration": {
            "seed": seed,
            "batch_size": batch_size,
            "query_heads": query_heads,
            "key_value_heads": key_value_heads,
            "cached_tokens": cached_tokens,
            "query_tokens": query_tokens,
            "key_head_dim": key_head_dim,
            "value_head_dim": value_head_dim,
            "quantization": "symmetric INT8 absmax per batch/KV-head/token vector",
            "code_range": [-127, 127],
        },
        "storage": {
            "reference_fp32_kv_bytes": cache.reference_fp32_bytes,
            "actual_int8_code_bytes": cache.int8_code_bytes,
            "float32_scale_metadata_bytes": cache.scale_metadata_bytes,
            "actual_code_plus_scale_payload_bytes": cache.payload_bytes,
            "payload_compression_ratio": cache.payload_compression_ratio,
        },
        "error": {
            "key": asdict(quantization_error(key, restored_key)),
            "value": asdict(quantization_error(value, restored_value)),
            "attention_probabilities": asdict(
                quantization_error(baseline_probabilities, quantized_probabilities)
            ),
            "attention_output": asdict(
                quantization_error(baseline_output, quantized_output)
            ),
        },
        "invariants": {
            "zero_key_vector_scale": float(cache.key_scales[0, 0, 0]),
            "zero_value_vector_scale": float(cache.value_scales[0, 0, 0]),
            "maximum_future_attention_probability": maximum_future_mass,
            "maximum_probability_row_sum_error": float(
                np.max(np.abs(np.sum(quantized_probabilities, axis=-1) - 1))
            ),
        },
        "scope": {
            "device": "CPU",
            "cache": "seeded synthetic K/V tensors",
            "actual_int8_codes_and_fp32_scales_materialized": True,
            "dequantized_gqa_attention_executed": True,
            "attention_on_int8_codes_or_fused_kv_kernel_executed": False,
            "paged_runtime_layout_or_resident_vram_measured": False,
            "target_model_quality_or_speed_proved": False,
        },
    }


def positive_integer(value: str) -> int:
    """供 argparse 复用的正整数校验器。"""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """定义 Q/K/V shape 和随机种子。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--batch-size", type=positive_integer, default=1)
    parser.add_argument("--query-heads", type=positive_integer, default=4)
    parser.add_argument("--key-value-heads", type=positive_integer, default=2)
    parser.add_argument("--cached-tokens", type=positive_integer, default=8)
    parser.add_argument("--query-tokens", type=positive_integer, default=3)
    parser.add_argument("--key-head-dim", type=positive_integer, default=16)
    parser.add_argument("--value-head-dim", type=positive_integer, default=16)
    return parser.parse_args()


def main() -> None:
    """运行 KV 量化实验并打印存储、误差和 Attention 不变量。"""

    args = parse_args()
    result = run_experiment(
        seed=args.seed,
        batch_size=args.batch_size,
        query_heads=args.query_heads,
        key_value_heads=args.key_value_heads,
        cached_tokens=args.cached_tokens,
        query_tokens=args.query_tokens,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
