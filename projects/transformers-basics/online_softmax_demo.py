"""Run a deterministic blockwise online-softmax correctness control."""

from __future__ import annotations

import json

import numpy as np

from about_llm.from_scratch import (
    blockwise_online_attention,
    causal_mask,
    scaled_dot_product_attention,
)


def build_report() -> dict[str, object]:
    """Compare the online recurrence with a dense NumPy reference."""
    rng = np.random.default_rng(47)
    query = rng.normal(size=(5, 4))
    key = rng.normal(size=(7, 4))
    value = rng.normal(size=(7, 3))
    mask = causal_mask(query_length=5, key_length=7)
    dense_output, _ = scaled_dot_product_attention(query, key, value, mask=mask)
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
    print(json.dumps(build_report(), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
