"""Exercise real paged K/V tensors, prefix COW, and dense attention parity."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any

import torch

from about_llm.inference import PagedKVTensorStore


def run_experiment() -> dict[str, Any]:
    store = PagedKVTensorStore(
        num_layers=1,
        total_blocks=4,
        block_size_tokens=3,
        num_kv_heads=2,
        head_dim=2,
        dtype=torch.float64,
    )
    prefix_key = torch.arange(20, dtype=torch.float64).reshape(1, 5, 2, 2) / 10
    prefix_value = torch.arange(20, 40, dtype=torch.float64).reshape(1, 5, 2, 2) / 7
    next_key = torch.tensor([[[[4.0, 4.1], [4.2, 4.3]]]], dtype=torch.float64)
    next_value = torch.tensor([[[[8.0, 8.2], [8.4, 8.6]]]], dtype=torch.float64)
    store.create_sequence("request-a")
    store.append("request-a", prefix_key, prefix_value)
    store.fork_sequence("request-a", "request-b")
    cow = store.append("request-a", next_key, next_value)

    child_key, child_value = store.materialize("request-b")
    parent_key, parent_value = store.materialize("request-a")
    query = torch.tensor(
        [
            [[0.2, 0.1], [0.3, -0.2], [0.5, 0.4], [-0.1, 0.7]],
            [[0.4, 0.6], [-0.3, 0.8], [0.9, -0.2], [0.1, 0.5]],
        ],
        dtype=torch.float64,
    )
    observed = store.attention("request-a", layer=0, query=query)

    dense_key = parent_key[0].repeat_interleave(2, dim=1).transpose(0, 1)
    dense_value = parent_value[0].repeat_interleave(2, dim=1).transpose(0, 1)
    scores = query.transpose(0, 1) @ dense_key.transpose(-2, -1) / math.sqrt(2)
    causal = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )
    scores = scores.masked_fill(~causal.unsqueeze(0), torch.finfo(scores.dtype).min)
    expected = (torch.softmax(scores, dim=-1) @ dense_value).transpose(0, 1)
    report_before_release = store.report()
    store.release_sequence("request-a")

    return {
        "schema_version": 1,
        "configuration": {
            "num_layers": 1,
            "total_blocks": 4,
            "block_size_tokens": 3,
            "num_kv_heads": 2,
            "query_heads": 4,
            "head_dim": 2,
            "dtype": "float64",
            "device": str(store.device),
        },
        "storage_shape": store.storage_shape,
        "resident_bytes": store.resident_bytes,
        "copy_on_write_append": asdict(cow),
        "child_prefix_unchanged": (
            torch.equal(child_key, prefix_key) and torch.equal(child_value, prefix_value)
        ),
        "parent_append_materialized": (
            torch.equal(parent_key, torch.cat((prefix_key, next_key), dim=1))
            and torch.equal(parent_value, torch.cat((prefix_value, next_value), dim=1))
        ),
        "attention_matches_dense_reference": torch.allclose(
            observed, expected, rtol=1e-12, atol=1e-12
        ),
        "attention_checksum": float(observed.sum()),
        "allocator_before_release": asdict(report_before_release),
        "allocator_after_parent_release": asdict(store.report()),
        "scope": {
            "real_pytorch_kv_tensor_values_stored_and_copied": True,
            "fixed_cpu_tensor_arena_preallocated": True,
            "dense_causal_gqa_reference_compared": True,
            "paged_attention_gpu_kernel_executed": False,
            "scheduler_or_model_decode_integrated": False,
            "latency_throughput_or_vram_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()