"""Exercise prefix sharing, partial-tail COW, and atomic KV capacity failure."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from about_llm.inference import KVCapacityError, PagedKVAllocator


def run_experiment() -> dict[str, Any]:
    allocator = PagedKVAllocator(total_blocks=3, block_size_tokens=4)
    allocator.create_sequence("request-a")
    initial = allocator.append("request-a", 6)
    allocator.fork_sequence("request-a", "request-b")
    after_fork = allocator.report()
    cow = allocator.append("request-a", 1)
    allocator.append("request-b", 2)
    before_failure = allocator.report()
    state_before_failure = allocator.sequence_state("request-a")
    failure: str | None = None
    try:
        allocator.append("request-a", 2)
    except KVCapacityError as error:
        failure = str(error)
    state_after_failure = allocator.sequence_state("request-a")
    after_failure = allocator.report()
    allocator.release_sequence("request-a")
    after_release = allocator.report()

    return {
        "schema_version": 1,
        "configuration": {"total_blocks": 3, "block_size_tokens": 4},
        "initial_append": asdict(initial),
        "after_prefix_fork": asdict(after_fork),
        "copy_on_write_append": asdict(cow),
        "before_capacity_failure": asdict(before_failure),
        "capacity_failure": failure,
        "failure_was_atomic": (
            state_before_failure == state_after_failure
            and before_failure == after_failure
        ),
        "block_states_after_failure": [
            asdict(block) for block in allocator.block_states()
        ],
        "after_releasing_request_a": asdict(after_release),
        "scope": {
            "metadata_only_cpu_simulation": True,
            "real_kv_tensor_values_stored_or_copied": False,
            "paged_attention_gpu_kernel_executed": False,
            "eviction_preemption_or_swap_implemented": False,
            "latency_throughput_or_vram_proved": False,
        },
    }


def main() -> None:
    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
