"""模拟 Paged KV block 的前缀共享、copy-on-write 与容量失败。

request-b 从 request-a fork 后共享完整与部分 block；父序列继续追加时，未填满的共享尾 block
必须先复制。最后故意耗尽容量，并检查失败前后账本完全一致。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from about_llm.inference import KVCapacityError, PagedKVAllocator


def run_experiment() -> dict[str, Any]:
    """按固定时间线操作三块 KV block，并返回每一步引用计数。"""

    # 三块、每块四 token：首次追加六 token 会占用一个满 block 和一个半满 block。
    allocator = PagedKVAllocator(total_blocks=3, block_size_tokens=4)
    allocator.create_sequence("request-a")
    initial = allocator.append("request-a", 6)
    # fork 不复制数据，只增加前缀 block 的引用计数。
    allocator.fork_sequence("request-a", "request-b")
    after_fork = allocator.report()
    # 父序列修改共享的半满尾 block 前必须 COW，避免改变子序列看到的前缀。
    cow = allocator.append("request-a", 1)
    allocator.append("request-b", 2)
    before_failure = allocator.report()
    state_before_failure = allocator.sequence_state("request-a")
    failure: str | None = None
    # 再追加两 token 需要新 block，但容量已经用尽；操作必须原子失败。
    try:
        allocator.append("request-a", 2)
    except KVCapacityError as error:
        failure = str(error)
    state_after_failure = allocator.sequence_state("request-a")
    after_failure = allocator.report()
    # 释放父序列后，仍被 request-b 引用的共享 block 不能回到 free list。
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
    """打印 block 分配、引用计数、COW 和失败原子性。"""

    print(json.dumps(run_experiment(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
