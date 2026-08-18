from __future__ import annotations

from pathlib import Path

import pytest

from about_llm.inference import KVCapacityError, PagedKVAllocator

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]


def test_append_allocates_fixed_blocks_and_reports_physical_fragmentation() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=4)
    allocator.create_sequence("a")
    result = allocator.append("a", 6)

    assert result.physical_block_ids == (0, 1)
    assert result.newly_allocated_block_ids == (0, 1)
    assert result.copied_partial_block is None
    assert [(block.used_tokens, block.reference_count) for block in allocator.block_states()] == [
        (4, 1),
        (2, 1),
    ]
    report = allocator.report()
    assert report.allocated_token_slots == 8
    assert report.physical_token_values == 6
    assert report.internal_fragmentation_slots == 2


def test_fork_shares_prefix_and_partial_append_uses_copy_on_write() -> None:
    allocator = PagedKVAllocator(total_blocks=4, block_size_tokens=4)
    allocator.create_sequence("a")
    allocator.append("a", 6)
    child = allocator.fork_sequence("a", "b")

    assert child.physical_block_ids == (0, 1)
    assert allocator.report().sharing_saved_blocks == 2
    assert allocator.report().shared_physical_blocks == 2

    result = allocator.append("a", 1)
    assert result.copied_partial_block == (1, 2)
    assert result.physical_block_ids == (0, 2)
    assert allocator.sequence_state("b").physical_block_ids == (0, 1)
    observed_blocks = [
        (state.block_id, state.used_tokens, state.reference_count)
        for state in allocator.block_states()
    ]
    assert observed_blocks == [
        (0, 4, 2),
        (1, 2, 1),
        (2, 3, 1),
    ]


def test_append_after_shared_full_tail_allocates_without_copy() -> None:
    allocator = PagedKVAllocator(total_blocks=3, block_size_tokens=4)
    allocator.create_sequence("a")
    allocator.append("a", 4)
    allocator.fork_sequence("a", "b")

    result = allocator.append("a", 1)
    assert result.copied_partial_block is None
    assert result.newly_allocated_block_ids == (1,)
    assert result.physical_block_ids == (0, 1)
    assert allocator.sequence_state("b").physical_block_ids == (0,)


def test_capacity_failure_is_atomic_before_mutating_an_exclusive_tail() -> None:
    allocator = PagedKVAllocator(total_blocks=2, block_size_tokens=4)
    allocator.create_sequence("a")
    allocator.append("a", 6)
    before_state = allocator.sequence_state("a")
    before_blocks = allocator.block_states()
    before_report = allocator.report()

    with pytest.raises(KVCapacityError, match="requires 1 free block"):
        allocator.append("a", 3)

    assert allocator.sequence_state("a") == before_state
    assert allocator.block_states() == before_blocks
    assert allocator.report() == before_report


def test_shared_partial_tail_requires_a_free_cow_block() -> None:
    allocator = PagedKVAllocator(total_blocks=2, block_size_tokens=4)
    allocator.create_sequence("a")
    allocator.append("a", 6)
    allocator.fork_sequence("a", "b")
    before = allocator.report()

    with pytest.raises(KVCapacityError, match="requires 1 free block"):
        allocator.append("a", 1)

    assert allocator.sequence_state("a").physical_block_ids == (0, 1)
    assert allocator.sequence_state("b").physical_block_ids == (0, 1)
    assert allocator.report() == before


def test_release_decrements_references_and_recycles_blocks_deterministically() -> None:
    allocator = PagedKVAllocator(total_blocks=3, block_size_tokens=4)
    allocator.create_sequence("a")
    allocator.append("a", 5)
    allocator.fork_sequence("a", "b")
    allocator.release_sequence("a")

    assert allocator.report().allocated_blocks == 2
    assert all(block.reference_count == 1 for block in allocator.block_states())
    allocator.release_sequence("b")
    assert allocator.report().allocated_blocks == 0
    allocator.create_sequence("c")
    assert allocator.append("c", 1).newly_allocated_block_ids == (0,)


def test_allocator_rejects_invalid_and_unknown_sequences() -> None:
    with pytest.raises(ValueError):
        PagedKVAllocator(total_blocks=0, block_size_tokens=4)
    allocator = PagedKVAllocator(total_blocks=2, block_size_tokens=4)
    with pytest.raises(ValueError):
        allocator.create_sequence("")
    allocator.create_sequence("a")
    with pytest.raises(ValueError, match="already exists"):
        allocator.create_sequence("a")
    with pytest.raises(KeyError, match="unknown sequence"):
        allocator.append("missing", 1)
    with pytest.raises(ValueError, match="positive integer"):
        allocator.append("a", 0)

