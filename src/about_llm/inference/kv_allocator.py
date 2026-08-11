"""Metadata-only paged KV block allocator with prefix sharing and COW."""

from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass


class KVCapacityError(RuntimeError):
    """Raised when an append cannot reserve all required blocks atomically."""


@dataclass(frozen=True)
class KVAppendResult:
    sequence_id: str
    appended_tokens: int
    sequence_length: int
    physical_block_ids: tuple[int, ...]
    newly_allocated_block_ids: tuple[int, ...]
    copied_partial_block: tuple[int, int] | None


@dataclass(frozen=True)
class KVSequenceState:
    sequence_id: str
    length_tokens: int
    physical_block_ids: tuple[int, ...]


@dataclass(frozen=True)
class KVBlockState:
    block_id: int
    used_tokens: int
    reference_count: int


@dataclass(frozen=True)
class KVAllocatorReport:
    total_blocks: int
    allocated_blocks: int
    free_blocks: int
    block_size_tokens: int
    sequence_count: int
    logical_block_references: int
    sharing_saved_blocks: int
    shared_physical_blocks: int
    logical_tokens: int
    physical_token_values: int
    allocated_token_slots: int
    internal_fragmentation_slots: int
    physical_block_utilization: float


@dataclass
class _PhysicalBlock:
    used_tokens: int
    reference_count: int = 1


@dataclass
class _Sequence:
    length_tokens: int
    block_ids: list[int]


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")
    return value


class PagedKVAllocator:
    """Track fixed-size physical blocks without storing any real K/V tensors."""

    def __init__(self, *, total_blocks: int, block_size_tokens: int) -> None:
        self.total_blocks = _positive_integer(total_blocks, "total_blocks")
        self.block_size_tokens = _positive_integer(
            block_size_tokens, "block_size_tokens"
        )
        self._free_block_ids = list(range(total_blocks))
        heapq.heapify(self._free_block_ids)
        self._blocks: dict[int, _PhysicalBlock] = {}
        self._sequences: dict[str, _Sequence] = {}
        self._lock = threading.RLock()

    def create_sequence(self, sequence_id: str) -> None:
        sequence_id = _identifier(sequence_id, "sequence_id")
        with self._lock:
            if sequence_id in self._sequences:
                raise ValueError("sequence_id already exists")
            self._sequences[sequence_id] = _Sequence(0, [])
            self._check_invariants()

    def fork_sequence(self, parent_id: str, child_id: str) -> KVSequenceState:
        parent_id = _identifier(parent_id, "parent_id")
        child_id = _identifier(child_id, "child_id")
        with self._lock:
            parent = self._require_sequence(parent_id)
            if child_id in self._sequences:
                raise ValueError("child_id already exists")
            child_blocks = list(parent.block_ids)
            for block_id in child_blocks:
                self._blocks[block_id].reference_count += 1
            self._sequences[child_id] = _Sequence(
                parent.length_tokens, child_blocks
            )
            self._check_invariants()
            return self._sequence_state(child_id)

    def append(self, sequence_id: str, token_count: int) -> KVAppendResult:
        sequence_id = _identifier(sequence_id, "sequence_id")
        token_count = _positive_integer(token_count, "token_count")
        with self._lock:
            sequence = self._require_sequence(sequence_id)
            copy_tail = False
            tail_free_slots = 0
            if sequence.block_ids:
                tail = self._blocks[sequence.block_ids[-1]]
                tail_free_slots = self.block_size_tokens - tail.used_tokens
                copy_tail = tail_free_slots > 0 and tail.reference_count > 1

            tokens_after_tail = max(0, token_count - tail_free_slots)
            required_blocks = math.ceil(tokens_after_tail / self.block_size_tokens)
            if copy_tail:
                required_blocks += 1
            if required_blocks > len(self._free_block_ids):
                raise KVCapacityError(
                    f"append requires {required_blocks} free block(s), "
                    f"only {len(self._free_block_ids)} available"
                )

            reserved = [
                heapq.heappop(self._free_block_ids)
                for _ in range(required_blocks)
            ]
            reserved_index = 0
            copied_partial_block: tuple[int, int] | None = None
            if copy_tail:
                old_block_id = sequence.block_ids[-1]
                old_block = self._blocks[old_block_id]
                new_block_id = reserved[reserved_index]
                reserved_index += 1
                old_block.reference_count -= 1
                self._blocks[new_block_id] = _PhysicalBlock(old_block.used_tokens)
                sequence.block_ids[-1] = new_block_id
                copied_partial_block = (old_block_id, new_block_id)

            remaining = token_count
            if sequence.block_ids:
                tail_block = self._blocks[sequence.block_ids[-1]]
                available = self.block_size_tokens - tail_block.used_tokens
                appended_to_tail = min(available, remaining)
                tail_block.used_tokens += appended_to_tail
                remaining -= appended_to_tail

            while remaining > 0:
                block_id = reserved[reserved_index]
                reserved_index += 1
                used_tokens = min(self.block_size_tokens, remaining)
                self._blocks[block_id] = _PhysicalBlock(used_tokens)
                sequence.block_ids.append(block_id)
                remaining -= used_tokens

            if reserved_index != len(reserved):
                raise RuntimeError("allocator reservation accounting mismatch")
            sequence.length_tokens += token_count
            self._check_invariants()
            return KVAppendResult(
                sequence_id=sequence_id,
                appended_tokens=token_count,
                sequence_length=sequence.length_tokens,
                physical_block_ids=tuple(sequence.block_ids),
                newly_allocated_block_ids=tuple(reserved),
                copied_partial_block=copied_partial_block,
            )

    def release_sequence(self, sequence_id: str) -> None:
        sequence_id = _identifier(sequence_id, "sequence_id")
        with self._lock:
            sequence = self._require_sequence(sequence_id)
            for block_id in sequence.block_ids:
                block = self._blocks[block_id]
                block.reference_count -= 1
                if block.reference_count == 0:
                    del self._blocks[block_id]
                    heapq.heappush(self._free_block_ids, block_id)
            del self._sequences[sequence_id]
            self._check_invariants()

    def sequence_state(self, sequence_id: str) -> KVSequenceState:
        sequence_id = _identifier(sequence_id, "sequence_id")
        with self._lock:
            self._require_sequence(sequence_id)
            return self._sequence_state(sequence_id)

    def block_states(self) -> tuple[KVBlockState, ...]:
        with self._lock:
            return tuple(
                KVBlockState(block_id, block.used_tokens, block.reference_count)
                for block_id, block in sorted(self._blocks.items())
            )

    def report(self) -> KVAllocatorReport:
        with self._lock:
            allocated_blocks = len(self._blocks)
            logical_references = sum(
                len(sequence.block_ids) for sequence in self._sequences.values()
            )
            physical_tokens = sum(
                block.used_tokens for block in self._blocks.values()
            )
            allocated_slots = allocated_blocks * self.block_size_tokens
            return KVAllocatorReport(
                total_blocks=self.total_blocks,
                allocated_blocks=allocated_blocks,
                free_blocks=len(self._free_block_ids),
                block_size_tokens=self.block_size_tokens,
                sequence_count=len(self._sequences),
                logical_block_references=logical_references,
                sharing_saved_blocks=logical_references - allocated_blocks,
                shared_physical_blocks=sum(
                    block.reference_count > 1 for block in self._blocks.values()
                ),
                logical_tokens=sum(
                    sequence.length_tokens for sequence in self._sequences.values()
                ),
                physical_token_values=physical_tokens,
                allocated_token_slots=allocated_slots,
                internal_fragmentation_slots=allocated_slots - physical_tokens,
                physical_block_utilization=allocated_blocks / self.total_blocks,
            )

    def _require_sequence(self, sequence_id: str) -> _Sequence:
        try:
            return self._sequences[sequence_id]
        except KeyError as error:
            raise KeyError(f"unknown sequence_id: {sequence_id}") from error

    def _sequence_state(self, sequence_id: str) -> KVSequenceState:
        sequence = self._sequences[sequence_id]
        return KVSequenceState(
            sequence_id=sequence_id,
            length_tokens=sequence.length_tokens,
            physical_block_ids=tuple(sequence.block_ids),
        )

    def _check_invariants(self) -> None:
        free_ids = set(self._free_block_ids)
        allocated_ids = set(self._blocks)
        if (
            len(free_ids) != len(self._free_block_ids)
            or free_ids & allocated_ids
            or free_ids | allocated_ids != set(range(self.total_blocks))
        ):
            raise RuntimeError("free/allocated block partition is inconsistent")
        expected_references = {block_id: 0 for block_id in allocated_ids}
        for sequence in self._sequences.values():
            if len(sequence.block_ids) != len(set(sequence.block_ids)):
                raise RuntimeError("a sequence references the same block twice")
            if any(block_id not in self._blocks for block_id in sequence.block_ids):
                raise RuntimeError("sequence references an unknown physical block")
            represented_tokens = sum(
                self._blocks[block_id].used_tokens for block_id in sequence.block_ids
            )
            if represented_tokens != sequence.length_tokens:
                raise RuntimeError("sequence length and block occupancy disagree")
            if any(
                self._blocks[block_id].used_tokens != self.block_size_tokens
                for block_id in sequence.block_ids[:-1]
            ):
                raise RuntimeError("only a sequence tail block may be partial")
            for block_id in sequence.block_ids:
                expected_references[block_id] += 1
        for block_id, block in self._blocks.items():
            if not 1 <= block.used_tokens <= self.block_size_tokens:
                raise RuntimeError("physical block occupancy is out of bounds")
            if block.reference_count != expected_references[block_id]:
                raise RuntimeError("physical block reference count is inconsistent")
