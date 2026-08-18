"""PyTorch-backed paged KV storage built on the reference block allocator."""

from __future__ import annotations

import math
import threading

import torch
from torch import Tensor

from about_llm.inference.kv_allocator import (
    KVAllocatorReport,
    KVAppendResult,
    KVBlockState,
    KVSequenceState,
    PagedKVAllocator,
)


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


class KVTensorStorePoisonedError(RuntimeError):
    """Raised after a tensor backend failure may have desynchronized the store."""


class PagedKVTensorStore:
    """Store real K/V tensors in a fixed block arena.

    Appended tensors use ``[layers, tokens, kv_heads, head_dim]``. The backing
    arena uses ``[layers, physical_blocks, kv_heads, block_tokens, head_dim]``.
    This class validates allocator and tensor synchronization; its attention
    method gathers K/V and materializes dense scores rather than running a GPU
    PagedAttention kernel.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        total_blocks: int,
        block_size_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_layers = _positive_integer(num_layers, "num_layers")
        self.total_blocks = _positive_integer(total_blocks, "total_blocks")
        self.block_size_tokens = _positive_integer(
            block_size_tokens, "block_size_tokens"
        )
        self.num_kv_heads = _positive_integer(num_kv_heads, "num_kv_heads")
        self.head_dim = _positive_integer(head_dim, "head_dim")
        if dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise ValueError("dtype must be a floating-point torch dtype")
        self.dtype = dtype
        shape = (
            self.num_layers,
            self.total_blocks,
            self.num_kv_heads,
            self.block_size_tokens,
            self.head_dim,
        )
        self._key_blocks = torch.zeros(shape, dtype=dtype, device=device)
        self._value_blocks = torch.zeros(shape, dtype=dtype, device=device)
        self.device = self._key_blocks.device
        self._allocator = PagedKVAllocator(
            total_blocks=self.total_blocks,
            block_size_tokens=self.block_size_tokens,
        )
        self._lock = threading.RLock()
        self._poisoned_error: Exception | None = None

    @property
    def storage_shape(self) -> tuple[int, ...]:
        return tuple(self._key_blocks.shape)

    @property
    def resident_bytes(self) -> int:
        return (
            self._key_blocks.numel()
            + self._value_blocks.numel()
        ) * self._key_blocks.element_size()

    def create_sequence(self, sequence_id: str) -> None:
        with self._lock:
            self._ensure_healthy()
            self._allocator.create_sequence(sequence_id)

    def fork_sequence(self, parent_id: str, child_id: str) -> KVSequenceState:
        with self._lock:
            self._ensure_healthy()
            return self._allocator.fork_sequence(parent_id, child_id)

    def append(
        self,
        sequence_id: str,
        key: Tensor,
        value: Tensor,
    ) -> KVAppendResult:
        with self._lock:
            self._ensure_healthy()
            token_count = self._validate_append_tensors(key, value)
            old_length = self._allocator.sequence_state(sequence_id).length_tokens
            result = self._allocator.append(sequence_id, token_count)
            try:
                self._write_append(result, old_length, key, value)
            except Exception as error:
                self._poisoned_error = error
                raise KVTensorStorePoisonedError(
                    "tensor append failed after allocator commit; store is poisoned"
                ) from error
            return result

    def release_sequence(self, sequence_id: str) -> None:
        with self._lock:
            self._ensure_healthy()
            allocated_before = {state.block_id for state in self._allocator.block_states()}
            self._allocator.release_sequence(sequence_id)
            allocated_after = {state.block_id for state in self._allocator.block_states()}
            try:
                with torch.no_grad():
                    for block_id in allocated_before - allocated_after:
                        self._key_blocks[:, block_id].zero_()
                        self._value_blocks[:, block_id].zero_()
            except Exception as error:
                self._poisoned_error = error
                raise KVTensorStorePoisonedError(
                    "tensor release failed after allocator commit; store is poisoned"
                ) from error

    def materialize(self, sequence_id: str) -> tuple[Tensor, Tensor]:
        """Return defensive contiguous snapshots in logical token order."""

        with self._lock:
            self._ensure_healthy()
            state = self._allocator.sequence_state(sequence_id)
            if not state.physical_block_ids:
                shape = (self.num_layers, 0, self.num_kv_heads, self.head_dim)
                return (
                    torch.empty(shape, dtype=self.dtype, device=self.device),
                    torch.empty(shape, dtype=self.dtype, device=self.device),
                )
            occupancy = {
                block.block_id: block.used_tokens
                for block in self._allocator.block_states()
            }
            keys = [
                self._key_blocks[:, block_id, :, : occupancy[block_id]].permute(0, 2, 1, 3)
                for block_id in state.physical_block_ids
            ]
            values = [
                self._value_blocks[:, block_id, :, : occupancy[block_id]].permute(0, 2, 1, 3)
                for block_id in state.physical_block_ids
            ]
            return torch.cat(keys, dim=1).clone(), torch.cat(values, dim=1).clone()

    def attention(
        self,
        sequence_id: str,
        *,
        layer: int,
        query: Tensor,
        scale: float | None = None,
    ) -> Tensor:
        """Run causal MHA/GQA over the sequence's materialized paged K/V.

        ``query`` has shape ``[query_tokens, query_heads, head_dim]`` and is
        interpreted as the final ``query_tokens`` positions in the sequence.
        """

        with self._lock:
            self._ensure_healthy()
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ValueError("layer must be an integer")
        if not 0 <= layer < self.num_layers:
            raise ValueError("layer is outside the configured range")
        if query.ndim != 3 or query.shape[2] != self.head_dim:
            raise ValueError("query must have shape [query_tokens, query_heads, head_dim]")
        if query.shape[0] == 0:
            raise ValueError("query must contain at least one token")
        if query.shape[1] % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if query.dtype != self.dtype or query.device != self.device:
            raise ValueError("query dtype and device must match the KV store")
        if scale is None:
            attention_scale = 1.0 / math.sqrt(self.head_dim)
        elif (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("scale must be finite and positive")
        else:
            attention_scale = float(scale)

        key, value = self.materialize(sequence_id)
        sequence_length = key.shape[1]
        query_length, query_heads, _ = query.shape
        if query_length > sequence_length:
            raise ValueError("query cannot be longer than the cached sequence")
        head_groups = query_heads // self.num_kv_heads
        layer_key = key[layer].repeat_interleave(head_groups, dim=1).transpose(0, 1)
        layer_value = value[layer].repeat_interleave(head_groups, dim=1).transpose(0, 1)
        head_query = query.transpose(0, 1)
        scores = torch.matmul(head_query, layer_key.transpose(-2, -1)) * attention_scale
        query_start = sequence_length - query_length
        maximum_visible = query_start + torch.arange(query_length, device=self.device)
        key_positions = torch.arange(sequence_length, device=self.device)
        visible = key_positions.unsqueeze(0) <= maximum_visible.unsqueeze(1)
        scores = scores.masked_fill(~visible.unsqueeze(0), torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, layer_value).transpose(0, 1).contiguous()

    def sequence_state(self, sequence_id: str) -> KVSequenceState:
        with self._lock:
            self._ensure_healthy()
            return self._allocator.sequence_state(sequence_id)

    def block_states(self) -> tuple[KVBlockState, ...]:
        with self._lock:
            self._ensure_healthy()
            return self._allocator.block_states()

    def report(self) -> KVAllocatorReport:
        with self._lock:
            self._ensure_healthy()
            return self._allocator.report()

    def _write_append(
        self,
        result: KVAppendResult,
        old_length: int,
        key: Tensor,
        value: Tensor,
    ) -> None:
        with torch.no_grad():
            if result.copied_partial_block is not None:
                old_block_id, new_block_id = result.copied_partial_block
                self._key_blocks[:, new_block_id].copy_(
                    self._key_blocks[:, old_block_id]
                )
                self._value_blocks[:, new_block_id].copy_(
                    self._value_blocks[:, old_block_id]
                )
            for input_offset in range(result.appended_tokens):
                logical_position = old_length + input_offset
                logical_block = logical_position // self.block_size_tokens
                block_offset = logical_position % self.block_size_tokens
                physical_block = result.physical_block_ids[logical_block]
                self._key_blocks[:, physical_block, :, block_offset].copy_(
                    key[:, input_offset]
                )
                self._value_blocks[:, physical_block, :, block_offset].copy_(
                    value[:, input_offset]
                )

    def _ensure_healthy(self) -> None:
        if self._poisoned_error is not None:
            raise KVTensorStorePoisonedError(
                "a previous tensor update failed; store is poisoned"
            ) from self._poisoned_error

    def _validate_append_tensors(self, key: Tensor, value: Tensor) -> int:
        if not isinstance(key, Tensor) or not isinstance(value, Tensor):
            raise ValueError("key and value must be torch tensors")
        if key.shape != value.shape:
            raise ValueError("key and value must have identical shapes")
        if key.ndim != 4:
            raise ValueError("key and value must have shape [layers, tokens, kv_heads, head_dim]")
        expected_non_token_axes = (
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
        )
        if (key.shape[0], key.shape[2], key.shape[3]) != expected_non_token_axes:
            raise ValueError("key and value dimensions do not match the KV store")
        if key.shape[1] <= 0:
            raise ValueError("key and value must contain at least one token")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("key and value dtype must match the KV store")
        if key.device != self.device or value.device != self.device:
            raise ValueError("key and value device must match the KV store")
        return key.shape[1]