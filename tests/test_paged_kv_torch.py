from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from about_llm.inference import (  # noqa: E402
    KVCapacityError,
    KVTensorStorePoisonedError,
    PagedKVTensorStore,
)

pytestmark = [pytest.mark.contract, pytest.mark.integration]


def _store(
    *, total_blocks: int = 4, block_size_tokens: int = 2
) -> PagedKVTensorStore:
    return PagedKVTensorStore(
        num_layers=2,
        total_blocks=total_blocks,
        block_size_tokens=block_size_tokens,
        num_kv_heads=2,
        head_dim=2,
        dtype=torch.float64,
    )


def _kv(token_count: int, *, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(
        start,
        start + 2 * token_count * 2 * 2,
        dtype=torch.float64,
    ).reshape(2, token_count, 2, 2)
    return values / 10, values / 7


def test_inference_package_lazily_exports_paged_kv_tensor_store() -> None:
    from about_llm import inference

    assert inference.PagedKVTensorStore is PagedKVTensorStore


def test_paged_store_round_trips_real_kv_and_reports_resident_bytes() -> None:
    store = _store(block_size_tokens=3)
    key, value = _kv(3)
    store.create_sequence("request")
    result = store.append("request", key, value)

    materialized_key, materialized_value = store.materialize("request")

    assert result.physical_block_ids == (0,)
    assert torch.equal(materialized_key, key)
    assert torch.equal(materialized_value, value)
    assert store.storage_shape == (2, 4, 2, 3, 2)
    assert store.resident_bytes == 2 * math.prod(store.storage_shape) * 8


def test_partial_tail_copy_on_write_preserves_child_tensor_values() -> None:
    store = _store()
    initial_key, initial_value = _kv(3)
    appended_key, appended_value = _kv(1, start=100)
    store.create_sequence("parent")
    store.append("parent", initial_key, initial_value)
    store.fork_sequence("parent", "child")

    result = store.append("parent", appended_key, appended_value)
    parent_key, parent_value = store.materialize("parent")
    child_key, child_value = store.materialize("child")

    assert result.copied_partial_block == (1, 2)
    assert torch.equal(child_key, initial_key)
    assert torch.equal(child_value, initial_value)
    assert torch.equal(parent_key, torch.cat((initial_key, appended_key), dim=1))
    assert torch.equal(parent_value, torch.cat((initial_value, appended_value), dim=1))


def test_capacity_failure_preserves_allocator_and_tensor_state() -> None:
    store = _store(total_blocks=2)
    key, value = _kv(3)
    store.create_sequence("request")
    store.append("request", key, value)
    before_state = store.sequence_state("request")
    before_key, before_value = store.materialize("request")

    extra_key, extra_value = _kv(2, start=100)
    with pytest.raises(KVCapacityError):
        store.append("request", extra_key, extra_value)

    after_key, after_value = store.materialize("request")
    assert store.sequence_state("request") == before_state
    assert torch.equal(after_key, before_key)
    assert torch.equal(after_value, before_value)


def test_release_clears_and_deterministically_reuses_physical_blocks() -> None:
    store = _store(total_blocks=2)
    key, value = _kv(2)
    store.create_sequence("first")
    first = store.append("first", key, value)
    store.release_sequence("first")
    assert torch.count_nonzero(store._key_blocks) == 0
    assert torch.count_nonzero(store._value_blocks) == 0
    store.create_sequence("second")
    zero = torch.zeros_like(key)
    second = store.append("second", zero, zero)

    materialized_key, materialized_value = store.materialize("second")
    assert first.physical_block_ids == second.physical_block_ids == (0,)
    assert torch.equal(materialized_key, zero)
    assert torch.equal(materialized_value, zero)


def test_full_shared_tail_append_and_parent_release_preserve_child() -> None:
    store = _store(total_blocks=3)
    key, value = _kv(2)
    extra_key, extra_value = _kv(1, start=100)
    store.create_sequence("parent")
    store.append("parent", key, value)
    store.fork_sequence("parent", "child")

    appended = store.append("parent", extra_key, extra_value)
    store.release_sequence("parent")
    child_key, child_value = store.materialize("child")

    assert appended.copied_partial_block is None
    assert appended.physical_block_ids == (0, 1)
    assert torch.equal(child_key, key)
    assert torch.equal(child_value, value)
    assert [(block.block_id, block.reference_count) for block in store.block_states()] == [
        (0, 1)
    ]


def test_tensor_backend_update_failure_poisons_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store()
    key, value = _kv(1)
    store.create_sequence("request")

    def fail_write(*args: object) -> None:
        raise RuntimeError("injected tensor backend failure")

    monkeypatch.setattr(store, "_write_append", fail_write)
    with pytest.raises(KVTensorStorePoisonedError, match="store is poisoned"):
        store.append("request", key, value)
    with pytest.raises(KVTensorStorePoisonedError, match="previous tensor update failed"):
        store.materialize("request")
    with pytest.raises(KVTensorStorePoisonedError, match="previous tensor update failed"):
        store.release_sequence("request")


@pytest.mark.formula
def test_paged_attention_matches_independent_dense_causal_gqa_reference() -> None:
    store = _store()
    key, value = _kv(3)
    store.create_sequence("request")
    store.append("request", key, value)
    query = torch.tensor(
        [
            [[0.2, 0.1], [0.3, -0.2], [0.5, 0.4], [-0.1, 0.7]],
            [[0.4, 0.6], [-0.3, 0.8], [0.9, -0.2], [0.1, 0.5]],
        ],
        dtype=torch.float64,
    )

    observed = store.attention("request", layer=1, query=query)

    dense_key = key[1].repeat_interleave(2, dim=1).transpose(0, 1)
    dense_value = value[1].repeat_interleave(2, dim=1).transpose(0, 1)
    dense_query = query.transpose(0, 1)
    scores = dense_query @ dense_key.transpose(-2, -1) / math.sqrt(2)
    causal = torch.tensor([[True, True, False], [True, True, True]])
    scores = scores.masked_fill(~causal.unsqueeze(0), torch.finfo(scores.dtype).min)
    expected = (torch.softmax(scores, dim=-1) @ dense_value).transpose(0, 1)

    assert torch.allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_store_rejects_shape_dtype_and_attention_contract_drift() -> None:
    store = _store()
    store.create_sequence("request")
    key, value = _kv(2)

    with pytest.raises(ValueError, match="identical shapes"):
        store.append("request", key, value[:, :1])
    with pytest.raises(ValueError, match="dtype must match"):
        store.append("request", key.float(), value.float())

    store.append("request", key, value)
    with pytest.raises(ValueError, match="divisible"):
        store.attention(
            "request",
            layer=0,
            query=torch.zeros((1, 3, 2), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="cannot be longer"):
        store.attention(
            "request",
            layer=0,
            query=torch.zeros((3, 2, 2), dtype=torch.float64),
        )