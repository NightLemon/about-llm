import pytest

from about_llm.inference import (
    estimate_causal_generation_forward_positions,
    estimate_kv_cache_bytes,
)

pytestmark = pytest.mark.formula


def test_causal_generation_work_matches_toy_and_prefix_reuse_ledgers() -> None:
    assert estimate_causal_generation_forward_positions(
        prompt_tokens=4,
        output_tokens=3,
    ) == 6
    assert estimate_causal_generation_forward_positions(
        prompt_tokens=768,
        output_tokens=8,
    ) == 775
    assert estimate_causal_generation_forward_positions(
        prompt_tokens=768,
        output_tokens=8,
        cached_prompt_tokens=512,
    ) == 263
    assert estimate_causal_generation_forward_positions(
        prompt_tokens=768,
        output_tokens=8,
        cached_prompt_tokens=256,
    ) == 519


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt_tokens": 0, "output_tokens": 1},
        {"prompt_tokens": 1, "output_tokens": 0},
        {"prompt_tokens": 1, "output_tokens": 1, "cached_prompt_tokens": -1},
        {"prompt_tokens": 1, "output_tokens": 1, "cached_prompt_tokens": 1},
        {"prompt_tokens": True, "output_tokens": 1},
    ],
)
def test_causal_generation_work_rejects_invalid_or_fully_cached_prompt(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        estimate_causal_generation_forward_positions(**kwargs)


def test_kv_cache_example_is_exactly_one_gibibyte() -> None:
    value = estimate_kv_cache_bytes(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        tokens=8192,
        bytes_per_element=2,
    )
    assert value == 1024**3


def test_gqa_whiteboard_example_scales_with_kv_head_count() -> None:
    common = {
        "num_layers": 32,
        "head_dim": 128,
        "tokens": 4096,
        "batch_size": 1,
        "bytes_per_element": 2,
    }
    mha = estimate_kv_cache_bytes(num_kv_heads=32, **common)
    gqa = estimate_kv_cache_bytes(num_kv_heads=8, **common)

    assert mha == 2 * 1024**3
    assert gqa == 512 * 1024**2
    assert mha == 4 * gqa


def test_kv_cache_scales_with_batch_and_rejects_invalid_dimensions() -> None:
    one = estimate_kv_cache_bytes(num_layers=2, num_kv_heads=2, head_dim=4, tokens=8)
    four = estimate_kv_cache_bytes(
        num_layers=2, num_kv_heads=2, head_dim=4, tokens=8, batch_size=4
    )
    assert four == 4 * one
    with pytest.raises(ValueError):
        estimate_kv_cache_bytes(num_layers=0, num_kv_heads=2, head_dim=4, tokens=8)
