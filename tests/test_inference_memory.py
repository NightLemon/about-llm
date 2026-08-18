import pytest

from about_llm.inference import estimate_kv_cache_bytes

pytestmark = pytest.mark.formula


def test_kv_cache_example_is_exactly_one_gibibyte() -> None:
    value = estimate_kv_cache_bytes(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        tokens=8192,
        bytes_per_element=2,
    )
    assert value == 1024**3


def test_kv_cache_scales_with_batch_and_rejects_invalid_dimensions() -> None:
    one = estimate_kv_cache_bytes(num_layers=2, num_kv_heads=2, head_dim=4, tokens=8)
    four = estimate_kv_cache_bytes(
        num_layers=2, num_kv_heads=2, head_dim=4, tokens=8, batch_size=4
    )
    assert four == 4 * one
    with pytest.raises(ValueError):
        estimate_kv_cache_bytes(num_layers=0, num_kv_heads=2, head_dim=4, tokens=8)
