import pytest

from about_llm.finetuning import estimate_qlora_memory, oom_degradation_order

pytestmark = pytest.mark.formula


def test_memory_estimate_exposes_components_and_scales_with_sequence() -> None:
    common = dict(
        num_parameters=7_000_000_000,
        num_layers=32,
        hidden_size=4096,
        lora_rank=16,
    )
    short = estimate_qlora_memory(sequence_length=512, **common)
    long = estimate_qlora_memory(sequence_length=2048, **common)

    assert 3 < short.quantized_base_gib < 5
    assert short.lora_parameter_count > 0
    assert long.activations_gib == pytest.approx(short.activations_gib * 4)
    assert long.total_gib > short.total_gib


def test_checkpointing_reduces_only_activation_estimate() -> None:
    kwargs = dict(
        num_parameters=1_000_000_000,
        num_layers=24,
        hidden_size=2048,
        sequence_length=1024,
    )
    checkpointed = estimate_qlora_memory(activation_checkpointing=True, **kwargs)
    plain = estimate_qlora_memory(activation_checkpointing=False, **kwargs)

    assert checkpointed.quantized_base_gib == plain.quantized_base_gib
    assert checkpointed.activations_gib < plain.activations_gib


def test_invalid_estimator_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError):
        estimate_qlora_memory(
            num_parameters=0,
            num_layers=1,
            hidden_size=1,
            sequence_length=1,
        )
    assert "micro-batch" in oom_degradation_order()[0]
