"""Transparent first-order memory estimates for single-GPU QLoRA planning."""

from __future__ import annotations

from dataclasses import dataclass

_GIB = 1024**3


@dataclass(frozen=True)
class QLoRAMemoryEstimate:
    quantized_base_gib: float
    adapter_and_optimizer_gib: float
    activations_gib: float
    temporary_and_runtime_gib: float
    total_gib: float
    lora_parameter_count: int


def estimate_qlora_memory(
    *,
    num_parameters: int,
    num_layers: int,
    hidden_size: int,
    sequence_length: int,
    micro_batch_size: int = 1,
    lora_rank: int = 16,
    target_linears_per_layer: int = 7,
    activation_checkpointing: bool = True,
    quantization_overhead: float = 1.18,
    runtime_reserve_gib: float = 1.0,
) -> QLoRAMemoryEstimate:
    """Estimate major allocations; use a real dry-run for the final capacity decision.

    The approximation assumes square target projections and AdamW-like adapter
    states. Architecture, kernels, allocator fragmentation, attention backend,
    vocabulary logits and paged optimizer residency can move the observed peak.
    """
    integer_values = (
        num_parameters,
        num_layers,
        hidden_size,
        sequence_length,
        micro_batch_size,
        lora_rank,
        target_linears_per_layer,
    )
    if any(value <= 0 for value in integer_values):
        raise ValueError("model, sequence, batch, rank and target counts must be positive")
    if quantization_overhead < 1:
        raise ValueError("quantization_overhead must be at least 1")
    if runtime_reserve_gib < 0:
        raise ValueError("runtime_reserve_gib cannot be negative")

    base_bytes = num_parameters * 0.5 * quantization_overhead
    lora_parameters = num_layers * target_linears_per_layer * 2 * hidden_size * lora_rank
    # BF16 weights + BF16 gradients + FP32 master/Adam first and second moments.
    adapter_bytes = lora_parameters * (2 + 2 + 4 + 4 + 4)
    checkpoint_factor = 0.35 if activation_checkpointing else 1.0
    activation_bytes = (
        micro_batch_size
        * sequence_length
        * hidden_size
        * num_layers
        * 2
        * 8
        * checkpoint_factor
    )
    # Dequantization workspaces, CUDA context, allocator fragmentation and kernels.
    temporary_bytes = base_bytes * 0.10 + runtime_reserve_gib * _GIB
    total_bytes = base_bytes + adapter_bytes + activation_bytes + temporary_bytes
    return QLoRAMemoryEstimate(
        quantized_base_gib=base_bytes / _GIB,
        adapter_and_optimizer_gib=adapter_bytes / _GIB,
        activations_gib=activation_bytes / _GIB,
        temporary_and_runtime_gib=temporary_bytes / _GIB,
        total_gib=total_bytes / _GIB,
        lora_parameter_count=lora_parameters,
    )


def oom_degradation_order() -> tuple[str, ...]:
    """Return the preferred changes, preserving experiment meaning where possible."""
    return (
        "reduce micro-batch size to 1 and keep effective batch via gradient accumulation",
        "enable gradient checkpointing and a memory-efficient attention backend",
        "reduce sequence length after measuring the training length distribution",
        "reduce LoRA target modules or rank and report the changed trainable capacity",
        "select a smaller base model; do not silently change the evaluation baseline",
    )
