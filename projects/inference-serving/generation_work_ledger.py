"""Compare causal-generation work with and without nano-vLLM prefix reuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from about_llm.inference import estimate_causal_generation_forward_positions
from about_llm.inference.nano_vllm_study import load_study_manifest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "projects"
    / "inference-serving"
    / "nano-vllm-qwen3-0.6b.study.json"
)


def _scenario(
    name: str,
    *,
    prompt_tokens: int,
    output_tokens: int,
    cached_prompt_tokens: int,
) -> dict[str, Any]:
    scheduled_prefill_tokens = prompt_tokens - cached_prompt_tokens
    decode_positions = output_tokens - 1
    return {
        "name": name,
        "logical_prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "scheduled_prefill_tokens": scheduled_prefill_tokens,
        "decode_positions": decode_positions,
        "evaluated_forward_positions": estimate_causal_generation_forward_positions(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        ),
    }


def build_ledger() -> dict[str, Any]:
    manifest = load_study_manifest(MANIFEST)
    workload = manifest["workload"]
    engine = manifest["engine"]
    prompt_tokens = int(workload["prompt_tokens"])
    output_tokens = int(workload["output_tokens"])
    block_size = int(engine["kvcache_block_size"])
    exact_cached_tokens = int(workload["prefix_cached_blocks"]) * block_size
    drift_index = int(workload["drift_token_index"])
    drift_cached_tokens = (drift_index // block_size) * block_size

    if not 0 <= drift_cached_tokens < exact_cached_tokens < prompt_tokens:
        raise ValueError("manifest prefix settings do not form the expected work ledger")

    scenarios = [
        _scenario(
            "no_prefix_reuse",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            cached_prompt_tokens=0,
        ),
        _scenario(
            "exact_prefix",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            cached_prompt_tokens=exact_cached_tokens,
        ),
        _scenario(
            "one_token_drift",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            cached_prompt_tokens=drift_cached_tokens,
        ),
    ]
    baseline = int(scenarios[0]["evaluated_forward_positions"])
    for scenario in scenarios:
        scenario["positions_saved_vs_no_reuse"] = (
            baseline - int(scenario["evaluated_forward_positions"])
        )

    return {
        "ledger_version": "about-llm.generation-work-ledger.v1",
        "manifest": {
            "study_id": manifest["study_id"],
            "manifest_fingerprint": manifest["manifest_fingerprint"],
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "block_size_tokens": block_size,
            "drift_token_index": drift_index,
        },
        "scenarios": scenarios,
        "scope": {
            "standard_decoder_only_generation_modeled": True,
            "first_output_produced_by_final_scheduled_prefill_position": True,
            "prefix_cache_savings_counted_as_skipped_prompt_positions": True,
            "chunked_prefill_changes_step_count_not_total_prefill_positions": True,
            "gpu_model_or_nano_vllm_engine_executed": False,
            "speculation_beam_recomputation_padding_or_kernel_work_modeled": False,
        },
    }


def main() -> None:
    print(json.dumps(build_ledger(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
