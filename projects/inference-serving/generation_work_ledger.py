"""计算 nano-vLLM 实验中无缓存、精确前缀和单 token 漂移的 forward 工作量。

它读取实验 manifest 中的 prompt、输出长度和 KV block 大小，用“实际评估过多少 token 位置”
统一计算 prefill 与 decode。这里只做静态账本，不启动 nano-vLLM 或 GPU。
"""

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
    """计算一个缓存场景的 prefill、decode 与总 forward 位置数。"""

    # 第一个输出 token 由最后一个 prefill 位置产生，后续 output_tokens-1 才是 decode。
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
    """从固定 study manifest 推导三种前缀复用场景。"""

    # 直接读取 manifest，确保教学账本与真实 GPU runner 使用同一组参数。
    manifest = load_study_manifest(MANIFEST)
    workload = manifest["workload"]
    engine = manifest["engine"]
    prompt_tokens = int(workload["prompt_tokens"])
    output_tokens = int(workload["output_tokens"])
    block_size = int(engine["kvcache_block_size"])
    # prefix cache 只复用完整 block；漂移所在 block 及其后内容都必须重算。
    exact_cached_tokens = int(workload["prefix_cached_blocks"]) * block_size
    drift_index = int(workload["drift_token_index"])
    drift_cached_tokens = (drift_index // block_size) * block_size

    if not 0 <= drift_cached_tokens < exact_cached_tokens < prompt_tokens:
        raise ValueError("manifest prefix settings do not form the expected work ledger")

    # 三个场景只改变已缓存 prompt 长度，逻辑 prompt 和输出长度保持一致。
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
    # 以完全无缓存为基线，saved positions 才有直观含义。
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
    """输出三种 prefix cache 场景的工作量账本。"""

    print(json.dumps(build_ledger(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
