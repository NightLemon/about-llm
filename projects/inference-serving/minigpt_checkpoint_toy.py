"""导出、严格重载并运行仓库原生的量化 MiniGPT checkpoint。

artifact 同时保存模型结构、版本身份、Byte-BPE merges、全部参数和 tied-weight 关系。实验从
随机 FP32 模型量化导出，再仅凭 artifact 重建 tokenizer 与模型，完成前向和贪心生成。
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import torch

from about_llm.from_scratch import ByteBPETokenizer, GPTConfig, MiniGPT
from about_llm.inference import (
    MiniGPTCheckpointIdentity,
    load_quantized_minigpt_checkpoint,
    read_quantized_minigpt_checkpoint,
    serialize_quantized_minigpt_checkpoint,
    write_quantized_minigpt_checkpoint_new,
)

_HEADER = struct.Struct("<8sB3xIII")


def run_toy(
    *,
    seed: int,
    bit_width: int,
    group_size: int,
    prompt: str,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """构建 MiniGPT checkpoint，重新加载并比较量化前后的 logits。"""

    # 随机种子同时决定模型权重和 artifact 中记录的模型 revision。
    torch.manual_seed(seed)

    # 两条手写 merge 让 tokenizer payload 很小，但仍会生成高于 255 的 BPE token。
    tokenizer = ByteBPETokenizer(((97, 98), (256, 99)))
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=8,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2,
        dropout=0,
        bias=True,
    )
    fp32_model = MiniGPT(config).eval()
    identity = MiniGPTCheckpointIdentity(
        model_id="authored-minigpt",
        model_revision=f"fixture-seed-{seed}",
        tokenizer_revision="authored-merges-v1",
    )
    # 序列化器会保留向量参数为 FP32，并对合适的矩阵参数做低比特量化。
    artifact = serialize_quantized_minigpt_checkpoint(
        fp32_model,
        tokenizer,
        identity=identity,
        bit_width=bit_width,
        group_size=group_size,
    )
    # 第一次从内存 bytes 重载；指定路径时再验证写盘与读盘路径。
    loaded = load_quantized_minigpt_checkpoint(artifact)
    disk_round_trip = False
    if artifact_path is not None:
        write_quantized_minigpt_checkpoint_new(artifact_path, artifact)
        loaded = read_quantized_minigpt_checkpoint(artifact_path)
        disk_round_trip = True

    # 使用 artifact 中同一 tokenizer 语义编码 prompt，并检查上下文窗口。
    input_ids_list = tokenizer.encode(prompt)
    if not input_ids_list:
        raise ValueError("prompt must encode to at least one token")
    if len(input_ids_list) > config.context_length:
        raise ValueError("prompt exceeds the authored MiniGPT context length")
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)
    # 再独立加载一次，检查相同 bytes 是否重建出逐位相同的量化 logits。
    repeated = load_quantized_minigpt_checkpoint(artifact)
    # 三条前向分别给出 FP32 参考、量化输出和重复加载确定性；最后做三 token 贪心生成。
    with torch.inference_mode():
        fp32_logits, _ = fp32_model(input_ids)
        quantized_logits, _ = loaded.model(input_ids)
        repeated_logits, _ = repeated.model(input_ids)
        generated = loaded.model.generate(input_ids, 3, temperature=0)
    logit_delta = fp32_logits - quantized_logits
    _, _, manifest_bytes, parameter_count, payload_bytes = _HEADER.unpack_from(artifact)
    unique_parameters = tuple(fp32_model.named_parameters())
    return {
        "schema_version": 1,
        "configuration": {
            "seed": seed,
            "bit_width": bit_width,
            "group_size": group_size,
            "generation_max_new_tokens": 3,
            "generation_strategy": "greedy",
        },
        "identity": {
            "model_id": loaded.identity.model_id,
            "model_revision": loaded.identity.model_revision,
            "tokenizer_revision": loaded.identity.tokenizer_revision,
        },
        "config": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "model_dim": config.model_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "mlp_ratio": config.mlp_ratio,
            "bias": config.bias,
            "tied_token_embedding_and_lm_head": bool(
                loaded.model.lm_head.weight is loaded.model.token_embedding.weight
            ),
        },
        "tokenizer": {
            "kind": "about-llm.byte-bpe",
            "merges": [list(pair) for pair in loaded.tokenizer.merges],
            "vocab_size": loaded.tokenizer.vocab_size,
            "prompt": prompt,
            "prompt_token_ids": input_ids_list,
            "round_trip_text": loaded.tokenizer.decode(input_ids_list),
        },
        "storage": {
            "unique_parameter_count": len(unique_parameters),
            "header_parameter_count": parameter_count,
            "reference_fp32_parameter_bytes": sum(
                parameter.numel() * parameter.element_size()
                for _, parameter in unique_parameters
            ),
            "manifest_bytes": manifest_bytes,
            "parameter_payload_bytes": payload_bytes,
            "checkpoint_artifact_bytes": len(artifact),
            "container_overhead_bytes": len(artifact) - payload_bytes,
        },
        "forward": {
            "logits_shape": list(quantized_logits.shape),
            "logit_rmse_vs_fp32": torch.sqrt(torch.mean(logit_delta**2)).item(),
            "logit_max_abs_error_vs_fp32": torch.max(torch.abs(logit_delta)).item(),
            "exact_repeated_load_logits": bool(
                torch.equal(quantized_logits, repeated_logits)
            ),
            "greedy_generated_token_ids": generated[0].tolist(),
        },
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "disk_round_trip": disk_round_trip,
        "scope": {
            "network_used": False,
            "byte_bpe_merge_payload_embedded": True,
            "architecture_config_and_revision_embedded": True,
            "all_unique_model_parameters_embedded": True,
            "quantized_matrices_and_float32_vectors_embedded": True,
            "tied_lm_head_restored": True,
            "repo_minigpt_forward_executed": True,
            "full_repo_native_minigpt_inference_checkpoint": True,
            "forward_source_code_embedded": False,
            "trusted_repo_loader_required": True,
            "normalizer_special_tokens_or_chat_template_supported": False,
            "optimizer_rng_or_training_resume_state_embedded": False,
            "gguf_safetensors_or_external_runtime_compatible": False,
            "packed_low_bit_kernel_executed": False,
            "pretrained_or_target_llm_quality_proved": False,
            "resident_vram_latency_or_speedup_measured": False,
            "cryptographic_origin_authenticated": False,
            "general_purpose_llm_checkpoint": False,
        },
    }


def parse_args() -> argparse.Namespace:
    """定义量化配置、prompt 和可选 artifact 输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bit-width", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--prompt", default="abc abc")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        help="optional new path; existing files are rejected",
    )
    return parser.parse_args()


def main() -> None:
    """运行 checkpoint 导出与推理闭环并打印结构、存储和误差。"""

    args = parse_args()
    print(
        json.dumps(
            run_toy(
                seed=args.seed,
                bit_width=args.bit_width,
                group_size=args.group_size,
                prompt=args.prompt,
                artifact_path=args.artifact_path,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
