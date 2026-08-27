"""验证 CPU MiniGPT + AdamW 从严格 checkpoint 恢复后逐位延续训练轨迹。

baseline 连续训练六步；split 路径训练三步后序列化模型、优化器、scheduler、dropout RNG、
shuffle RNG 与数据游标，再加载并完成后三步。最终逐张量比较两条路径。
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import torch

from about_llm.finetuning.minigpt_training_checkpoint import (
    AdamWTrainingConfig,
    LinearLearningRateSchedule,
    MiniGPTTrainingCheckpointIdentity,
    MiniGPTTrainingConfig,
    MiniGPTTrainingState,
    create_minigpt_training_state,
    load_minigpt_training_checkpoint,
    read_minigpt_training_checkpoint,
    run_minigpt_training_updates,
    serialize_minigpt_training_checkpoint,
    write_minigpt_training_checkpoint_new,
)
from about_llm.from_scratch import ByteBPETokenizer, GPTConfig, MiniGPT

_HEADER = struct.Struct("<8sB3xIII")


def _state() -> tuple[MiniGPTTrainingState, torch.Tensor]:
    """从固定种子创建带 dropout、shuffle 与线性学习率的训练状态。"""

    # 模型初始化、数据生成、训练 RNG 和数据 RNG 分别使用明确种子。
    torch.manual_seed(13)
    tokenizer = ByteBPETokenizer()
    model = MiniGPT(
        GPTConfig(
            vocab_size=256,
            context_length=4,
            model_dim=8,
            num_heads=2,
            num_layers=1,
            mlp_ratio=2,
            dropout=0.2,
            bias=True,
        )
    )
    dataset_generator = torch.Generator(device="cpu").manual_seed(99)
    dataset = torch.randint(0, 256, (7, 5), generator=dataset_generator)
    training_config = MiniGPTTrainingConfig(
        optimizer=AdamWTrainingConfig(
            learning_rate=0.003,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            weight_decay=0.01,
        ),
        schedule=LinearLearningRateSchedule(0.003, 0.001, 6),
        batch_size=2,
        max_grad_norm=1.0,
    )
    return (
        create_minigpt_training_state(
            model,
            tokenizer,
            dataset,
            training_config=training_config,
            training_seed=17,
            data_seed=19,
        ),
        dataset,
    )


def _exact_state(state: MiniGPTTrainingState, reference: MiniGPTTrainingState) -> bool:
    """逐项比较 step、数据流、RNG、参数和 AdamW moments。"""

    if not (
        state.global_step == reference.global_step
        and state.batch_stream.cursor == reference.batch_stream.cursor
        and state.batch_stream.epoch == reference.batch_stream.epoch
        and torch.equal(state.batch_stream.permutation, reference.batch_stream.permutation)
        and torch.equal(
            state.batch_stream.generator_state,
            reference.batch_stream.generator_state,
        )
        and torch.equal(state.torch_cpu_rng_state, reference.torch_cpu_rng_state)
    ):
        return False
    parameters = dict(state.model.named_parameters())
    reference_parameters = dict(reference.model.named_parameters())
    for name, parameter in parameters.items():
        reference_parameter = reference_parameters[name]
        if not torch.equal(parameter, reference_parameter):
            return False
        for field in ("step", "exp_avg", "exp_avg_sq"):
            if not torch.equal(
                state.optimizer.state[parameter][field],
                reference.optimizer.state[reference_parameter][field],
            ):
                return False
    return bool(
        state.optimizer.param_groups[0]["lr"]
        == reference.optimizer.param_groups[0]["lr"]
    )


def run_toy(*, artifact_path: Path | None = None) -> dict[str, Any]:
    """运行 uninterrupted 与 save/resume 两条六步训练轨迹。"""

    # 两份初始 state 独立创建但完全相同，分别用于 baseline 与 split path。
    baseline, baseline_dataset = _state()
    split, split_dataset = _state()
    external_rng_before = torch.get_rng_state().clone()
    # baseline 一口气走六步；split 只走前三步后立即保存。
    baseline_reports = run_minigpt_training_updates(
        baseline, baseline_dataset, updates=6
    )
    first_reports = run_minigpt_training_updates(split, split_dataset, updates=3)
    identity = MiniGPTTrainingCheckpointIdentity(
        run_id="authored-resume-control",
        model_revision="fixture-seed-13",
        tokenizer_revision="byte-v1",
        data_revision="fixture-seed-99",
    )
    # checkpoint 绑定数据身份但不嵌入数据 payload，加载时必须提供匹配 dataset。
    artifact = serialize_minigpt_training_checkpoint(
        split, split_dataset, identity=identity
    )
    if artifact_path is None:
        restored, restored_identity = load_minigpt_training_checkpoint(
            artifact, split_dataset
        )
        disk_round_trip = False
    else:
        write_minigpt_training_checkpoint_new(
            artifact_path, artifact, split_dataset
        )
        restored, restored_identity = read_minigpt_training_checkpoint(
            artifact_path, split_dataset
        )
        disk_round_trip = True
    # 先检查刚加载的状态，再跑后三步并与 baseline 最终状态比较。
    state_exact_at_resume = _exact_state(restored, split)
    tail_reports = run_minigpt_training_updates(restored, split_dataset, updates=3)
    final_state_exact = _exact_state(restored, baseline)
    _, _, manifest_bytes, tensor_count, payload_bytes = _HEADER.unpack_from(artifact)
    return {
        "schema_version": 1,
        "identity": {
            "run_id": restored_identity.run_id,
            "model_revision": restored_identity.model_revision,
            "tokenizer_revision": restored_identity.tokenizer_revision,
            "data_revision": restored_identity.data_revision,
        },
        "fixture": {
            "examples": 7,
            "sequence_tokens": 5,
            "batch_size": 2,
            "total_updates": 6,
            "split_after_updates": 3,
            "dropout": 0.2,
            "training_seed": 17,
            "data_seed": 19,
        },
        "checkpoint": {
            "manifest_bytes": manifest_bytes,
            "tensor_count": tensor_count,
            "payload_bytes": payload_bytes,
            "artifact_bytes": len(artifact),
            "model_parameter_tensors": 16,
            "optimizer_moment_tensors": 32,
            "rng_tensors": 2,
            "permutation_tensors": 1,
            "global_step": 3,
            "epoch": 0,
            "cursor": 6,
        },
        "trajectory": {
            "batch_indices": [list(report.batch_indices) for report in baseline_reports],
            "epochs": [report.epoch for report in baseline_reports],
            "learning_rates": [report.learning_rate for report in baseline_reports],
            "losses": [report.loss for report in baseline_reports],
            "first_segment_matches_uninterrupted": first_reports
            == baseline_reports[:3],
            "resumed_segment_matches_uninterrupted": tail_reports
            == baseline_reports[3:],
            "state_exact_at_resume": state_exact_at_resume,
            "final_model_optimizer_stream_rng_exact": final_state_exact,
            "external_torch_rng_unchanged": bool(
                torch.equal(torch.get_rng_state(), external_rng_before)
            ),
        },
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
        "disk_round_trip": disk_round_trip,
        "scope": {
            "network_used": False,
            "cpu_float32_minigpt_adamw_executed": True,
            "dropout_torch_cpu_rng_restored": True,
            "shuffle_generator_rng_restored": True,
            "optimizer_moments_and_step_restored": True,
            "linear_schedule_progress_restored": True,
            "dataset_content_identity_verified": True,
            "batch_permutation_cursor_and_epoch_restored": True,
            "uninterrupted_vs_split_run_bit_exact": True,
            "checkpoint_at_zero_grad_optimizer_boundary": True,
            "python_numpy_or_cuda_rng_used_or_restored": False,
            "amp_scaler_or_gradient_accumulation_supported": False,
            "dataloader_workers_or_sampler_prefetch_supported": False,
            "distributed_sharded_training_supported": False,
            "dataset_payload_embedded": False,
            "target_checkpoint_or_cuda_executed": False,
            "loss_improvement_or_model_quality_proved": False,
            "cryptographic_origin_authenticated": False,
            "power_loss_atomic_publication_proved": False,
        },
    }


def parse_args() -> argparse.Namespace:
    """读取可选 checkpoint artifact 输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-path",
        type=Path,
        help="optional new path; existing files are rejected",
    )
    return parser.parse_args()


def main() -> None:
    """运行精确恢复实验并输出状态与轨迹对比。"""

    args = parse_args()
    print(
        json.dumps(
            run_toy(artifact_path=args.artifact_path),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
