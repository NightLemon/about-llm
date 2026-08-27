"""用一个随机初始化的 MiniGPT 演示残差流 activation patching。

实验先对“干净输入”和“受损输入”分别做前向传播，再把干净运行中指定位置的
残差激活替换进受损运行。若目标 token 的分数差恢复，说明该位置携带了与指标有关的信息。
这里的模型没有训练，因此只能验证 patching 机制，不能据此解释真实语言能力。
"""

from __future__ import annotations

import json

import torch

from about_llm.from_scratch.activation_patching import run_residual_patch_experiment
from about_llm.from_scratch.gpt_torch import GPTConfig, MiniGPT


def main() -> None:
    """构造四种 patch 位置，对比它们恢复目标分数差的程度。"""

    # 固定随机种子，使随机权重和每次运行的 patching 结果保持一致。
    torch.manual_seed(23)

    # 两层小模型足以产生真实 residual hook，同时仍能在 CPU 上快速执行。
    model = MiniGPT(
        GPTConfig(
            vocab_size=32,
            context_length=8,
            model_dim=16,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2,
        )
    ).eval()
    # clean 与 corrupted 只在第 0 个 token 不同，便于追踪差异沿因果方向传播。
    clean = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    corrupted = torch.tensor([[5, 2, 3, 4]], dtype=torch.long)

    # 前三组 patch 覆盖可能影响指标位置的因果前缀；最后一组未来位置是负对照。
    configurations = {
        "changed_source_position": (0,),
        "metric_position": (1,),
        "joint_causal_prefix": (0, 1),
        "future_position_negative_control": (2,),
    }
    # 每组实验都会分别执行 clean、corrupted 和 patched 三次真实前向传播。
    results = {
        name: run_residual_patch_experiment(
            model,
            clean,
            corrupted,
            layer_index=0,
            positions=positions,
            metric_position=1,
            positive_token_id=27,
            negative_token_id=19,
        ).to_dict()
        for name, positions in configurations.items()
    }
    print(
        json.dumps(
            {
                "experiment": "seeded-random-minigpt-residual-patching-v1",
                "seed": 23,
                "clean_input_ids": clean.tolist(),
                "corrupted_input_ids": corrupted.tolist(),
                "metric_selection": (
                    "authored post-hoc token pair maximizing this fixture's "
                    "clean-minus-corrupt contrast"
                ),
                "results": results,
                # 报告明确区分“hook 确实执行”和“发现了真实模型电路”这两种结论。
                "scope": {
                    "actual_forward_hooks_executed": True,
                    "clean_corrupt_batch_alignment_required": True,
                    "normalized_recovery_clipped": False,
                    "trained_language_behavior": False,
                    "target_checkpoint_tested": False,
                    "unique_natural_circuit_proved": False,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
