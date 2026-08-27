"""在共享中文样本上追踪 MiniGPT 的一次完整训练步。

输出从 token IDs 和 labels 开始，经过 embedding、各层 hidden state、logits、交叉熵，
最后到反向传播、梯度范数和 AdamW 更新。默认给人类阅读，也可用 ``--json`` 查看完整轨迹。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from about_llm.from_scratch.mini_gpt_training_trace import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    run_minigpt_training_trace,
)


def build_parser() -> argparse.ArgumentParser:
    """定义随机种子、学习率和完整 JSON 输出开关。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete machine-readable trace instead of the guided view",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行一次确定性训练步，并选择导览视图或机器可读报告。"""

    # 训练样本包含中文，Windows 下显式指定 UTF-8 可避免输出乱码。
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    # 底层真实执行 forward、loss.backward() 和 optimizer.step()，不是预先写好的数字。
    report = run_minigpt_training_trace(
        seed=args.seed,
        learning_rate=args.learning_rate,
    )
    # JSON 适合核对所有张量统计；导览视图只保留理解训练闭环所需的关键节点。
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_guided_view(report))
    return 0


def _guided_view(report: dict[str, Any]) -> str:
    """把完整训练报告压缩成按计算顺序排列的中文阅读视图。"""

    sample = report["sample"]
    model = report["model"]
    forward = report["forward_before_update"]
    update = report["backward_and_update"]
    lines = [
        "同一个样本现在进入 MiniGPT",
        f"文本: {sample['text']}",
        f"模型输入: {sample['input_ids']}",
        f"监督标签: {sample['supervised_labels']} (-100 不计入 loss)",
        "",
        f"模型: 1 层、{model['model_dim']} 维、{model['num_heads']} 个 attention heads",
        f"Logits shape: {forward['logits_shape']}",
        "",
        "逐位置读取模型对正确目标的初始概率:",
    ]
    for row in forward["positions"]:
        suffix = (
            f"NLL={row['negative_log_probability']:.6f}"
            if row["included_in_loss"]
            else "不计入 loss"
        )
        lines.append(
            f"  位置 {row['position']}: {row['input_piece']} → {row['target_piece']} | "
            f"p={row['target_probability']:.6f} | {suffix}"
        )
    lines.extend(
        [
            "",
            f"三个有效目标的平均 NLL: {forward['mean_nll_from_model']:.6f}",
            f"由逐位置 NLL 复算: {forward['mean_nll_recomputed_from_positions']:.6f}",
            f"这三个目标上的 perplexity: {forward['perplexity_on_three_targets']:.3f}",
            "",
            f"反向传播后的全局梯度 L2: {update['gradient_global_l2']:.6f}",
            f"一步 SGD 后的平均 NLL: {update['mean_nll_after_one_step']:.6f}",
            f"发生变化的参数张量: {update['updated_parameter_tensor_count']} / "
            f"{update['parameter_tensor_count']}",
            "",
            "这只说明同一小样本上的 forward、loss、backward 和更新已经接通。",
            "随机初始化模型的一步 loss 下降。这不代表它已经学会中文或能够泛化。",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
