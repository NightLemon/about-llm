"""Run one readable MiniGPT training step on the shared Chinese sample."""

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
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    report = run_minigpt_training_trace(
        seed=args.seed,
        learning_rate=args.learning_rate,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_guided_view(report))
    return 0


def _guided_view(report: dict[str, Any]) -> str:
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
