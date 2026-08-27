"""在固定 Qwen2.5 checkpoint 上运行 activation patching 对照实验。

与随机 MiniGPT 示例相比，这个入口会先按 manifest 验证真实 checkpoint，再在指定层和位置
替换残差激活，观察目标分数差的变化。报告仍只支持该 prompt、指标和版本下的局部结论。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.integrations.transformers_activation_patching_control import (
    run_target_activation_patching_control,
)
from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)

DEFAULT_CHECKPOINT_MANIFEST = (
    Path(__file__).with_name("target-checkpoints")
    / "qwen2.5-0.5b-instruct.control.json"
)


def main() -> None:
    """加载模型约束并生成目标 checkpoint 的 patching 报告。"""

    parser = argparse.ArgumentParser(
        description=(
            "Execute the reviewed Qwen2.5-0.5B-Instruct CPU FP32 "
            "activation-patching control"
        )
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=DEFAULT_CHECKPOINT_MANIFEST,
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require every selected checkpoint file to exist in the local HF cache",
    )
    args = parser.parse_args()
    # 先验证 checkpoint 身份，再执行 hook，避免把结果归因给错误的模型版本。
    spec = load_checkpoint_control_spec(args.checkpoint_manifest)
    report = run_target_activation_patching_control(
        spec, local_files_only=args.local_files_only
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
