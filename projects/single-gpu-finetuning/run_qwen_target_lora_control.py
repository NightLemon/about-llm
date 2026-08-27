"""运行或离线复核固定 Qwen2.5 checkpoint 的一步 LoRA 更新。

实验验证 base checkpoint 与已发布 adapter bundle，执行真实前向/反向并检查只有 LoRA 参数改变；
``--verify`` 则在不加载模型的情况下核对仓库记录报告和 artifact 哈希。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from about_llm.finetuning.target_lora_control import (
    load_recorded_target_lora_report,
    load_target_lora_control_spec,
    run_target_lora_control,
)
from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)

PROJECT = Path(__file__).resolve().parent
REPOSITORY = PROJECT.parents[1]
DEFAULT_CONTROL = PROJECT / "qwen2.5-0.5b-lora.control.json"
DEFAULT_CHECKPOINT_CONTROL = (
    REPOSITORY
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)
DEFAULT_CHECKPOINT_REPORT = DEFAULT_CHECKPOINT_CONTROL.with_name(
    "qwen2.5-0.5b-instruct.recorded-report.json"
)
DEFAULT_ARTIFACT = PROJECT / "target-adapters" / "qwen2.5-0.5b-instruct-step1"


def _write_json_new(path: Path, value: object) -> None:
    """新建并持久化 UTF-8 JSON 报告，不覆盖现有文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    """定义 checkpoint、LoRA 控制、adapter 目录和运行/验证模式。"""

    parser = argparse.ArgumentParser(
        description="Run or verify the reviewed Qwen2.5-0.5B CPU FP32 LoRA control"
    )
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument(
        "--checkpoint-control", type=Path, default=DEFAULT_CHECKPOINT_CONTROL
    )
    parser.add_argument(
        "--checkpoint-report", type=Path, default=DEFAULT_CHECKPOINT_REPORT
    )
    parser.add_argument("--artifact-directory", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the reviewed checkpoint files to exist in the local Hugging Face cache",
    )
    return parser.parse_args()


def main() -> None:
    """加载共同身份约束，再选择一步 LoRA 运行或报告复核。"""

    args = parse_args()
    checkpoint_spec = load_checkpoint_control_spec(args.checkpoint_control)
    spec = load_target_lora_control_spec(
        args.control, checkpoint_spec=checkpoint_spec
    )
    if args.verify is not None:
        # 只读取 recorded report 和 adapter 文件，不执行模型训练。
        report = load_recorded_target_lora_report(
            args.verify,
            spec=spec,
            checkpoint_spec=checkpoint_spec,
            checkpoint_report_path=args.checkpoint_report,
            artifact_directory=args.artifact_directory,
        )
    else:
        # 真实路径加载固定 base 与 adapter，执行一步梯度更新并记录可训练参数变化。
        report = run_target_lora_control(
            spec,
            checkpoint_spec=checkpoint_spec,
            checkpoint_report_path=args.checkpoint_report,
            artifact_directory=args.artifact_directory,
            local_files_only=args.local_files_only,
        )
        if args.output_report is not None:
            _write_json_new(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
