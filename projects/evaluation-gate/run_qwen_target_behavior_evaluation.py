"""运行或离线复核固定 Qwen checkpoint 的行为评测套件。

实际运行路径加载 manifest 指定的模型并生成答案；``--verify`` 路径不加载权重，只复核已有
报告的 case、指标、哈希和套件身份。两条路径都绑定同一份评测 suite。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from about_llm.evaluation.target_qwen_control import (
    load_target_qwen_evaluation_spec,
    run_target_qwen_evaluation_control,
    verify_recorded_target_qwen_evaluation_report,
)
from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]
DEFAULT_SUITE = PROJECT / "target-qwen-behavior-suite.control.json"
DEFAULT_CHECKPOINT_MANIFEST = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)


def main() -> None:
    """选择真实 checkpoint 运行或 recorded report 离线验证。"""

    # 评测输出可能含中文，Windows 下显式设置 UTF-8 并对编码错误严格失败。
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser(
        description="Run or verify the fixed target-Qwen behavior evaluation"
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="verify a recorded report without loading model weights",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the reviewed checkpoint files in the local Hugging Face cache",
    )
    args = parser.parse_args()
    # suite 固定 prompts、预期行为和指标门槛，是两种模式共同的评测定义。
    suite = load_target_qwen_evaluation_spec(args.suite)
    if args.verify is not None:
        # CI 可走此分支，在无权重环境核对报告没有漂移或被篡改。
        report = verify_recorded_target_qwen_evaluation_report(args.verify, suite)
    else:
        # 真实运行先验证 checkpoint 身份，再按 suite 逐条生成与打分。
        checkpoint = load_checkpoint_control_spec(args.checkpoint_manifest)
        report = run_target_qwen_evaluation_control(
            checkpoint, suite, local_files_only=args.local_files_only
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
