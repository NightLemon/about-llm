"""按 manifest 校验并运行固定的 Qwen2.5-0.5B-Instruct checkpoint。

入口先核对仓库、revision、文件哈希和运行参数，再用 Transformers 在 CPU/FP32 下加载模型。
这样生成结果对应一个明确的模型快照，而不是名称相同但内容已经变化的浮动版本。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
    run_checkpoint_control,
)

DEFAULT_MANIFEST = (
    Path(__file__).with_name("target-checkpoints")
    / "qwen2.5-0.5b-instruct.control.json"
)


def main() -> None:
    """读取 checkpoint 约束，验证本地或缓存文件并执行目标推理。"""

    parser = argparse.ArgumentParser(
        description=(
            "Verify and execute the reviewed Qwen2.5-0.5B-Instruct CPU FP32 "
            "checkpoint control"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require all reviewed files to already exist in the Hugging Face cache",
    )
    args = parser.parse_args()
    # manifest 是实验身份的来源；模型加载和输出检查都以它为准。
    spec = load_checkpoint_control_spec(args.manifest)
    report = run_checkpoint_control(spec, local_files_only=args.local_files_only)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
