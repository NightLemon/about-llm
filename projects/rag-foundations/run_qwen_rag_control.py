"""运行固定 Qwen checkpoint 的 retrieval-to-generation 基线实验。

它按 manifest 执行真实检索与 ``generate``，原样记录模型是否给出合规引用、证据不足时是否
拒答。基线不会事后修补输出，因而可以与 guarded policy 的阻断效果公平比较。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)
from about_llm.rag.transformers_control import (
    load_rag_transformers_control_spec,
    run_rag_transformers_control,
)

PROJECT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT / "qwen2.5-0.5b-rag.control.json"
DEFAULT_CHECKPOINT_MANIFEST = (
    PROJECT.parent
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)


def main() -> None:
    """加载固定模型与 RAG 控制规范，运行并保存观察报告。"""

    # 报告保留中文 prompt 和答案，统一 stdout 编码。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=(
            "Execute the fixed Qwen2.5-0.5B retrieval-to-generation control; "
            "record observed citation and abstention failures without output repair"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the reviewed checkpoint files to exist in the local cache",
    )
    args = parser.parse_args()
    # checkpoint 与 RAG manifest 分别固定模型身份和请求/检索配置。
    checkpoint_spec = load_checkpoint_control_spec(args.checkpoint_manifest)
    spec = load_rag_transformers_control_spec(args.manifest)
    report = run_rag_transformers_control(
        spec,
        checkpoint_spec=checkpoint_spec,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
