"""追踪一段中文怎样从 UTF-8 字节变成因果语言模型训练样本。

实验先用少量文本训练 Byte-BPE，再依次展示 token、input_ids、右移后的 labels、
attention mask 和 loss mask。它回答“同一串 ID 在输入端和监督端为什么要错开一位”。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from about_llm.from_scratch.language_model_sample import (
    DEFAULT_TEXT,
    DEFAULT_TRAINING_DOCUMENTS,
    build_language_model_sample,
)


def build_parser() -> argparse.ArgumentParser:
    """定义待追踪文本、BPE 训练语料和最小词频等参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to trace")
    parser.add_argument(
        "--training-text",
        action="append",
        help="one Byte-BPE training document; repeat to provide more documents",
    )
    parser.add_argument("--vocab-size", type=int, default=280)
    parser.add_argument("--min-pair-frequency", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """训练 tokenizer、构造一条 LM 样本并输出完整中间状态。"""

    # 显式配置 UTF-8，确保 Windows 终端能直接显示输入文本和 token。
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    # 未提供自定义语料时使用共享小语料；每个 --training-text 都是一篇独立文档。
    training_documents = (
        tuple(args.training_text)
        if args.training_text
        else DEFAULT_TRAINING_DOCUMENTS
    )
    # 底层会依次执行 BPE 学习、编码、next-token 错位和 mask 构造。
    report = build_language_model_sample(
        text=args.text,
        training_documents=training_documents,
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
