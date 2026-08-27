"""训练并检查仓库从零实现的确定性 byte-level BPE。

实验把文本先转为 UTF-8 字节，再反复合并最常见的相邻 token。报告展示学习到的 merge、
编码结果和 decode 回原文的结果，用来建立“词表是如何从语料学出来的”直觉。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from about_llm.from_scratch import ByteBPETokenizer

# 默认语料同时含英文重复片段和中文 UTF-8 字节，便于观察两类合并。
DEFAULT_DOCUMENTS = (
    "low lower lowest",
    "newer wider lower",
    "你好你好，语言模型",  # noqa: RUF001
    "token tokenization tokenizer",
)
EVIDENCE_BOUNDARY = (
    "This CPU demo verifies deterministic merge learning and UTF-8 round trips for "
    "the supplied strings. It has no normalization, pre-tokenizer, special tokens, "
    "offset map, checkpoint compatibility, throughput claim, or target-corpus evidence."
)


def build_parser() -> argparse.ArgumentParser:
    """定义训练文档、编码样本、词表上限和最小 pair 频次。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        action="append",
        help="one independent training document; repeat to supply a corpus",
    )
    parser.add_argument(
        "--sample",
        action="append",
        help="text to encode after training; defaults to the training documents",
    )
    parser.add_argument("--vocab-size", type=int, default=280)
    parser.add_argument("--min-pair-frequency", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """学习 BPE merges，并对指定样本执行 encode/decode 往返。"""

    # 中文语料和 token 展示都依赖 UTF-8 终端输出。
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    # --text 可重复传入，每个值都作为独立文档，合并统计不会跨越文档边界。
    documents = tuple(args.text) if args.text else DEFAULT_DOCUMENTS
    samples = tuple(args.sample) if args.sample else documents
    # train 只从提供的语料学习 merge；没有预训练词表或外部 tokenizer 参与。
    tokenizer = ByteBPETokenizer.train(
        documents,
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
    )
    # 每条 merge 同时展示 token ID、原始字节和尽力解码的 UTF-8 预览。
    merges = [
        {
            "rank": rank,
            "left_id": left,
            "right_id": right,
            "new_id": tokenizer.base_vocab_size + rank,
            "bytes_hex": tokenizer.token_bytes(tokenizer.base_vocab_size + rank).hex(),
            "utf8_preview": tokenizer.token_bytes(
                tokenizer.base_vocab_size + rank
            ).decode("utf-8", errors="replace"),
        }
        for rank, (left, right) in enumerate(tokenizer.merges)
    ]
    # 对每个样本做一次往返，直接检查字节级 tokenizer 是否无损恢复原文。
    encoded_samples = []
    for text in samples:
        token_ids = tokenizer.encode(text)
        encoded_samples.append(
            {
                "text": text,
                "utf8_bytes": len(text.encode("utf-8")),
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "round_trip": tokenizer.decode(token_ids) == text,
            }
        )
    print(
        json.dumps(
            {
                "implementation": "about-llm.byte-bpe-reference.v1",
                "base_vocabulary": "raw byte ids 0..255",
                "normalization": "none",
                "pre_tokenization": "none",
                "requested_vocab_size": args.vocab_size,
                "actual_vocab_size": tokenizer.vocab_size,
                "min_pair_frequency": args.min_pair_frequency,
                "document_count": len(documents),
                "merges": merges,
                "samples": encoded_samples,
                "evidence_boundary": EVIDENCE_BOUNDARY,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
