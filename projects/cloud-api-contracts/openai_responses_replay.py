"""离线重放一份 OpenAI Responses 风格的 typed-event JSONL。

输入文件中的每行代表一个流式事件。重放器检查生命周期顺序，累积文本和 function call
参数，并生成最终 receipt。它不调用 SDK 或网络，只研究事件协议本身。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from about_llm.integrations.openai_responses_replay import (
    replay_response_event_file,
)


def build_parser() -> argparse.ArgumentParser:
    """读取必须显式提供的事件 JSONL 路径。"""

    parser = argparse.ArgumentParser(
        description="Replay an SDK-shaped OpenAI Responses event JSONL file"
    )
    parser.add_argument("--events", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """重放事件文件；格式或状态机错误以退出码 2 返回。"""

    args = build_parser().parse_args(argv)
    try:
        # parser 会逐事件推进状态机，并在结束时检查是否得到完整响应。
        receipt = replay_response_event_file(args.events)
    except (OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(
        receipt.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    sys.stdout.buffer.write((rendered + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
