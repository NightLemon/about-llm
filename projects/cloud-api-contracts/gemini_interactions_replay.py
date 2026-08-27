"""离线重放仓库记录的 Gemini Interactions SSE 工具调用事件。

程序逐个解析固定事件流，重建 interaction、function call 参数和完成状态，不联系 Gemini。
它用于学习流式协议生命周期，不代表当前线上 API 的实时响应。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接从项目目录运行脚本，而无需先把本仓库安装成 package。
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from about_llm.integrations.gemini_interactions_replay import (  # noqa: E402
    load_gemini_interactions_sse,
)

DEFAULT_EVENTS = Path(__file__).with_name(
    "gemini-interactions-function-call.example.sse"
)


def main() -> int:
    """解析 SSE 记录，并打印或写入重建后的结构化生命周期。"""

    parser = argparse.ArgumentParser(
        description="Replay a fixed Gemini Interactions SSE lifecycle"
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # loader 会验证事件顺序、ID 关联和结束状态，而不是简单拼接 data 行。
    payload = load_gemini_interactions_sse(args.events).to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.buffer.write(rendered.encode("utf-8"))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
