"""运行仓库从协议层实现的 MCP 2025-11-25 stdio 互操作实验。

它不依赖官方 SDK 的高层 client/server 封装，而是直接验证 JSON-RPC 帧、initialize、
tools/list、tools/call 和错误响应，帮助区分协议语义与某个 SDK 的便利 API。
"""

from __future__ import annotations

import sys

from about_llm.agents.mcp_stdio import run_stdio_control
from about_llm.llmops import canonical_json_bytes


def main() -> int:
    """启动本地 stdio 控制并以 canonical JSON 输出报告。"""

    # canonical bytes 固定 key 顺序和编码，便于对协议 artifact 做哈希或逐字节比较。
    report = run_stdio_control()
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
