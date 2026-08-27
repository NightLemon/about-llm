"""运行仓库从协议层实现的 MCP 2025-11-25 Streamable HTTP 实验。

实验在 loopback 上检查 session、JSON/流式响应和工具调用状态，不经官方 SDK 高层抽象。
它适合与 SDK 版本对照阅读，但不覆盖公网 TLS、反向代理或真实身份提供方。
"""

from __future__ import annotations

import json

from about_llm.agents.mcp_streamable_http import run_streamable_http_control


def main() -> None:
    """执行本地 HTTP 协议控制并打印生命周期报告。"""

    # 底层负责启动与关闭临时服务，本入口只呈现结构化证据。
    report = run_streamable_http_control()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
