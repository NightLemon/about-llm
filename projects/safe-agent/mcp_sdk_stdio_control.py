"""通过真实 stdio 子进程运行官方 MCP SDK client/server 实验。

客户端会拉起独立 server 进程，经 stdin/stdout 完成初始化、工具发现和调用。与内存版本相比，
它额外覆盖进程边界、帧传输和关闭顺序，但仍不访问网络。
"""

from __future__ import annotations

# 共享 main 同时实现 client 与子进程 server 模式，本入口默认启动完整控制流程。
from about_llm.agents.mcp_sdk_stdio import main

if __name__ == "__main__":
    raise SystemExit(main())
