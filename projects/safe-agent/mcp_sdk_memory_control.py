"""通过官方 MCP SDK 的内存 transport 运行 client/server 互操作实验。

内存模式不创建子进程和网络连接，适合先观察 initialize、list_tools 与 call_tool 生命周期。
实际协议实现位于 package 中，本文件只是读者运行入口。
"""

from __future__ import annotations

# 直接调用共享实现，避免教学入口复制协议状态机。
from about_llm.agents.mcp_sdk_memory import main

if __name__ == "__main__":
    raise SystemExit(main())
