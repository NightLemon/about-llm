"""通过官方 MCP SDK 运行 Streamable HTTP loopback 实验。

服务只绑定本机，客户端执行初始化、session 管理、工具发现与调用，再检查协议版本和响应。
这验证当前 SDK 的传输互操作，不代表公网身份、TLS 或生产代理配置。
"""

from __future__ import annotations

import json

from about_llm.agents.mcp_sdk_streamable_http import run_mcp_sdk_http_control


def main() -> int:
    """执行 SDK HTTP 控制并输出严格 JSON 报告。"""

    # allow_nan=False 确保协议报告不会夹带非标准 JSON 数值。
    print(
        json.dumps(
            run_mcp_sdk_http_control(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
