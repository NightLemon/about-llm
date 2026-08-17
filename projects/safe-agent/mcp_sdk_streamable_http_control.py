"""Run the official MCP SDK Streamable HTTP loopback control."""

from __future__ import annotations

import json

from about_llm.agents.mcp_sdk_streamable_http import run_mcp_sdk_http_control


def main() -> int:
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
