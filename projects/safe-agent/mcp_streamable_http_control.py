"""Run the local MCP 2025-11-25 Streamable HTTP loopback control."""

from __future__ import annotations

import json

from about_llm.agents.mcp_streamable_http import run_streamable_http_control


def main() -> None:
    report = run_streamable_http_control()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
