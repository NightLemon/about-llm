"""Run the official MCP SDK client/server over a real stdio subprocess."""

from __future__ import annotations

from about_llm.agents.mcp_sdk_stdio import main

if __name__ == "__main__":
    raise SystemExit(main())
