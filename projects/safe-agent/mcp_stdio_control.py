"""Run the local MCP 2025-11-25 stdio interoperability control."""

from __future__ import annotations

import sys

from about_llm.agents.mcp_stdio import run_stdio_control
from about_llm.llmops import canonical_json_bytes


def main() -> int:
    report = run_stdio_control()
    sys.stdout.buffer.write(canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
