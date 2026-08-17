"""Run the local A2A 1.0 loopback interoperability control."""

from __future__ import annotations

import sys

from about_llm.agents.a2a_loopback import main

if __name__ == "__main__":
    raise SystemExit(main(["control", *sys.argv[1:]]))
