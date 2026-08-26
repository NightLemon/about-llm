"""Replay the repository's fixed Gemini Interactions SSE example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from about_llm.integrations.gemini_interactions_replay import (  # noqa: E402
    load_gemini_interactions_sse,
)

DEFAULT_EVENTS = Path(__file__).with_name(
    "gemini-interactions-function-call.example.sse"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a fixed Gemini Interactions SSE lifecycle"
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = load_gemini_interactions_sse(args.events).to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.buffer.write(rendered.encode("utf-8"))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
