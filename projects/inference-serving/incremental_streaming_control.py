"""Run or verify the deterministic incremental-streaming disconnect control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from about_llm.inference.incremental_streaming_control import (
    load_and_verify_incremental_streaming_report,
    run_incremental_streaming_control,
)

PROJECT = Path(__file__).resolve().parent
DEFAULT_RECORDED_REPORT = PROJECT / "incremental-streaming.recorded-report.json"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        type=Path,
        metavar="RECORDED_REPORT",
        help="verify a recorded report without starting a server",
    )
    args = parser.parse_args()
    report = (
        run_incremental_streaming_control()
        if args.verify is None
        else load_and_verify_incremental_streaming_report(args.verify)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
