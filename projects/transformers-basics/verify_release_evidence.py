from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.model_release_evidence import verify_model_release_evidence

DEFAULT_MANIFEST = Path(__file__).with_name("release-evidence") / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify hash-pinned Llama/Qwen/DeepSeek release evidence without "
            "loading model weights"
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--verify-upstream",
        action="store_true",
        help="download immutable public artifacts and verify exact bytes",
    )
    args = parser.parse_args()
    report = verify_model_release_evidence(
        args.manifest,
        verify_upstream=args.verify_upstream,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
