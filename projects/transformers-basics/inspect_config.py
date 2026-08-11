from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.model_config import (
    estimate_standard_kv_cache,
    inspect_decoder_config,
    load_model_config_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a local decoder config without loading model weights"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--tokens", type=int, action="append")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--element-bytes",
        type=int,
        default=2,
        help="ideal bytes per K/V element; excludes scale and allocator metadata",
    )
    args = parser.parse_args()
    inspection = inspect_decoder_config(load_model_config_json(args.config))
    token_counts = (1024, 4096, 8192) if args.tokens is None else args.tokens
    estimates: list[dict[str, object]] = []
    if inspection.standard_kv_layout.applicable:
        estimates = [
            estimate_standard_kv_cache(
                inspection,
                token_count=token_count,
                batch_size=args.batch_size,
                element_bytes=args.element_bytes,
            ).to_dict()
            for token_count in token_counts
        ]
    payload = {
        "source_path": str(args.config),
        "inspection": inspection.to_dict(),
        "standard_kv_estimates": estimates,
        "estimate_refused": not inspection.standard_kv_layout.applicable,
        "estimate_refusal_reason": (
            None
            if inspection.standard_kv_layout.applicable
            else inspection.standard_kv_layout.reason
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
