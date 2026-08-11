from __future__ import annotations

import argparse
import json
from pathlib import Path

from about_llm.generation_contract import (
    inspect_generation_protocol_document,
    load_generation_protocol_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare explicit tokenizer/model/generation special-token snapshots "
            "without inferring effective runtime defaults"
        )
    )
    parser.add_argument("protocol", type=Path)
    args = parser.parse_args()
    inspection = inspect_generation_protocol_document(
        load_generation_protocol_json(args.protocol)
    )
    print(json.dumps(inspection.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
