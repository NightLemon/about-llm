"""Run lightweight source and encoding checks for documentation."""

from __future__ import annotations

import sys

from content_quality import check_encoding, check_ledger, text_files


def main() -> int:
    files = text_files()
    errors = check_encoding(files) + check_ledger()
    if errors:
        print("Content source checks failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"OK: checked UTF-8 in {len(files)} files and validated the source registry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
