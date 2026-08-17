"""Reject generated documentation sites that were added to Git."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED_SITE_ROOTS = (".review-site", ".windows-site", "site")


def forbidden_generated_paths(paths: Iterable[str]) -> list[str]:
    """Return tracked paths that belong to a generated site tree."""

    prefixes = tuple(f"{root}/" for root in GENERATED_SITE_ROOTS)
    return sorted(
        path
        for raw_path in paths
        if (path := raw_path.replace("\\", "/"))
        and (path in GENERATED_SITE_ROOTS or path.startswith(prefixes))
    )


def tracked_files(root: Path = ROOT) -> list[str]:
    """Read the repository index instead of rejecting harmless ignored builds."""

    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="strict").split("\0")


def main() -> int:
    generated = forbidden_generated_paths(tracked_files())
    if generated:
        print("Repository hygiene check failed: generated site files are tracked")
        print("\n".join(f"- {path}" for path in generated[:20]))
        if len(generated) > 20:
            print(f"- ... and {len(generated) - 20} more")
        return 1
    print("OK: no generated documentation site files are tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
