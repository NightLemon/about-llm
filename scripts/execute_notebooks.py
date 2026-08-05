"""Execute notebooks in memory so committed sources remain output-free."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def execute(path: Path, timeout: int) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=notebook.metadata.kernelspec.name,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    print(f"OK: {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="notebooks/*.ipynb")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    paths = sorted(ROOT.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No notebooks matched {args.pattern}")
    for path in paths:
        execute(path, args.timeout)


if __name__ == "__main__":
    main()
