"""Execute notebooks in memory so committed sources remain output-free."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL_NAME = "python3"


def execute(path: Path, timeout: int, working_directory: Path | None = None) -> None:
    notebook = nbformat.read(path, as_version=4)
    kernelspec = notebook.metadata.get("kernelspec", {})
    kernel_name = kernelspec.get("name") if isinstance(kernelspec, dict) else None
    if not isinstance(kernel_name, str) or not kernel_name:
        kernel_name = DEFAULT_KERNEL_NAME
    execution_directory = (working_directory or path.parent).resolve()
    if not execution_directory.is_dir():
        raise ValueError(f"Notebook working directory does not exist: {execution_directory}")
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(execution_directory)}},
    )
    client.execute()
    display_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"OK: {display_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="notebooks/*.ipynb")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--working-directory",
        type=Path,
        help="Override the default working directory (each notebook's parent directory)",
    )
    args = parser.parse_args()
    paths = sorted(ROOT.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No notebooks matched {args.pattern}")
    for path in paths:
        execute(path, args.timeout, args.working_directory)


if __name__ == "__main__":
    main()
