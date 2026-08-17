from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import nbformat
import pytest

from scripts import execute_notebooks


def _write_notebook(path: Path, *, kernel_name: str | None = None) -> None:
    notebook = nbformat.v4.new_notebook()
    if kernel_name is not None:
        notebook.metadata["kernelspec"] = {
            "display_name": kernel_name,
            "language": "python",
            "name": kernel_name,
        }
    nbformat.write(notebook, path)


def test_execute_defaults_kernel_and_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "lesson.ipynb"
    _write_notebook(path)
    client = Mock()
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(execute_notebooks, "NotebookClient", client_factory)

    execute_notebooks.execute(path, timeout=17)

    assert client_factory.call_args.kwargs == {
        "timeout": 17,
        "kernel_name": "python3",
        "resources": {"metadata": {"path": str(tmp_path.resolve())}},
    }
    client.execute.assert_called_once_with()


def test_execute_honors_kernel_and_working_directory_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "lesson.ipynb"
    working_directory = tmp_path / "workspace"
    working_directory.mkdir()
    _write_notebook(path, kernel_name="teaching-kernel")
    client_factory = Mock(return_value=Mock())
    monkeypatch.setattr(execute_notebooks, "NotebookClient", client_factory)

    execute_notebooks.execute(path, timeout=9, working_directory=working_directory)

    assert client_factory.call_args.kwargs["kernel_name"] == "teaching-kernel"
    assert client_factory.call_args.kwargs["resources"] == {
        "metadata": {"path": str(working_directory.resolve())}
    }
