from __future__ import annotations

import subprocess
from collections.abc import Generator
from typing import Any

import pytest

DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 120


@pytest.fixture(autouse=True)
def bound_test_subprocesses(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    original_run = subprocess.run

    def run_with_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        kwargs.setdefault("timeout", DEFAULT_SUBPROCESS_TIMEOUT_SECONDS)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_with_timeout)
    yield
