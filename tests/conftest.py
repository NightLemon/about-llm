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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Reject tests whose evidence or scheduling purpose is implicit.

    ``extended`` is a scheduling decision, not an evidence category.  Every
    extended test must expose the concrete runtime reason through
    ``integration`` or ``slow`` so reviewers can revisit the decision.
    """

    evidence_markers = ("formula", "contract", "security", "smoke")
    missing_evidence = [
        item.nodeid
        for item in items
        if not any(item.get_closest_marker(marker) for marker in evidence_markers)
    ]
    if missing_evidence:
        formatted = "\n".join(f"- {nodeid}" for nodeid in missing_evidence)
        raise pytest.UsageError(
            "tests must declare formula, contract, security, or smoke evidence:\n"
            + formatted
        )

    invalid = [
        item.nodeid
        for item in items
        if item.get_closest_marker("extended") is not None
        and item.get_closest_marker("integration") is None
        and item.get_closest_marker("slow") is None
    ]
    if invalid:
        formatted = "\n".join(f"- {nodeid}" for nodeid in invalid)
        raise pytest.UsageError(
            "extended tests must also be marked integration or slow:\n" + formatted
        )
