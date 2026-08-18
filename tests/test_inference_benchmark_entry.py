from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "inference-serving" / "benchmark_openai.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("about_llm_benchmark_openai", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.smoke
def test_benchmark_cli_accepts_reproducible_open_loop_configuration() -> None:
    module = _load_script()

    constant = module.parse_args(
        [
            "--model",
            "fixture",
            "--arrival-process",
            "constant",
            "--request-rate",
            "4",
        ]
    )
    poisson = module.parse_args(
        [
            "--model",
            "fixture",
            "--arrival-process",
            "poisson",
            "--request-rate",
            "4",
        ]
    )

    assert constant.request_rate == 4
    assert constant.arrival_seed is None
    assert poisson.request_rate == 4
    assert poisson.arrival_seed == 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model", "fixture", "--arrival-process", "constant"],
        ["--model", "fixture", "--request-rate", "2"],
        [
            "--model",
            "fixture",
            "--arrival-process",
            "constant",
            "--request-rate",
            "2",
            "--arrival-seed",
            "7",
        ],
    ],
)
def test_benchmark_cli_rejects_ignored_or_incomplete_arrival_configuration(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        _load_script().parse_args(arguments)

    assert error.value.code == 2


def test_sleep_until_retries_after_an_early_wakeup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    ticks = iter((0.0, 0.5, 1.0))
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(module.asyncio, "sleep", fake_sleep)

    asyncio.run(module.sleep_until(1.0))

    assert delays == [1.0, 0.5]


@pytest.mark.smoke
def test_benchmark_applies_open_loop_offsets_before_semaphore_queue(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()

    class FakeAsyncClient:
        def __init__(self, **_: object) -> None:
            pass

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    async def fake_run_request(*_: object, **__: object) -> object:
        started_at = module.time.perf_counter()
        await asyncio.sleep(0.01)
        completed_at = module.time.perf_counter()
        return module.InferenceMeasurement(
            prompt_tokens=1,
            output_tokens=1,
            started_at=started_at,
            first_token_at=completed_at,
            completed_at=completed_at,
        )

    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(module, "run_request", fake_run_request)
    args = module.parse_args(
        [
            "--model",
            "fixture",
            "--requests",
            "3",
            "--concurrency",
            "1",
            "--arrival-process",
            "constant",
            "--request-rate",
            "1000",
        ]
    )

    asyncio.run(module.benchmark(args))
    payload = json.loads(capsys.readouterr().out)

    assert payload["workload"]["arrival_schedule"]["offsets_seconds"] == [
        0.0,
        0.001,
        0.002,
    ]
    assert payload["summary"]["successful_requests"] == 3
    assert payload["summary"]["client_queue_p95_seconds"] > 0
    offered = [attempt["offered_at"] for attempt in payload["attempts"]]
    assert offered[1] - offered[0] == pytest.approx(0.001)
    assert offered[2] - offered[1] == pytest.approx(0.001)
