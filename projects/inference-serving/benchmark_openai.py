"""Benchmark an OpenAI-compatible streaming chat endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from about_llm.inference import (
    ArrivalProcess,
    InferenceAttempt,
    InferenceMeasurement,
    RequestOutcome,
    build_arrival_schedule,
    classify_http_failure,
    summarize_attempts,
)
from about_llm.inference.sse import STREAM_FINISHED, parse_sse_data_line

DEFAULT_PROMPTS = (
    "用三句话解释 KV Cache。",
    "比较 RAG 与微调各自适合解决的问题。",
    "给出排查 LLM 服务首 token 延迟升高的步骤。",
)


async def sleep_until(deadline: float) -> None:
    """Wait until a monotonic deadline, tolerating an early event-loop wakeup."""

    while (remaining := deadline - time.perf_counter()) > 0:
        await asyncio.sleep(remaining)


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    prompts = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not prompts:
        raise ValueError("prompt file must contain at least one non-empty line")
    return prompts


async def run_request(
    client: httpx.AsyncClient,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
) -> InferenceMeasurement:
    started_at = time.perf_counter()
    first_token_at: float | None = None
    observed_content = False
    usage: dict[str, Any] = {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            event = parse_sse_data_line(line)
            if event is None:
                continue
            if event is STREAM_FINISHED:
                break
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                content = choices[0].get("delta", {}).get("content")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    observed_content = True
    completed_at = time.perf_counter()
    if first_token_at is None or not observed_content:
        raise RuntimeError("stream completed without a content token")
    if usage.get("completion_tokens") is None:
        raise RuntimeError(
            "stream did not report completion_tokens; SSE chunks are not tokens, "
            "so TPOT cannot be computed accurately"
        )
    output_tokens = int(usage["completion_tokens"])
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    return InferenceMeasurement(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        started_at=started_at,
        first_token_at=first_token_at,
        completed_at=completed_at,
    )


async def benchmark(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts)
    schedule = build_arrival_schedule(
        args.requests,
        process=args.arrival_process,
        requests_per_second=args.request_rate,
        seed=args.arrival_seed if args.arrival_seed is not None else 0,
    )
    api_key = os.getenv(args.api_key_env, "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    timeout = httpx.Timeout(args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        timeout=timeout,
    ) as client:

        async def limited(index: int, offset_seconds: float) -> InferenceAttempt:
            request_id = f"request-{index:06d}"
            offered_at = benchmark_started_at + offset_seconds
            await sleep_until(offered_at)
            async with semaphore:
                dispatch_started_at = time.perf_counter()
                try:
                    measurement = await run_request(
                        client,
                        model=args.model,
                        prompt=prompts[index % len(prompts)],
                        max_tokens=args.max_tokens,
                    )
                    return InferenceAttempt.from_measurement(
                        request_id, measurement, offered_at=offered_at
                    )
                except httpx.TimeoutException:
                    outcome = RequestOutcome.TIMEOUT
                except httpx.HTTPStatusError as error:
                    outcome = classify_http_failure(error.response.status_code)
                except httpx.HTTPError:
                    outcome = RequestOutcome.CLIENT_ERROR
                except (KeyError, RuntimeError, TypeError, ValueError):
                    outcome = RequestOutcome.PROTOCOL_ERROR
                return InferenceAttempt(
                    request_id=request_id,
                    outcome=outcome,
                    started_at=dispatch_started_at,
                    completed_at=time.perf_counter(),
                    offered_at=offered_at,
                )

        benchmark_started_at = time.perf_counter()
        attempts = await asyncio.gather(
            *(
                limited(index, offset_seconds)
                for index, offset_seconds in enumerate(schedule.offsets_seconds)
            )
        )
        completed_at = time.perf_counter()

    summary = summarize_attempts(
        attempts,
        benchmark_started_at=benchmark_started_at,
        benchmark_completed_at=completed_at,
    )
    payload = {
        "workload": {
            "arrival_schedule": schedule.to_dict(),
            "concurrency_limit": args.concurrency,
            "offered_at_semantics": (
                "benchmark_started_at + scheduled offset; local event-loop and semaphore "
                "delay remain visible as client queue"
            ),
            "finite_task_boundary": (
                "all request coroutines are materialized for this finite --requests run; "
                "this is not an unbounded production traffic generator"
            ),
        },
        "summary": asdict(summary),
        "attempts": [asdict(attempt) for attempt in attempts],
        "evidence_boundary": (
            "Client-side finite-schedule measurements for this model/runtime/workload only. "
            "Scheduled open-loop arrivals do not prove the load generator kept pace, reveal "
            "server-side queue time, or establish general GPU capacity or production SLOs."
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--arrival-process",
        choices=[item.value for item in ArrivalProcess],
        default=ArrivalProcess.BURST.value,
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        help="nominal requests/second; required for constant or Poisson arrivals",
    )
    parser.add_argument(
        "--arrival-seed",
        type=int,
        help="Poisson inter-arrival seed; defaults to 0",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args(argv)
    if args.requests <= 0 or args.concurrency <= 0 or args.max_tokens <= 0:
        parser.error("requests, concurrency and max-tokens must be positive")
    if args.arrival_process == ArrivalProcess.BURST.value:
        if args.request_rate is not None or args.arrival_seed is not None:
            parser.error("burst arrivals do not use --request-rate or --arrival-seed")
    else:
        if args.request_rate is None:
            parser.error("constant/poisson arrivals require --request-rate")
        if args.arrival_process == ArrivalProcess.CONSTANT.value:
            if args.arrival_seed is not None:
                parser.error("constant arrivals do not use --arrival-seed")
        elif args.arrival_seed is None:
            args.arrival_seed = 0
    return args


if __name__ == "__main__":
    asyncio.run(benchmark(parse_args()))
