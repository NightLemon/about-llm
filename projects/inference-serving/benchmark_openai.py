"""Benchmark an OpenAI-compatible streaming chat endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from about_llm.inference import InferenceMeasurement, summarize_measurements
from about_llm.inference.sse import STREAM_FINISHED, parse_sse_data_line

DEFAULT_PROMPTS = (
    "用三句话解释 KV Cache。",
    "比较 RAG 与微调各自适合解决的问题。",
    "给出排查 LLM 服务首 token 延迟升高的步骤。",
)


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
    api_key = os.getenv(args.api_key_env, "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    timeout = httpx.Timeout(args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        timeout=timeout,
    ) as client:

        async def limited(index: int) -> InferenceMeasurement:
            async with semaphore:
                return await run_request(
                    client,
                    model=args.model,
                    prompt=prompts[index % len(prompts)],
                    max_tokens=args.max_tokens,
                )

        started_at = time.perf_counter()
        measurements = await asyncio.gather(*(limited(index) for index in range(args.requests)))
        completed_at = time.perf_counter()

    summary = summarize_measurements(
        measurements,
        benchmark_started_at=started_at,
        benchmark_completed_at=completed_at,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or args.max_tokens <= 0:
        parser.error("requests, concurrency and max-tokens must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(benchmark(parse_args()))
