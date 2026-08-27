"""从客户端侧压测一个 OpenAI-compatible 流式聊天接口。

脚本按 burst、constant 或 Poisson 到达过程调度有限数量请求，记录每次请求的排队等待、
TTFT、TPOT、端到端延迟和成功/失败类别。它只能描述当前客户端、模型与 workload，
不能直接分解服务端队列、GPU kernel 或给出普适容量结论。
"""

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

# 多条 prompt 轮换使用，避免所有请求都只代表同一种输入长度与主题。
DEFAULT_PROMPTS = (
    "用三句话解释 KV Cache。",
    "比较 RAG 与微调各自适合解决的问题。",
    "给出排查 LLM 服务首 token 延迟升高的步骤。",
)


async def sleep_until(deadline: float) -> None:
    """等待到单调时钟 deadline；事件循环提前唤醒时继续等待剩余时间。"""

    while (remaining := deadline - time.perf_counter()) > 0:
        await asyncio.sleep(remaining)


def load_prompts(path: Path | None) -> list[str]:
    """读取每行一个 prompt 的 UTF-8 文件，未指定时使用内置样例。"""

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
    """发送一次流式请求，并从 SSE 与 usage 计算 token 级时刻。"""

    # perf_counter 是单调时钟，不会受系统时间校准影响，适合测量短时延。
    started_at = time.perf_counter()
    first_token_at: float | None = None
    observed_content = False
    usage: dict[str, Any] = {}
    # temperature=0 降低内容随机性；include_usage 提供真实 completion token 数。
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # aiter_lines 按 SSE 行增量消费，不把完整回答先缓冲到内存。
    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            event = parse_sse_data_line(line)
            if event is None:
                continue
            if event is STREAM_FINISHED:
                break
            # usage 往往只在尾部事件出现，与内容 chunk 分开保存。
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                content = choices[0].get("delta", {}).get("content")
                if content:
                    # TTFT 取第一段非空 content 到达时间，而不是 HTTP header 到达时间。
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    observed_content = True
    completed_at = time.perf_counter()
    if first_token_at is None or not observed_content:
        raise RuntimeError("stream completed without a content token")
    # SSE chunk 可能包含半个或多个 token，不能用 chunk 数冒充 output token 数。
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
    """按指定到达过程并发执行请求，汇总完整 attempt 级结果。"""

    # schedule 先一次性生成，使实际 dispatch 延迟能与计划到达时刻分开计算。
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
    # semaphore 限制真正进入 HTTP 调用的并发数；等待时间计入 client queue。
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        timeout=timeout,
    ) as client:

        async def limited(index: int, offset_seconds: float) -> InferenceAttempt:
            """在计划时刻投放请求，并把异常归类为稳定 outcome。"""

            request_id = f"request-{index:06d}"
            offered_at = benchmark_started_at + offset_seconds
            await sleep_until(offered_at)
            # 到达后仍可能等待并发槽位，这正是 offered load 超过容量时要观察的现象。
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
                # 将网络、HTTP 与协议错误分开，避免失败请求被延迟统计静默丢弃。
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
        # 本工具面向有限实验，所以预先创建全部 request coroutine，而不是无限流量发生器。
        attempts = await asyncio.gather(
            *(
                limited(index, offset_seconds)
                for index, offset_seconds in enumerate(schedule.offsets_seconds)
            )
        )
        completed_at = time.perf_counter()

    # summary 只用成功 measurement 计算分位数，同时保留所有失败 attempt 计数。
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
    """解析服务、负载、并发和鉴权参数，并检查到达过程组合。"""

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
