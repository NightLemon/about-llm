"""Serve the persistent deterministic RAG baseline with demo bearer auth."""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path

import uvicorn

from about_llm.rag.service import (
    AuthContext,
    PersistentExtractiveRAGService,
    RAGServiceConfig,
    StaticBearerAuthResolver,
    create_rag_app,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--subject", default="demo-user")
    parser.add_argument("--principal", action="append", default=[])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--queue-timeout-seconds", type=float, default=0.25)
    parser.add_argument("--execution-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--token-env", default="ABOUT_LLM_RAG_DEMO_TOKEN")
    parser.add_argument(
        "--allow-non-loopback-demo-auth",
        action="store_true",
        help="acknowledge that static demo auth is not a production identity boundary",
    )
    return parser


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_app(args: argparse.Namespace) -> object:
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if not _loopback(args.host) and not args.allow_non_loopback_demo_auth:
        raise ValueError(
            "static demo auth may bind only to loopback unless explicitly acknowledged"
        )
    token = os.environ.get(args.token_env)
    if token is None or not token:
        raise ValueError(f"demo bearer token environment variable {args.token_env!r} is missing")
    context = AuthContext(
        subject_id=args.subject,
        tenant_id=args.tenant,
        principals=tuple(args.principal),
    )
    service = PersistentExtractiveRAGService(
        args.database,
        config=RAGServiceConfig(
            max_concurrency=args.max_concurrency,
            queue_timeout_seconds=args.queue_timeout_seconds,
            execution_timeout_seconds=args.execution_timeout_seconds,
        ),
    )
    return create_rag_app(
        service,
        StaticBearerAuthResolver({token: context}),
        allowed_hosts=(args.host, "localhost"),
    )


def main() -> None:
    args = build_parser().parse_args()
    app = build_app(args)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        proxy_headers=False,
        server_header=False,
        access_log=True,
    )


if __name__ == "__main__":
    main()
