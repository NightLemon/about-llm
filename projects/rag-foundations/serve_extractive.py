"""启动带 SQLite 持久化和演示 bearer auth 的确定性抽取式 RAG 服务。

该服务适合本机学习请求、权限和并发控制，不包含生产身份系统。默认只监听 loopback；若要
绑定外部地址，必须显式确认静态 token 不是可靠的生产安全边界。
"""

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
    """定义数据库、演示身份、监听地址与并发/超时限制。"""

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
    """判断 host 是否只在本机可访问。"""

    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_app(args: argparse.Namespace) -> object:
    """从命令行参数构造持久化 RAG 服务与静态鉴权解析器。"""

    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    # 静态 token 暴露到非本机地址风险更高，必须由操作者显式确认。
    if not _loopback(args.host) and not args.allow_non_loopback_demo_auth:
        raise ValueError(
            "static demo auth may bind only to loopback unless explicitly acknowledged"
        )
    # token 只从环境变量读取，避免把秘密放进命令历史或源码。
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
    """构造应用并交给 Uvicorn 运行。"""

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
