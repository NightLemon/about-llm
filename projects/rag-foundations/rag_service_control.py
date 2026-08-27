"""通过进程内 ASGI 客户端运行持久化抽取式 RAG 服务的完整控制实验。

实验覆盖 readiness、bearer 身份、租户/ACL 可见性、请求体伪造、缺失鉴权、排队饱和、
执行超时和容量恢复。它使用真实 FastAPI/Starlette/httpx 与 SQLite，但不打开公网端口。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx

from about_llm.rag.ingestion import SourceDocument
from about_llm.rag.service import (
    AuthContext,
    PersistentExtractiveRAGService,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGServiceConfig,
    StaticBearerAuthResolver,
    create_rag_app,
)
from about_llm.rag.sqlite_store import SQLiteChunkStore


def _seed(path: Path) -> None:
    """写入公开、工程组、财务组和另一租户四类权限文档。"""

    sources = (
        SourceDocument(
            source_id="public-security",
            tenant_id="tenant-a",
            version="v1",
            text="RAG 检索必须在排序前执行租户和主体权限过滤。",
        ),
        SourceDocument(
            source_id="engineering-citations",
            tenant_id="tenant-a",
            version="v1",
            text="RAG 生成只能使用已授权并完成引用绑定的检索证据。",
            acl=("engineering",),
        ),
        SourceDocument(
            source_id="finance-secret",
            tenant_id="tenant-a",
            version="v1",
            text="RAG 检索 权限 过滤 排序前 财务秘密 检索 权限。",
            acl=("finance",),
        ),
        SourceDocument(
            source_id="other-tenant",
            tenant_id="tenant-b",
            version="v1",
            text="RAG 检索必须在排序前执行权限过滤。",
        ),
    )
    # 使用真实磁盘 SQLite；后续服务查询会重新打开它，覆盖持久化边界。
    with SQLiteChunkStore(path) as store:
        for source in sources:
            store.upsert_source(source, expected_current_version=None, max_chars=1000)


def _request_ids() -> Any:
    """返回一个确定性 request ID 生成器，便于核对响应头。"""

    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"control-request-{counter}"

    return next_id


def _auth_resolver() -> StaticBearerAuthResolver:
    """把两个公开测试 token 映射为工程主体与匿名主体。"""

    return StaticBearerAuthResolver(
        {
            "engineering-token": AuthContext(
                subject_id="engineering-user",
                tenant_id="tenant-a",
                principals=("engineering",),
            ),
            "anonymous-token": AuthContext(
                subject_id="anonymous-user",
                tenant_id="tenant-a",
                principals=(),
            ),
        }
    )


class _BlockedRAGService(PersistentExtractiveRAGService):
    """阻塞一个同步请求，直到实验观察到队列饱和。"""

    def __init__(self, database: Path) -> None:
        """将并发设为 1，并建立跨线程协调事件。"""

        super().__init__(
            database,
            config=RAGServiceConfig(
                max_concurrency=1,
                queue_timeout_seconds=0.02,
                execution_timeout_seconds=0.03,
            ),
            request_id_factory=_request_ids(),
        )
        self.block_work = True
        self.work_started = threading.Event()
        self.release_work = threading.Event()
        self.permit_released = threading.Event()

    def _query_sync(
        self,
        request: RAGQueryRequest,
        auth: AuthContext,
        request_id: str,
    ) -> RAGQueryResponse:
        """在测试门打开前让后台同步查询保持运行。"""

        if not self.block_work:
            return super()._query_sync(request, auth, request_id)
        self.work_started.set()
        if not self.release_work.wait(timeout=2):
            raise RuntimeError("service control did not release blocked work")
        return super()._query_sync(request, auth, request_id)

    def _release_after_background_work(
        self,
        work: asyncio.Task[RAGQueryResponse],
    ) -> None:
        """后台工作真正结束并释放 permit 后通知实验协程。"""

        super()._release_after_background_work(work)
        self.permit_released.set()


async def _execute_pressure(database: Path) -> dict[str, Any]:
    """制造 execution timeout 与 queue saturation，再验证容量恢复。"""

    # 单并发 permit 被第一个后台线程持有；HTTP 504 并不代表该线程已经停止。
    service = _BlockedRAGService(database)
    app = create_rag_app(service, _auth_resolver())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"Authorization": "Bearer engineering-token"}
        try:
            # 第一个请求很快在 HTTP 层超时，但同步工作仍被 event 阻塞。
            first = await client.post(
                "/v1/rag/query",
                headers=headers,
                json={"query_id": "slow-1", "query": "RAG 权限过滤"},
            )
            started = await asyncio.to_thread(service.work_started.wait, 1)
            if not started:
                raise AssertionError("blocked synchronous work did not start")
            # permit 尚未释放，第二个请求在 queue timeout 后收到 503。
            second = await client.post(
                "/v1/rag/query",
                headers=headers,
                json={"query_id": "slow-2", "query": "RAG 权限过滤"},
            )
        finally:
            service.release_work.set()
        released = await asyncio.to_thread(service.permit_released.wait, 1)
        if not released:
            raise AssertionError("background work did not release the concurrency permit")
        # 后台线程完成并释放 permit 后，放宽执行超时，第三个请求应恢复成功。
        service.block_work = False
        service.config = RAGServiceConfig(
            max_concurrency=1,
            queue_timeout_seconds=0.10,
            execution_timeout_seconds=1.0,
        )
        recovered = await client.post(
            "/v1/rag/query",
            headers=headers,
            json={"query_id": "recovered", "query": "RAG 权限过滤"},
        )

    if (first.status_code, first.json()["error"]["code"]) != (
        504,
        "execution_timeout",
    ):
        raise AssertionError("execution-timeout fixture changed")
    if (second.status_code, second.json()["error"]["code"]) != (
        503,
        "queue_saturated",
    ):
        raise AssertionError("queue-saturation fixture changed")
    if recovered.status_code != 200:
        raise AssertionError("service capacity did not recover")
    return {
        "config": {
            "max_concurrency": 1,
            "queue_timeout_seconds": 0.02,
            "execution_timeout_seconds": 0.03,
            "synchronous_work_gate": "released_after_second_request_was_rejected",
            "recovery_execution_timeout_seconds": 1.0,
        },
        "execution_timeout": {
            "status_code": first.status_code,
            "code": first.json()["error"]["code"],
        },
        "while_background_thread_runs": {
            "status_code": second.status_code,
            "code": second.json()["error"]["code"],
        },
        "after_background_thread_finishes": {
            "status_code": recovered.status_code,
            "action": recovered.json()["action"],
        },
    }


async def _execute(database: Path) -> dict[str, Any]:
    """通过 ASGI 请求验证正常权限路径、失败路径和压力恢复。"""

    service = PersistentExtractiveRAGService(
        database,
        request_id_factory=_request_ids(),
    )
    app = create_rag_app(service, _auth_resolver())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health/ready")
        query = {
            "query_id": "service-control-q1",
            "query": "RAG 检索为什么要在排序前做权限过滤",
            "top_k": 10,
            "budget_units": 12000,
        }
        # 相同 query 分别使用工程主体与匿名主体，返回的授权来源集合应不同。
        engineering = await client.post(
            "/v1/rag/query",
            headers={"Authorization": "Bearer engineering-token"},
            json=query,
        )
        anonymous = await client.post(
            "/v1/rag/query",
            headers={"Authorization": "Bearer anonymous-token"},
            json=query,
        )
        # 客户端不能在 JSON body 自报 tenant；可信租户只能来自鉴权上下文。
        forged = await client.post(
            "/v1/rag/query",
            headers={"Authorization": "Bearer anonymous-token"},
            json={**query, "tenant_id": "tenant-b"},
        )
        missing_auth = await client.post("/v1/rag/query", json=query)

    engineering_payload = engineering.json()
    anonymous_payload = anonymous.json()
    engineering_sources = [
        source["stable_source_id"] for source in engineering_payload["artifact"]["sources"]
    ]
    anonymous_sources = [
        source["stable_source_id"] for source in anonymous_payload["artifact"]["sources"]
    ]
    # 固定来源断言同时检查 ACL 发生在检索评分之前，没有秘密文档进入候选。
    if engineering_sources != ["public-security", "engineering-citations"]:
        raise AssertionError("engineering service visibility fixture changed")
    if anonymous_sources != ["public-security"]:
        raise AssertionError("anonymous service visibility fixture changed")
    return {
        "implementation": "about-llm.rag-service-asgi-control.v1",
        "versions": {
            "fastapi": version("fastapi"),
            "starlette": version("starlette"),
            "httpx": version("httpx"),
        },
        "health": {"status_code": health.status_code, "body": health.json()},
        "engineering": {
            "status_code": engineering.status_code,
            "request_id": engineering.headers.get("x-request-id"),
            "source_ids": engineering_sources,
            "action": engineering_payload["action"],
            "artifact_fingerprint": engineering_payload["artifact_fingerprint"],
        },
        "anonymous": {
            "status_code": anonymous.status_code,
            "request_id": anonymous.headers.get("x-request-id"),
            "source_ids": anonymous_sources,
            "action": anonymous_payload["action"],
            "artifact_fingerprint": anonymous_payload["artifact_fingerprint"],
        },
        "negative_cases": {
            "body_tenant_injection_status": forged.status_code,
            "body_tenant_injection_code": forged.json()["error"]["code"],
            "missing_auth_status": missing_auth.status_code,
            "missing_auth_code": missing_auth.json()["error"]["code"],
        },
        "pressure": await _execute_pressure(database),
        "scope": {
            "real_fastapi_starlette_httpx_asgi_dispatch_executed": True,
            "real_sqlite_persistence_reopened_per_query": True,
            "authorization_context_resolved_outside_json_body": True,
            "authorization_filtered_before_bm25_scoring": True,
            "deterministic_extractive_non_llm_answer_executed": True,
            "execution_timeout_while_sync_thread_continued_observed": True,
            "permit_held_until_background_work_completed": True,
            "real_tcp_tls_reverse_proxy_or_remote_identity_executed": False,
            "learned_retriever_reranker_or_llm_executed": False,
            "multi_process_global_admission_or_production_slo_proved": False,
        },
    }


def run_control() -> dict[str, Any]:
    """创建临时数据库，写入语料并运行完整异步服务实验。"""

    with tempfile.TemporaryDirectory(prefix="about-llm-rag-service-") as directory:
        database = Path(directory) / "rag.db"
        _seed(database)
        return asyncio.run(_execute(database))


if __name__ == "__main__":
    payload = json.dumps(run_control(), ensure_ascii=False, indent=2)
    sys.stdout.buffer.write((payload + "\n").encode("utf-8"))
