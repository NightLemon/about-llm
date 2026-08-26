"""Execute the persistent extractive RAG service through an in-process ASGI client."""

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
    with SQLiteChunkStore(path) as store:
        for source in sources:
            store.upsert_source(source, expected_current_version=None, max_chars=1000)


def _request_ids() -> Any:
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"control-request-{counter}"

    return next_id


def _auth_resolver() -> StaticBearerAuthResolver:
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
    """Hold one sync request until the queue-saturation state has been observed."""

    def __init__(self, database: Path) -> None:
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
        super()._release_after_background_work(work)
        self.permit_released.set()


async def _execute_pressure(database: Path) -> dict[str, Any]:
    service = _BlockedRAGService(database)
    app = create_rag_app(service, _auth_resolver())
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"Authorization": "Bearer engineering-token"}
        try:
            first = await client.post(
                "/v1/rag/query",
                headers=headers,
                json={"query_id": "slow-1", "query": "RAG 权限过滤"},
            )
            started = await asyncio.to_thread(service.work_started.wait, 1)
            if not started:
                raise AssertionError("blocked synchronous work did not start")
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
    with tempfile.TemporaryDirectory(prefix="about-llm-rag-service-") as directory:
        database = Path(directory) / "rag.db"
        _seed(database)
        return asyncio.run(_execute(database))


if __name__ == "__main__":
    payload = json.dumps(run_control(), ensure_ascii=False, indent=2)
    sys.stdout.buffer.write((payload + "\n").encode("utf-8"))
