"""Execute the persistent extractive RAG service through an in-process ASGI client."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx

from about_llm.rag.ingestion import SourceDocument
from about_llm.rag.service import (
    AuthContext,
    PersistentExtractiveRAGService,
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


async def _execute(database: Path) -> dict[str, Any]:
    service = PersistentExtractiveRAGService(
        database,
        request_id_factory=_request_ids(),
    )
    resolver = StaticBearerAuthResolver(
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
    app = create_rag_app(service, resolver)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://control") as client:
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
        "scope": {
            "real_fastapi_starlette_httpx_asgi_dispatch_executed": True,
            "real_sqlite_persistence_reopened_per_query": True,
            "authorization_context_resolved_outside_json_body": True,
            "authorization_filtered_before_bm25_scoring": True,
            "deterministic_extractive_non_llm_answer_executed": True,
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
