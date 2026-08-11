from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("fastapi")

from about_llm.rag.ingestion import SourceDocument
from about_llm.rag.sqlite_store import SQLiteChunkStore

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "projects" / "rag-foundations" / "rag_service_control.py"
SERVER = ROOT / "projects" / "rag-foundations" / "serve_extractive.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _database(path: Path) -> Path:
    with SQLiteChunkStore(path) as store:
        store.upsert_source(
            SourceDocument(
                source_id="source",
                tenant_id="tenant-a",
                version="v1",
                text="RAG 检索必须先做权限过滤。",
            ),
            expected_current_version=None,
            max_chars=1000,
        )
    return path


def test_asgi_control_locks_identity_acl_and_artifact_evidence() -> None:
    report = _load(CONTROL, "rag_service_control").run_control()

    assert report["health"]["body"]["status"] == "ready"
    assert report["engineering"] == {
        "status_code": 200,
        "request_id": "control-request-1",
        "source_ids": ["public-security", "engineering-citations"],
        "action": "answer",
        "artifact_fingerprint": (
            "sha256:cdc57ac0c4f54562b2d3e595046febd78cd635476d067c227ebafc98f73fbe89"
        ),
    }
    assert report["anonymous"] == {
        "status_code": 200,
        "request_id": "control-request-2",
        "source_ids": ["public-security"],
        "action": "answer",
        "artifact_fingerprint": (
            "sha256:5bc0701cb8b5d54705541273a2200327965e1572af631a80396d8a5b1f37d91a"
        ),
    }
    assert report["negative_cases"] == {
        "body_tenant_injection_status": 422,
        "body_tenant_injection_code": "invalid_request",
        "missing_auth_status": 401,
        "missing_auth_code": "unauthorized",
    }
    assert report["scope"] == {
        "real_fastapi_starlette_httpx_asgi_dispatch_executed": True,
        "real_sqlite_persistence_reopened_per_query": True,
        "authorization_context_resolved_outside_json_body": True,
        "authorization_filtered_before_bm25_scoring": True,
        "deterministic_extractive_non_llm_answer_executed": True,
        "real_tcp_tls_reverse_proxy_or_remote_identity_executed": False,
        "learned_retriever_reranker_or_llm_executed": False,
        "multi_process_global_admission_or_production_slo_proved": False,
    }


def test_asgi_control_is_deterministic_and_cli_is_utf8_json() -> None:
    first = _load(CONTROL, "rag_service_control_first").run_control()
    second = _load(CONTROL, "rag_service_control_second").run_control()
    first.pop("versions")
    second.pop("versions")
    assert first == second

    completed = subprocess.run(
        [sys.executable, str(CONTROL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert json.loads(completed.stdout)["implementation"] == (
        "about-llm.rag-service-asgi-control.v1"
    )


def test_demo_server_refuses_public_bind_without_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(SERVER, "serve_extractive_public_bind")
    database = _database(tmp_path / "rag.db")
    args = module.build_parser().parse_args(
        [
            "--database",
            str(database),
            "--tenant",
            "tenant-a",
            "--host",
            "0.0.0.0",
        ]
    )
    monkeypatch.setenv("ABOUT_LLM_RAG_DEMO_TOKEN", "secret-token")

    with pytest.raises(ValueError, match="loopback"):
        module.build_app(args)


def test_demo_server_requires_token_from_environment_not_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(SERVER, "serve_extractive_token")
    database = _database(tmp_path / "rag.db")
    args = module.build_parser().parse_args(
        ["--database", str(database), "--tenant", "tenant-a"]
    )
    monkeypatch.delenv("ABOUT_LLM_RAG_DEMO_TOKEN", raising=False)
    with pytest.raises(ValueError, match="environment variable"):
        module.build_app(args)

    monkeypatch.setenv("ABOUT_LLM_RAG_DEMO_TOKEN", "secret-token")
    app = module.build_app(args)
    assert app.docs_url is None
    assert "secret-token" not in repr(app)
