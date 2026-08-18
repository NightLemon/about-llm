from __future__ import annotations

import importlib.util
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
pytestmark = [pytest.mark.contract, pytest.mark.security, pytest.mark.integration]


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
