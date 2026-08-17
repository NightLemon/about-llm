from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("llama_index.core")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "rag-framework-adapters" / "parity_control.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rag_framework_parity_control", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parity_control_executes_both_framework_retriever_apis() -> None:
    report = _load_script().run_control()

    engineering = report["cases"]["engineering"]
    anonymous = report["cases"]["anonymous"]
    assert engineering["canonical_document_ids"] == [
        "acl-before-ranking",
        "citation-binding",
    ]
    assert engineering["langchain_document_ids"] == engineering["canonical_document_ids"]
    assert engineering["llamaindex_document_ids"] == engineering["canonical_document_ids"]
    assert anonymous["canonical_document_ids"] == ["acl-before-ranking"]
    assert report["metrics"] == {
        "engineering_recall_at_4": 1.0,
        "engineering_ndcg_at_4": 1.0,
    }
    assert all(report["assertions"].values())


def test_parity_control_binds_prompt_and_answer_identity() -> None:
    first = _load_script().run_control()
    second = _load_script().run_control()
    first.pop("framework_versions")
    second.pop("framework_versions")
    assert first == second

    engineering = first["cases"]["engineering"]
    assert engineering["answer_action"] == "answer"
    assert engineering["answer_coverage"] == 1.0
    assert engineering["answer_text"].endswith("[S1]")
    assert engineering["answer_artifact_fingerprint"].startswith("sha256:")
    assert len(engineering["prompt_sha256"]) == 64


def test_parity_control_cli_emits_machine_readable_scope() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(completed.stdout)

    assert report["scope"] == {
        "real_langchain_and_llamaindex_core_executed": True,
        "canonical_bm25_authorization_and_ranking_used": True,
        "deterministic_extractive_non_llm_answer_used": True,
        "learned_embedding_vector_index_or_reranker_executed": False,
        "provider_or_local_llm_generation_executed": False,
        "framework_default_acl_or_security_proved": False,
        "model_quality_latency_scalability_or_production_safety_proved": False,
    }
