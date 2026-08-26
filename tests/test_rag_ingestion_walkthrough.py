from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.contract,
    pytest.mark.security,
    pytest.mark.smoke,
    pytest.mark.integration,
]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "rag-foundations" / "rag_ingestion_walkthrough.py"


def test_walkthrough_connects_chunk_identity_update_and_acl_visibility() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout.decode("utf-8"))

    assert payload["walkthrough_version"] == "about-llm.rag-ingestion-walkthrough.v1"
    assert payload["source"] == {
        "source_id": "rag-security",
        "tenant_id": "tenant-a",
        "acl": ["engineering"],
        "metadata": {"uri": "kb://tenant-a/rag-security"},
    }
    assert payload["version_1"]["source_version"] == "1"
    assert payload["version_1"]["chunk_count"] == 2
    assert payload["version_2"]["source_version"] == "2"
    assert payload["version_2"]["chunk_count"] == 3

    identity = payload["identity"]
    assert identity["preserved_id_across_versions"] is True
    assert identity["edited_content_gets_new_id"] is True

    plan = payload["incremental_plan"]
    assert len(plan["upsert_chunk_ids"]) == 3
    assert len(plan["new_identity_chunk_ids"]) == 2
    assert plan["same_identity_payload_updates"] == [identity["preserved_chunk_id"]]
    assert plan["delete_chunk_ids"] == [identity["edited_old_chunk_id"]]
    assert plan["unchanged_chunk_ids"] == []

    sqlite = payload["sqlite"]
    assert sqlite["initial_upsert_count"] == 2
    assert sqlite["update_upsert_count"] == 3
    assert sqlite["update_delete_count"] == 1
    assert sqlite["current_version"] == "2"
    assert sqlite["anonymous_visible_chunk_ids"] == []
    assert len(sqlite["engineering_visible_chunk_ids"]) == 3
    assert sqlite["stale_update"] == {
        "rejected": True,
        "message": "source version conflict: expected '1', found '2'",
    }

    assert payload["scope"] == {
        "sample_corpus_loaded": True,
        "markdown_split_executed": True,
        "incremental_plan_executed": True,
        "sqlite_transactions_executed": True,
        "acl_filter_executed": True,
        "embedding_or_search_index_executed": False,
        "external_source_authenticity_verified": False,
        "multi_store_production_update_verified": False,
    }
