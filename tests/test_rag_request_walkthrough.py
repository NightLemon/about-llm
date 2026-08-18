from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.smoke, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "rag-foundations" / "rag_request_walkthrough.py"


def test_walkthrough_connects_answerable_and_no_answer_requests() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout.decode("utf-8"))

    assert payload["walkthrough_version"] == "about-llm.rag-request-walkthrough.v1"
    answerable, no_answer = payload["requests"]

    assert answerable["final"] == {
        "action": "answer",
        "reason": "exact_span_and_citation_syntax_passed",
    }
    assert answerable["answer"]["coverage"] == 1.0
    assert answerable["citation"]["syntactically_valid"] is True
    assert answerable["packing"]["source_map"] == {
        "S1": "rag-security",
        "S2": "rag-security",
    }
    assert [row["stable_source_id"] for row in answerable["rerank"]["results"]] == [
        "rag-security",
        "rag-security",
    ]

    assert no_answer["retrieval"]["candidate_count"] == 3
    assert no_answer["final"] == {
        "action": "abstain",
        "reason": "insufficient_lexical_evidence",
    }
    assert no_answer["answer"]["coverage"] == pytest.approx(2 / 9)
    assert no_answer["citation"]["cited_source_ids"] == []

    assert payload["scope"]["authorization_rechecked_before_reranker"] is True
    assert payload["scope"]["learned_reranker_executed"] is False
    assert payload["scope"]["llm_executed"] is False
    assert payload["scope"]["semantic_entailment_or_source_truth_verified"] is False
