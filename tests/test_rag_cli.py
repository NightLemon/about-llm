from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from about_llm.rag.cli import (
    MarkdownBM25Pipeline,
    RetrievalCase,
    evaluate_retrieval,
    load_cases,
    load_corpus,
    main,
)

pytestmark = [pytest.mark.contract, pytest.mark.security]

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "rag-foundations"
CORPUS = PROJECT / "sample_corpus.jsonl"
CASES = PROJECT / "sample_eval.jsonl"
ANSWERS = PROJECT / "sample_answers.jsonl"
TRACES = PROJECT / "generation-traces.example.jsonl"
RERANK_SCORES = PROJECT / "reranker-scores.example.jsonl"
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] + ' ' + message['content'] + ' ' + eos_token + ' ' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
)


def _save_test_tokenizer(path: Path) -> None:
    vocabulary = {
        "[UNK]": 0,
        "[PAD]": 1,
        "</s>": 2,
        "system": 3,
        "user": 4,
        "assistant": 5,
        "question": 6,
        "evidence": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="</s>",
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(path)


def test_pipeline_enforces_tenant_and_principal_acl() -> None:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))

    anonymous = pipeline.retrieve("引用", tenant_id="tenant-a", top_k=20)
    engineer = pipeline.retrieve(
        "引用", tenant_id="tenant-a", principals=("engineering",), top_k=20
    )

    assert all(result.document.tenant_id == "tenant-a" for result in engineer)
    assert {result.document.metadata["source_id"] for result in anonymous} == {
        "rag-evaluation"
    }
    assert "rag-security" in {
        result.document.metadata["source_id"] for result in engineer
    }
    assert "tenant-b-secret" not in {
        result.document.metadata["source_id"] for result in engineer
    }


def test_source_level_evaluation_uses_reusable_metrics() -> None:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))
    report = evaluate_retrieval(pipeline, load_cases(CASES), k=3)

    assert report["case_count"] == 5
    assert report["answerable_metrics"]["case_count"] == 3
    assert report["no_answer_metrics"]["case_count"] == 2
    assert report["recall_at_k"] == pytest.approx(1.0)
    assert report["mrr_at_k"] == pytest.approx(1.0)
    assert report["ndcg_at_k"] == pytest.approx(1.0)
    assert report["all_evidence_recall_at_k"] == pytest.approx(1.0)
    assert report["no_answer_metrics"]["zero_result_accuracy"] == pytest.approx(0.5)
    assert report["legacy_metric_scope"] == "answerable cases only"

    rows = {row["query_id"]: row for row in report["cases"]}
    assert rows["metrics-and-entailment"]["missing_required_source_ids"] == []
    assert rows["unrelated-no-answer"]["metrics"]["zero_results"] is True
    assert rows["topical-no-answer"]["metrics"]["zero_results"] is False
    assert "does not prove" in rows["topical-no-answer"]["metrics"]["note"]


def test_evaluation_diagnoses_acl_blocked_gold_without_leaking_other_tenants() -> None:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))
    case = RetrievalCase(
        query_id="blocked",
        query="SFT chat template labels mask",
        tenant_id="tenant-a",
        relevant_source_ids=frozenset({"finetuning-basics"}),
    )

    report = evaluate_retrieval(pipeline, [case], k=3)
    row = report["cases"][0]
    assert row["gold_source_status"] == {"finetuning-basics": "acl_blocked"}
    assert row["missing_required_source_ids"] == ["finetuning-basics"]
    assert report["recall_at_k"] == 0.0


def test_load_cases_keeps_legacy_schema_and_requires_explicit_no_answer(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        '{"query_id":"q","query":"x","tenant_id":"t",'
        '"relevant_source_ids":["s"]}\n',
        encoding="utf-8",
    )
    loaded = load_cases(legacy)
    assert loaded[0].relevance == {"s": 1.0}
    assert loaded[0].required_source_ids == frozenset({"s"})

    ambiguous = tmp_path / "ambiguous.jsonl"
    ambiguous.write_text(
        '{"query_id":"q","query":"x","tenant_id":"t","relevance":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires answerable=false"):
        load_cases(ambiguous)


def test_load_cases_validates_graded_and_no_answer_contracts(tmp_path: Path) -> None:
    graded = tmp_path / "graded.jsonl"
    graded.write_text(
        '{"query_id":"q","query":"x","tenant_id":"t","answerable":true,'
        '"relevance":{"best":3,"judged-noise":0},'
        '"required_source_ids":["best"]}\n',
        encoding="utf-8",
    )
    loaded = load_cases(graded)[0]
    assert loaded.relevant_source_ids == frozenset({"best"})
    assert loaded.relevance == {"best": 3.0, "judged-noise": 0.0}

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        '{"query_id":"q","query":"x","tenant_id":"t","answerable":false,'
        '"relevance":{"hidden-gold":1}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no-answer case"):
        load_cases(invalid)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ('{"source_id":"a","source_id":"b"}\n', "duplicate JSON key"),
        (
            '{"source_id":"a","tenant_id":"t","version":"v1",'
            '"text":"x","metadata":{"score":NaN}}\n',
            "non-finite JSON constant",
        ),
        (
            '{"source_id":"a","tenant_id":"t","version":"v1",'
            '"text":"x","metadata":{"score":1e999}}\n',
            "non-finite JSON number",
        ),
    ],
)
def test_corpus_loader_rejects_ambiguous_or_nonfinite_json(
    tmp_path: Path, record: str, message: str
) -> None:
    corpus = tmp_path / "invalid.jsonl"
    corpus.write_text(record, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_corpus(corpus)


def test_all_no_answer_report_does_not_publish_answerable_metric_aliases() -> None:
    pipeline = MarkdownBM25Pipeline(load_corpus(CORPUS))
    report = evaluate_retrieval(
        pipeline,
        [
            RetrievalCase(
                query_id="no-answer",
                query="量子退火天气预测",
                tenant_id="tenant-a",
                relevant_source_ids=frozenset(),
                answerable=False,
            )
        ],
        k=3,
    )

    assert report["answerable_metrics"] == {"case_count": 0}
    assert report["no_answer_metrics"]["zero_result_accuracy"] == 1.0
    assert "recall_at_k" not in report


@pytest.mark.smoke
def test_cli_retrieve_prints_authorized_context(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "retrieve",
            "--corpus",
            str(CORPUS),
            "--query",
            "ACL 引用",
            "--tenant",
            "tenant-a",
            "--principal",
            "engineering",
            "--top-k",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert {item["source_id"] for item in payload["retrieved"]} == {
        "rag-evaluation",
        "rag-security",
    }
    assert set(payload["context"]["sources"]) == {"S1", "S2"}


@pytest.mark.smoke
def test_sqlite_store_cli_upsert_retrieve_update_and_delete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "rag.db"
    corpus_v1 = tmp_path / "corpus-v1.jsonl"
    corpus_v1.write_text(
        '{"source_id":"public","tenant_id":"tenant-a","version":"v1",'
        '"text":"# Public\\n\\n共享检索说明。","acl":[],"metadata":{}}\n'
        '{"source_id":"restricted","tenant_id":"tenant-a","version":"v1",'
        '"text":"# Restricted\\n\\n机密检索流程。",'
        '"acl":["engineering"],"metadata":{}}\n',
        encoding="utf-8",
    )

    for source_id in ("public", "restricted"):
        assert main(
            [
                "store-upsert",
                "--database",
                str(database),
                "--corpus",
                str(corpus_v1),
                "--tenant",
                "tenant-a",
                "--source-id",
                source_id,
                "--expect-absent",
            ]
        ) == 0
        committed = json.loads(capsys.readouterr().out)
        assert committed["committed"] is True
        assert committed["expected_current_version"] is None
        assert committed["plan"]["upsert_chunk_ids"]
        assert committed["scope"]["cross_store_atomicity_proved"] is False

    assert main(
        [
            "store-retrieve",
            "--database",
            str(database),
            "--query",
            "机密检索流程",
            "--tenant",
            "tenant-a",
            "--top-k",
            "10",
        ]
    ) == 0
    anonymous = json.loads(capsys.readouterr().out)
    assert anonymous["authorized_candidate_count"] == 1
    assert {row["source_id"] for row in anonymous["retrieved"]} <= {"public"}

    assert main(
        [
            "store-retrieve",
            "--database",
            str(database),
            "--query",
            "机密检索流程",
            "--tenant",
            "tenant-a",
            "--principal",
            "engineering",
            "--top-k",
            "10",
        ]
    ) == 0
    authorized = json.loads(capsys.readouterr().out)
    assert authorized["authorized_candidate_count"] == 2
    assert "restricted" in {row["source_id"] for row in authorized["retrieved"]}
    assert authorized["scope"]["principal_acl_filtered_before_scoring"] is True

    corpus_v2 = tmp_path / "corpus-v2.jsonl"
    corpus_v2.write_text(
        '{"source_id":"restricted","tenant_id":"tenant-a","version":"v2",'
        '"text":"# Restricted\\n\\n更新后的机密检索流程。",'
        '"acl":["engineering"],"metadata":{}}\n',
        encoding="utf-8",
    )
    assert main(
        [
            "store-upsert",
            "--database",
            str(database),
            "--corpus",
            str(corpus_v2),
            "--tenant",
            "tenant-a",
            "--source-id",
            "restricted",
            "--expected-current-version",
            "v1",
        ]
    ) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["new_version"] == "v2"
    assert updated["plan"]["delete_chunk_ids"]

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "store-delete",
                "--database",
                str(database),
                "--tenant",
                "tenant-a",
                "--source-id",
                "restricted",
                "--expected-current-version",
                "v1",
            ]
        )
    assert "version conflict" in capsys.readouterr().err

    assert main(
        [
            "store-delete",
            "--database",
            str(database),
            "--tenant",
            "tenant-a",
            "--source-id",
            "restricted",
            "--expected-current-version",
            "v2",
        ]
    ) == 0
    deleted = json.loads(capsys.readouterr().out)
    assert deleted["committed"] is True
    assert deleted["deleted_chunk_ids"]


@pytest.mark.smoke
def test_cli_pack_labels_byte_budget_without_calling_it_tokens(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "pack",
            "--corpus",
            str(CORPUS),
            "--query",
            "ACL 引用",
            "--tenant",
            "tenant-a",
            "--principal",
            "engineering",
            "--candidate-k",
            "20",
            "--budget-bytes",
            "10000",
            "--max-chunks-per-source",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    packing = payload["packing"]
    assert exit_code == 0
    assert packing["cost_unit"] == "utf8_bytes"
    assert packing["used_cost_units"] <= packing["budget_units"]
    assert "not model tokens" in packing["warning"]
    selected = [decision for decision in packing["decisions"] if decision["selected"]]
    dropped = [decision for decision in packing["decisions"] if not decision["selected"]]
    assert len(selected) == 2
    assert {decision["reason"] for decision in dropped} == {"source_quota"}
    assert {source["stable_source_id"] for source in payload["context"]["sources"].values()} == {
        "rag-evaluation",
        "rag-security",
    }


@pytest.mark.smoke
def test_cli_pack_tokenized_uses_full_chat_template_and_output_reservation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer_path = tmp_path / "tokenizer"
    _save_test_tokenizer(tokenizer_path)
    system_prompt = tmp_path / "system.txt"
    system_prompt.write_text("use authorized evidence only", encoding="utf-8")
    user_template = tmp_path / "user.txt"
    user_template.write_text(
        "question {query} evidence {context}", encoding="utf-8"
    )

    exit_code = main(
        [
            "pack-tokenized",
            "--corpus",
            str(CORPUS),
            "--query",
            "ACL 引用",
            "--tenant",
            "tenant-a",
            "--principal",
            "engineering",
            "--candidate-k",
            "20",
            "--max-total-tokens",
            "64",
            "--reserved-output-tokens",
            "16",
            "--tokenizer",
            str(tokenizer_path),
            "--tokenizer-revision",
            "local-wordlevel-v1",
            "--local-files-only",
            "--system-prompt-file",
            str(system_prompt),
            "--user-prompt-template-file",
            str(user_template),
            "--max-chunks-per-source",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    packing = payload["packing"]
    assert exit_code == 0
    assert payload["target_tokenizer"]["revision"] == "local-wordlevel-v1"
    assert payload["target_tokenizer"]["transformers_version"]
    assert payload["target_tokenizer"]["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert payload["target_tokenizer"]["chat_template_sha256"].startswith("sha256:")
    assert packing["reserved_output_tokens"] == 16
    assert packing["used_total_with_output_reservation"] <= 64
    assert packing["used_prompt_tokens"] + 16 == (
        packing["used_total_with_output_reservation"]
    )
    assert packing["final_prompt_token_count"] == len(
        packing["final_prompt_token_ids"]
    )
    assert packing["final_prompt_token_count"] == packing["used_prompt_tokens"]
    assert payload["scope"]["complete_chat_prompt_retokenized_per_candidate"] is True
    assert payload["scope"]["final_prompt_token_ids_recorded"] is True
    assert payload["scope"]["model_context_window_verified"] is False
    assert payload["scope"]["generation_quality_or_grounding_verified"] is False


@pytest.mark.smoke
def test_cli_evaluate_prints_explicit_metric_scopes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "evaluate",
            "--corpus",
            str(CORPUS),
            "--cases",
            str(CASES),
            "--top-k",
            "3",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["answerable_metrics"]["case_count"] == 3
    assert payload["no_answer_metrics"]["case_count"] == 2
    assert payload["legacy_metric_scope"] == "answerable cases only"


@pytest.mark.smoke
def test_cli_evaluate_answers_keeps_supplied_judgment_scope_explicit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "evaluate-answers",
            "--corpus",
            str(CORPUS),
            "--cases",
            str(CASES),
            "--answers",
            str(ANSWERS),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["recorded_gate_pass_rate"] == 1.0
    assert payload["claim_judgment_coverage"] == 1.0
    assert "not entailment inferred" in payload["scope_warning"]


@pytest.mark.smoke
def test_cli_audit_traces_passes_reconstructable_fixture(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "audit-traces",
            "--corpus",
            str(CORPUS),
            "--cases",
            str(CASES),
            "--answers",
            str(ANSWERS),
            "--traces",
            str(TRACES),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["gate_passed"] is True
    assert payload["trace_count"] == 5
    assert payload["finding_counts"] == {}
    assert payload["scope"]["trace_context_reconstructed_from_current_chunks"] is True
    assert payload["scope"]["raw_output_claim_semantics_verified"] is False
    assert payload["scope"]["remote_model_execution_verified"] is False


def test_cli_audit_traces_returns_one_for_content_tampering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tampered = tmp_path / "traces.jsonl"
    tampered.write_text(
        TRACES.read_text(encoding="utf-8").replace(
            "sha256:cf73f4abd570a45a2c8d94d3a001dbe1a85b8ac0c07042195730f2067743cc68",
            "sha256:" + "0" * 64,
            1,
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "audit-traces",
            "--corpus",
            str(CORPUS),
            "--cases",
            str(CASES),
            "--answers",
            str(ANSWERS),
            "--traces",
            str(tampered),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["gate_passed"] is False
    assert payload["finding_counts"] == {"source_content_mismatch": 1}


@pytest.mark.smoke
def test_recorded_rerank_cli_binds_scores_and_reorders_authorized_candidates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "rerank-recorded",
            "--corpus",
            str(CORPUS),
            "--scores",
            str(RERANK_SCORES),
            "--query",
            "RAG 为什么要先做 ACL 权限过滤",
            "--tenant",
            "tenant-a",
            "--principal",
            "engineering",
            "--candidate-k",
            "3",
            "--top-k",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    report = payload["rerank_report"]
    assert exit_code == 0
    assert [result["document_id"] for result in report["results"]] == [
        "chk_bd3e8a6757a7f05c07fdbcc4",
        "chk_8d8a68a02d85a198190fc293",
    ]
    assert report["scorer_identity"] == "authored-reranker-fixture@v1"
    assert report["authorized_candidate_count"] == 3
    assert report["scope"]["authorization_rechecked_before_scorer"] is True
    assert "do not prove a learned model ran" in payload["evidence_boundary"]


def test_recorded_rerank_cli_rejects_stale_query_binding() -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "rerank-recorded",
                "--corpus",
                str(CORPUS),
                "--scores",
                str(RERANK_SCORES),
                "--query",
                "changed query",
                "--tenant",
                "tenant-a",
                "--principal",
                "engineering",
                "--candidate-k",
                "3",
                "--top-k",
                "2",
            ]
        )

    assert error.value.code == 2
