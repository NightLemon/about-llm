"""离线比较 LangChain 与 LlamaIndex 适配后的 RAG 结果和权限边界。

实验让两个框架包装同一个 canonical BM25 检索器，再比较检索 ID、prompt 和抽取式回答。
同时使用 engineering 与匿名主体运行两遍，确认框架适配没有绕过租户或 ACL 过滤。
"""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import version
from typing import Any

from about_llm.evaluation.retrieval import (
    normalized_discounted_cumulative_gain,
    recall_at_k,
)
from about_llm.integrations.rag_frameworks import (
    build_langchain_retriever,
    build_llamaindex_retriever,
    validate_langchain_round_trip,
    validate_llamaindex_round_trip,
)
from about_llm.rag import BM25Index, Document, SearchResult
from about_llm.rag.context_packing import utf8_byte_length
from about_llm.rag.extractive import generate_extractive_answer

QUERY = "RAG 检索为什么要在排序前做权限过滤"
TENANT_ID = "tenant-a"
PROMPT_TEMPLATE = """你是一个证据受限的问答系统。
只能依据以下已授权证据回答; 证据不足时拒答。

{context}

问题: {query}
回答: """


def _corpus() -> tuple[Document, ...]:
    """构造公开、工程组、财务组和另一租户四类权限文档。"""

    return (
        Document(
            "acl-before-ranking",
            "RAG 检索必须在排序前执行租户和主体权限过滤。",
            TENANT_ID,
            {"source_id": "security", "source_version": "v1"},
        ),
        Document(
            "citation-binding",
            "RAG 生成只应使用经过授权并完成引用绑定的检索证据。",
            TENANT_ID,
            {"source_id": "generation", "source_version": "v1"},
            acl=("engineering",),
        ),
        Document(
            "finance-secret",
            "RAG 检索 权限 过滤 排序前 检索 权限。",
            TENANT_ID,
            {"source_id": "finance", "source_version": "v1"},
            acl=("finance",),
        ),
        Document(
            "other-tenant",
            "RAG 检索必须在排序前执行权限过滤。",
            "tenant-b",
            {"source_id": "other", "source_version": "v1"},
        ),
    )


def _context(results: tuple[SearchResult, ...]) -> str:
    """把已授权结果按排名拼成两个框架共享的 context。"""

    return "\n\n".join(
        f"[{result.rank}] document_id={result.document.document_id}\n{result.document.text}"
        for result in results
    )


def _render_prompts(context: str) -> tuple[str, str]:
    """分别调用两个框架的 PromptTemplate 渲染同一模板。"""

    # 延迟导入使缺少可选框架依赖时错误只出现在真正运行实验的位置。
    from langchain_core.prompts import PromptTemplate as LangChainPromptTemplate
    from llama_index.core.prompts import PromptTemplate as LlamaIndexPromptTemplate

    langchain_prompt = LangChainPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context,
        query=QUERY,
    )
    llamaindex_prompt = LlamaIndexPromptTemplate(PROMPT_TEMPLATE).format(
        context=context,
        query=QUERY,
    )
    return langchain_prompt, llamaindex_prompt


def _run_case(
    index: BM25Index,
    *,
    principals: tuple[str, ...],
) -> dict[str, Any]:
    """以一组 principals 运行 canonical、LangChain 和 LlamaIndex 三条路径。"""

    # canonical 结果是权限与排序的唯一来源，两个适配器应逐项保持它。
    canonical = tuple(
        index.search(
            QUERY,
            tenant_id=TENANT_ID,
            principals=principals,
            top_k=4,
        )
    )
    langchain_documents = build_langchain_retriever(
        index,
        tenant_id=TENANT_ID,
        principals=principals,
        top_k=4,
    ).invoke(QUERY)
    llamaindex_nodes = build_llamaindex_retriever(
        index,
        tenant_id=TENANT_ID,
        principals=principals,
        top_k=4,
    ).retrieve(QUERY)
    langchain_results = validate_langchain_round_trip(langchain_documents, canonical)
    llamaindex_results = validate_llamaindex_round_trip(llamaindex_nodes, canonical)

    # 检索 parity 通过后，再比较框架模板渲染和最终回答 artifact。
    context = _context(canonical)
    langchain_prompt, llamaindex_prompt = _render_prompts(context)
    if langchain_prompt != llamaindex_prompt:
        raise AssertionError("framework prompt renderings differ")

    answer_kwargs = {
        "query_id": "framework-parity-query",
        "query": QUERY,
        "tenant_id": TENANT_ID,
        "principals": principals,
        "cost_fn": utf8_byte_length,
        "budget_units": 4096,
        "cost_unit": "utf8_bytes",
    }
    langchain_answer = generate_extractive_answer(langchain_results, **answer_kwargs)
    llamaindex_answer = generate_extractive_answer(llamaindex_results, **answer_kwargs)
    if langchain_answer.artifact_fingerprint != llamaindex_answer.artifact_fingerprint:
        raise AssertionError("framework answer artifacts differ")

    document_ids = [result.document.document_id for result in canonical]
    return {
        "principals": list(principals),
        "authorized_evidence": [
            {
                "rank": result.rank,
                "document_id": result.document.document_id,
                "stable_source_id": result.document.metadata["source_id"],
                "text": result.document.text,
                "score": result.score,
            }
            for result in canonical
        ],
        "canonical_document_ids": document_ids,
        "langchain_document_ids": [document.id for document in langchain_documents],
        "llamaindex_document_ids": [item.node.node_id for item in llamaindex_nodes],
        "retrieval_scores": [result.score for result in canonical],
        "prompt_sha256": hashlib.sha256(langchain_prompt.encode("utf-8")).hexdigest(),
        "prompt_utf8_bytes": len(langchain_prompt.encode("utf-8")),
        "rendered_prompt": langchain_prompt,
        "answer_action": langchain_answer.action.value,
        "answer_text": langchain_answer.answer_text,
        "answer_artifact_fingerprint": langchain_answer.artifact_fingerprint,
        "answer_coverage": langchain_answer.coverage,
    }


def run_control() -> dict[str, Any]:
    """运行工程组与匿名访问两组对照，返回可复核报告。"""

    index = BM25Index(_corpus())
    engineering = _run_case(index, principals=("engineering",))
    anonymous = _run_case(index, principals=())
    engineering_ids = engineering["canonical_document_ids"]
    retrieved = {"q1": engineering_ids}
    relevant = {"q1": {"acl-before-ranking", "citation-binding"}}
    graded = {"q1": {"acl-before-ranking": 2.0, "citation-binding": 1.0}}

    # 固定预期 ID 能让文档顺序或权限集合的意外变化立即失败。
    expected_engineering = ["acl-before-ranking", "citation-binding"]
    expected_anonymous = ["acl-before-ranking"]
    if engineering_ids != expected_engineering:
        raise AssertionError("engineering retrieval fixture changed")
    if anonymous["canonical_document_ids"] != expected_anonymous:
        raise AssertionError("anonymous retrieval fixture changed")

    return {
        "implementation": "about-llm.rag-framework-parity-control.v1",
        "framework_versions": {
            "langchain_core": version("langchain-core"),
            "llama_index_core": version("llama-index-core"),
        },
        "query": QUERY,
        "tenant_id": TENANT_ID,
        "cases": {
            "engineering": engineering,
            "anonymous": anonymous,
        },
        "metrics": {
            "engineering_recall_at_4": recall_at_k(retrieved, relevant, k=4),
            "engineering_ndcg_at_4": normalized_discounted_cumulative_gain(
                retrieved,
                graded,
                k=4,
            ),
        },
        "assertions": {
            "langchain_retriever_api_executed": True,
            "llamaindex_retriever_api_executed": True,
            "framework_results_exactly_match_canonical_retrieval": True,
            "protected_metadata_excluded_from_llamaindex_llm_and_embed_content": True,
            "cross_tenant_document_excluded": "other-tenant" not in engineering_ids,
            "wrong_principal_document_excluded": "finance-secret" not in engineering_ids,
            "acl_changes_visible_result_set": engineering_ids
            != anonymous["canonical_document_ids"],
            "framework_prompts_and_answer_artifacts_match": True,
        },
        "scope": {
            "real_langchain_and_llamaindex_core_executed": True,
            "canonical_bm25_authorization_and_ranking_used": True,
            "deterministic_extractive_non_llm_answer_used": True,
            "learned_embedding_vector_index_or_reranker_executed": False,
            "provider_or_local_llm_generation_executed": False,
            "framework_default_acl_or_security_proved": False,
            "model_quality_latency_scalability_or_production_safety_proved": False,
        },
    }


if __name__ == "__main__":
    payload = json.dumps(run_control(), ensure_ascii=False, indent=2)
    sys.stdout.buffer.write((payload + "\n").encode("utf-8"))
