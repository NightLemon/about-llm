"""追踪一份带版本的 RAG 文档如何切块并增量更新到 SQLite。

实验从 sample corpus 读取 v1，构造包含“新增一段 + 修改一句”的 v2，然后比较 chunk identity、
upsert/delete 计划、乐观并发控制和 ACL 可见结果。它覆盖 ingestion，不执行 embedding 或检索。
"""

# ruff: noqa: RUF001 -- Full-width punctuation is part of the Chinese source text.

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from about_llm.rag import (
    SourceChunk,
    SourceDocument,
    SQLiteChunkStore,
    plan_incremental_update,
    split_markdown,
)
from about_llm.rag.cli import load_corpus

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "projects" / "rag-foundations" / "sample_corpus.jsonl"
SOURCE_ID = "rag-security"
ORIGINAL_SENTENCE = (
    "检索必须先执行租户隔离和 ACL 权限过滤，再进行排序与上下文构建。"
)
UPDATED_SENTENCE = (
    "检索必须先执行租户隔离和 ACL 权限过滤，再进行候选召回、排序与上下文构建。"
)
INSERTED_PARAGRAPH = "这份指南说明在线检索的权限边界。"
PRESERVED_PARAGRAPH = "生成器只能看到已授权证据；引用编号不能证明语义蕴含。"


def _chunk_rows(chunks: tuple[SourceChunk, ...]) -> list[dict[str, Any]]:
    """把 chunk 对象转换成只含教学所需字段的可读列表。"""

    return [
        {
            "chunk_id": chunk.chunk_id,
            "ordinal": chunk.ordinal,
            "heading_path": list(chunk.heading_path),
            "text": chunk.text,
            "source_version": chunk.source_version,
            "acl": list(chunk.acl),
        }
        for chunk in chunks
    ]


def _updated_source(source: SourceDocument) -> SourceDocument:
    """在 v1 中插入一段并编辑一句，构造确定性的 v2 文档。"""

    # 先确认固定样例没有漂移，否则字符串 replace 可能悄悄修改错误位置。
    heading_prefix = "# RAG 安全\n\n"
    if heading_prefix not in source.text or ORIGINAL_SENTENCE not in source.text:
        raise ValueError("rag-security sample text no longer matches the walkthrough")
    text = source.text.replace(
        heading_prefix,
        heading_prefix + INSERTED_PARAGRAPH + "\n\n",
        1,
    ).replace(ORIGINAL_SENTENCE, UPDATED_SENTENCE, 1)
    return SourceDocument(
        source_id=source.source_id,
        tenant_id=source.tenant_id,
        version="2",
        text=text,
        acl=source.acl,
        metadata=dict(source.metadata),
    )


def _chunk_for_text(chunks: tuple[SourceChunk, ...], text: str) -> SourceChunk:
    """按完整文本找到唯一 chunk，用于比较内容变化前后的 ID。"""

    matches = [chunk for chunk in chunks if chunk.text == text]
    if len(matches) != 1:
        raise ValueError(f"expected one chunk for {text!r}, found {len(matches)}")
    return matches[0]


def build_walkthrough() -> dict[str, Any]:
    """完成加载、切块、增量规划、事务更新和权限读取。"""

    # 从 corpus 中精确选择教学文档，避免其他样例参与当前实验。
    sources = load_corpus(CORPUS)
    matching = [source for source in sources if source.source_id == SOURCE_ID]
    if len(matching) != 1:
        raise ValueError(f"expected one {SOURCE_ID!r} source, found {len(matching)}")
    version_1 = matching[0]
    version_2 = _updated_source(version_1)

    # 两个版本使用相同切块器；plan 只需处理真正新增、修改或删除的 chunk。
    chunks_1 = tuple(split_markdown(version_1))
    chunks_2 = tuple(split_markdown(version_2))
    plan = plan_incremental_update(chunks_1, chunks_2)

    old_by_id = {chunk.chunk_id: chunk for chunk in chunks_1}
    new_by_id = {chunk.chunk_id: chunk for chunk in chunks_2}
    # 内容寻址 ID 相同但 payload 不同的情况也必须 upsert，例如 source_version 更新。
    same_ids = old_by_id.keys() & new_by_id.keys()
    payload_updates = sorted(
        chunk_id for chunk_id in same_ids if old_by_id[chunk_id] != new_by_id[chunk_id]
    )

    preserved_1 = _chunk_for_text(chunks_1, PRESERVED_PARAGRAPH)
    preserved_2 = _chunk_for_text(chunks_2, PRESERVED_PARAGRAPH)
    edited_1 = _chunk_for_text(chunks_1, ORIGINAL_SENTENCE)
    edited_2 = _chunk_for_text(chunks_2, UPDATED_SENTENCE)
    inserted = _chunk_for_text(chunks_2, INSERTED_PARAGRAPH)

    # 使用临时真实 SQLite 文件执行事务，而不是在内存里伪造结果。
    with tempfile.TemporaryDirectory(prefix="about-llm-rag-ingestion-") as directory:
        database = Path(directory) / "rag.db"
        with SQLiteChunkStore(database) as store:
            initial = store.upsert_source(
                version_1,
                expected_current_version=None,
            )
            update = store.upsert_source(
                version_2,
                expected_current_version="1",
            )
            # 用旧 expected_current_version 再写 v1，模拟过期 worker 的 stale update。
            try:
                store.upsert_source(
                    version_1,
                    expected_current_version="1",
                )
            except ValueError as error:
                stale_update = {"rejected": True, "message": str(error)}
            else:
                raise AssertionError("stale update unexpectedly succeeded")
            # 同一 tenant 下，匿名主体与 engineering 主体应看到不同 ACL 集合。
            anonymous = store.visible_chunks(tenant_id=version_2.tenant_id)
            engineering = store.visible_chunks(
                tenant_id=version_2.tenant_id,
                principals=("engineering",),
            )
            current_version = store.current_version(
                tenant_id=version_2.tenant_id,
                source_id=version_2.source_id,
            )

    return {
        "walkthrough_version": "about-llm.rag-ingestion-walkthrough.v1",
        "source": {
            "source_id": version_1.source_id,
            "tenant_id": version_1.tenant_id,
            "acl": list(version_1.acl),
            "metadata": dict(version_1.metadata),
        },
        "version_1": {
            "source_version": version_1.version,
            "chunk_count": len(chunks_1),
            "chunks": _chunk_rows(chunks_1),
        },
        "version_2": {
            "source_version": version_2.version,
            "changes": {
                "inserted_paragraph": INSERTED_PARAGRAPH,
                "edited_from": ORIGINAL_SENTENCE,
                "edited_to": UPDATED_SENTENCE,
            },
            "chunk_count": len(chunks_2),
            "chunks": _chunk_rows(chunks_2),
        },
        "identity": {
            "preserved_text": PRESERVED_PARAGRAPH,
            "preserved_chunk_id": preserved_1.chunk_id,
            "preserved_id_across_versions": preserved_1.chunk_id == preserved_2.chunk_id,
            "edited_old_chunk_id": edited_1.chunk_id,
            "edited_new_chunk_id": edited_2.chunk_id,
            "edited_content_gets_new_id": edited_1.chunk_id != edited_2.chunk_id,
            "inserted_chunk_id": inserted.chunk_id,
        },
        "incremental_plan": {
            "upsert_chunk_ids": [chunk.chunk_id for chunk in plan.upsert],
            "new_identity_chunk_ids": sorted(new_by_id.keys() - old_by_id.keys()),
            "same_identity_payload_updates": payload_updates,
            "delete_chunk_ids": list(plan.delete_chunk_ids),
            "unchanged_chunk_ids": list(plan.unchanged_chunk_ids),
        },
        "sqlite": {
            "initial_upsert_count": len(initial.upsert),
            "update_upsert_count": len(update.upsert),
            "update_delete_count": len(update.delete_chunk_ids),
            "current_version": current_version,
            "anonymous_visible_chunk_ids": [chunk.chunk_id for chunk in anonymous],
            "engineering_visible_chunk_ids": [chunk.chunk_id for chunk in engineering],
            "stale_update": stale_update,
        },
        "scope": {
            "sample_corpus_loaded": True,
            "markdown_split_executed": True,
            "incremental_plan_executed": True,
            "sqlite_transactions_executed": True,
            "acl_filter_executed": True,
            "embedding_or_search_index_executed": False,
            "external_source_authenticity_verified": False,
            "multi_store_production_update_verified": False,
        },
    }


def main() -> int:
    """运行 ingestion 导览并以 UTF-8 JSON 输出全部阶段。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rendered = json.dumps(build_walkthrough(), ensure_ascii=False, indent=2) + "\n"
    sys.stdout.buffer.write(rendered.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
