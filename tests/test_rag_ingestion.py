from __future__ import annotations

from about_llm.rag import SourceDocument, plan_incremental_update, split_markdown


def source(text: str, *, version: str = "v1") -> SourceDocument:
    return SourceDocument(
        source_id="handbook",
        tenant_id="tenant-a",
        version=version,
        text=text,
        acl=("engineering",),
        metadata={"uri": "kb://handbook"},
    )


def test_structure_split_preserves_security_and_source_metadata() -> None:
    chunks = split_markdown(source("# RAG\n\nIntro.\n\n## ACL\n\nFilter before ranking."))

    assert [chunk.heading_path for chunk in chunks] == [("RAG",), ("RAG", "ACL")]
    assert all(chunk.tenant_id == "tenant-a" for chunk in chunks)
    assert all(chunk.acl == ("engineering",) for chunk in chunks)
    assert all(chunk.metadata["uri"] == "kb://handbook" for chunk in chunks)


def test_unrelated_insertion_does_not_renumber_stable_chunk_ids() -> None:
    before = split_markdown(source("# Guide\n\nAlpha.\n\nOmega."))
    after = split_markdown(source("# Guide\n\nNew.\n\nAlpha.\n\nOmega."))

    before_by_text = {chunk.text: chunk.chunk_id for chunk in before}
    after_by_text = {chunk.text: chunk.chunk_id for chunk in after}
    assert before_by_text["Alpha."] == after_by_text["Alpha."]
    assert before_by_text["Omega."] == after_by_text["Omega."]


def test_incremental_plan_reports_edits_deletes_and_version_updates() -> None:
    before = split_markdown(source("# Guide\n\nKeep.\n\nEdit me.\n\nDelete me."))
    after = split_markdown(source("# Guide\n\nKeep.\n\nEdited.", version="v2"))
    plan = plan_incremental_update(before, after)

    assert {chunk.text for chunk in plan.upsert} == {"Keep.", "Edited."}
    assert len(plan.delete_chunk_ids) == 2
    assert plan.unchanged_chunk_ids == ()


def test_oversized_paragraph_is_bounded_without_loss() -> None:
    original = "a" * 150
    chunks = split_markdown(source(original), max_chars=64)

    assert all(len(chunk.text) <= 64 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == original
