from about_llm.rag import Document, SearchResult, audit_citations, build_citation_context


def result(document_id: str, tenant_id: str, rank: int) -> SearchResult:
    return SearchResult(
        Document(document_id, f"Evidence from {document_id}.", tenant_id),
        score=1 / rank,
        rank=rank,
        source="hybrid",
    )


def test_context_assigns_canonical_ids_and_deduplicates_documents() -> None:
    context = build_citation_context(
        [result("doc-b", "t1", 1), result("doc-b", "t1", 2), result("doc-a", "t1", 3)],
        tenant_id="t1",
    )

    assert list(context.sources) == ["S1", "S2"]
    assert context.sources["S1"].document_id == "doc-b"
    assert context.rendered.count('<source id="S1"') == 1


def test_context_rejects_cross_tenant_evidence() -> None:
    try:
        build_citation_context([result("secret", "t2", 1)], tenant_id="t1")
    except PermissionError as error:
        assert "tenant" in str(error)
    else:
        raise AssertionError("cross-tenant evidence should be rejected")


def test_context_rechecks_principal_acl() -> None:
    restricted = SearchResult(
        Document("restricted", "Evidence.", "t1", acl=("eng",)),
        score=1,
        rank=1,
        source="bm25",
    )

    try:
        build_citation_context([restricted], tenant_id="t1")
    except PermissionError as error:
        assert "not visible" in str(error)
    else:
        raise AssertionError("restricted evidence should be rejected")

    context = build_citation_context(
        [restricted], tenant_id="t1", principals=("eng",)
    )
    assert set(context.sources) == {"S1"}


def test_audit_reports_unknown_and_uncited_paragraphs() -> None:
    audit = audit_citations(
        "First supported claim. [S1]\n\nUnsupported syntax. [S9]\n\nNo citation here.",
        {"S1", "S2"},
    )

    assert audit.cited_source_ids == ("S1", "S9")
    assert audit.unknown_source_ids == ("S9",)
    assert audit.uncited_paragraphs == ("No citation here.",)
    assert audit.syntactically_valid is False


def test_syntax_success_does_not_claim_semantic_entailment() -> None:
    audit = audit_citations("The moon is cheese. [S1]", {"S1"})
    assert audit.syntactically_valid is True
