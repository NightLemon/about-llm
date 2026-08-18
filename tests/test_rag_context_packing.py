from __future__ import annotations

from collections.abc import Callable

import pytest

from about_llm.rag import (
    Document,
    PackingReason,
    SearchResult,
    build_citation_context,
    make_rag_chat_prompt_cost,
    pack_citation_context,
    utf8_byte_length,
)

pytestmark = [pytest.mark.contract, pytest.mark.security]


def result(
    document_id: str,
    text: str,
    rank: int,
    *,
    source_id: str | None = None,
    tenant_id: str = "t1",
    acl: tuple[str, ...] = (),
) -> SearchResult:
    metadata = {"source_id": source_id} if source_id is not None else {}
    return SearchResult(
        document=Document(document_id, text, tenant_id, metadata=metadata, acl=acl),
        score=1 / rank,
        rank=rank,
        source="test",
    )


def rendered_cost(results: list[SearchResult]) -> int:
    context = build_citation_context(results, tenant_id="t1")
    return utf8_byte_length(context.rendered)


def test_packer_skips_oversized_candidate_and_can_select_a_later_one() -> None:
    first = result("d1", "short evidence", 1, source_id="source-a")
    oversized = result("d2", "x" * 2000, 2, source_id="source-b")
    later = result("d3", "small", 3, source_id="source-c")
    budget = rendered_cost([first, later])

    packed = pack_citation_context(
        [first, oversized, later],
        tenant_id="t1",
        budget_units=budget,
        cost_fn=utf8_byte_length,
        cost_unit="utf8_bytes",
    )

    assert packed.selected_document_ids == ("d1", "d3")
    assert packed.dropped_document_ids == ("d2",)
    assert packed.used_cost_units == budget
    assert packed.base_cost_units == 0
    assert list(packed.context.sources) == ["S1", "S2"]
    assert [decision.reason for decision in packed.decisions] == [
        PackingReason.SELECTED,
        PackingReason.BUDGET,
        PackingReason.SELECTED,
    ]
    assert packed.decisions[1].cost_if_selected_units is not None
    assert packed.decisions[1].cost_if_selected_units > budget


def test_source_quota_uses_stable_source_id_not_chunk_document_id() -> None:
    first = result("chunk-a1", "one", 1, source_id="source-a")
    same_source = result("chunk-a2", "two", 2, source_id="source-a")
    other_source = result("chunk-b1", "three", 3, source_id="source-b")

    packed = pack_citation_context(
        [first, same_source, other_source],
        tenant_id="t1",
        budget_units=10_000,
        cost_fn=utf8_byte_length,
        cost_unit="utf8_bytes",
        max_chunks_per_source=1,
    )

    assert packed.selected_document_ids == ("chunk-a1", "chunk-b1")
    assert packed.decisions[1].reason is PackingReason.SOURCE_QUOTA
    assert packed.decisions[1].cost_if_selected_units is None


def test_duplicate_document_is_audited_and_never_consumes_another_slot() -> None:
    document = Document("same", "evidence", "t1")
    first = SearchResult(document, 1.0, 1, "first")
    duplicate = SearchResult(document, 0.5, 2, "second")

    packed = pack_citation_context(
        [first, duplicate],
        tenant_id="t1",
        budget_units=10_000,
        cost_fn=utf8_byte_length,
        cost_unit="utf8_bytes",
    )

    assert packed.selected_document_ids == ("same",)
    assert packed.decisions[1].reason is PackingReason.DUPLICATE_DOCUMENT


@pytest.mark.parametrize(
    "candidate,principals,error",
    [
        (result("cross", "secret", 1, tenant_id="t2"), (), "tenant"),
        (result("private", "secret", 1, acl=("eng",)), (), "not visible"),
    ],
)
def test_every_candidate_is_authorized_even_when_budget_would_drop_it(
    candidate: SearchResult,
    principals: tuple[str, ...],
    error: str,
) -> None:
    with pytest.raises(PermissionError, match=error):
        pack_citation_context(
            [candidate],
            tenant_id="t1",
            principals=principals,
            budget_units=0,
            cost_fn=utf8_byte_length,
            cost_unit="utf8_bytes",
        )


def test_authorized_principal_can_pack_restricted_candidate() -> None:
    private = result("private", "secret", 1, acl=("eng",))
    packed = pack_citation_context(
        [private],
        tenant_id="t1",
        principals=("eng",),
        budget_units=10_000,
        cost_fn=utf8_byte_length,
        cost_unit="utf8_bytes",
    )
    assert packed.selected_document_ids == ("private",)


def test_cost_function_can_measure_a_full_prompt_closure() -> None:
    candidate = result("d1", "evidence", 1)
    fixed_prompt_cost = 17

    def full_prompt_cost(context: str) -> int:
        return fixed_prompt_cost + len(context.encode("utf-8"))

    budget = full_prompt_cost(
        build_citation_context([candidate], tenant_id="t1").rendered
    )
    packed = pack_citation_context(
        [candidate],
        tenant_id="t1",
        budget_units=budget,
        cost_fn=full_prompt_cost,
        cost_unit="tokens:model@revision",
    )

    assert packed.base_cost_units == fixed_prompt_cost
    assert packed.used_cost_units == budget


def test_target_chat_cost_renders_full_prompt_and_reserves_output() -> None:
    observed: list[tuple[dict[str, str], ...]] = []

    def tokenize(messages: tuple[dict[str, str], ...]) -> list[int]:
        observed.append(messages)
        serialized = " ".join(message["content"] for message in messages)
        return list(range(len(serialized.split())))

    cost = make_rag_chat_prompt_cost(
        system_prompt="follow evidence only",
        query="what is RAG",
        user_prompt_template="question {query} evidence {context}",
        tokenize_messages=tokenize,
        reserved_output_tokens=7,
    )

    empty_cost = cost("")
    evidence_cost = cost("retrieved source text")

    assert empty_cost == 15
    assert evidence_cost == 18
    assert observed[-1][0] == {"role": "system", "content": "follow evidence only"}
    assert observed[-1][1]["content"] == (
        "question what is RAG evidence retrieved source text"
    )


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("{query} only", "{context}"),
        ("{context} only", "{query}"),
        ("{query} {query} {context}", "{query}"),
    ],
)
def test_target_chat_cost_requires_unambiguous_placeholders(
    template: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_rag_chat_prompt_cost(
            system_prompt="system",
            query="query",
            user_prompt_template=template,
            tokenize_messages=lambda _: [1],
            reserved_output_tokens=1,
        )


def test_target_chat_cost_rejects_invalid_token_ids() -> None:
    cost = make_rag_chat_prompt_cost(
        system_prompt="system",
        query="query",
        user_prompt_template="{query} {context}",
        tokenize_messages=lambda _: [True],  # type: ignore[list-item]
        reserved_output_tokens=1,
    )

    with pytest.raises(ValueError, match="non-negative integer"):
        cost("evidence")


def test_target_chat_cost_does_not_expand_placeholder_text_from_inputs() -> None:
    rendered_user: list[str] = []

    def tokenize(messages: tuple[dict[str, str], ...]) -> list[int]:
        rendered_user.append(messages[1]["content"])
        return [1]

    cost = make_rag_chat_prompt_cost(
        system_prompt="system",
        query="literal {context}",
        user_prompt_template="Q={query}; C={context}",
        tokenize_messages=tokenize,
        reserved_output_tokens=1,
    )

    assert cost("literal {query}") == 2
    assert rendered_user == ["Q=literal {context}; C=literal {query}"]


@pytest.mark.parametrize(
    "cost_fn,error_type",
    [
        (lambda _: True, TypeError),
        (lambda _: 1.5, TypeError),
        (lambda _: -1, ValueError),
    ],
)
def test_cost_function_must_return_a_non_negative_integer(
    cost_fn: Callable[[str], object], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        pack_citation_context(
            [],
            tenant_id="t1",
            budget_units=10,
            cost_fn=cost_fn,  # type: ignore[arg-type]
            cost_unit="test",
        )


def test_budget_must_cover_the_cost_of_an_empty_context() -> None:
    with pytest.raises(ValueError, match="empty-context base cost"):
        pack_citation_context(
            [],
            tenant_id="t1",
            budget_units=4,
            cost_fn=lambda context: 5 + len(context),
            cost_unit="full_prompt_units",
        )


def test_cost_is_not_assumed_monotonic_after_context_insertion() -> None:
    candidate = result("d1", "evidence", 1)

    def non_monotonic_cost(context: str) -> int:
        return 10 if not context else 5

    packed = pack_citation_context(
        [candidate],
        tenant_id="t1",
        budget_units=5,
        cost_fn=non_monotonic_cost,
        cost_unit="synthetic_token_boundary_example",
    )

    assert packed.base_cost_units == 10
    assert packed.used_cost_units == 5
    assert packed.selected_document_ids == ("d1",)


def test_rank_order_and_integer_configuration_are_strict() -> None:
    first = result("d1", "one", 2)
    second = result("d2", "two", 1)
    with pytest.raises(ValueError, match="strictly increasing rank"):
        pack_citation_context(
            [first, second],
            tenant_id="t1",
            budget_units=1000,
            cost_fn=utf8_byte_length,
            cost_unit="utf8_bytes",
        )
    with pytest.raises(TypeError, match="budget_units"):
        pack_citation_context(
            [],
            tenant_id="t1",
            budget_units=True,  # type: ignore[arg-type]
            cost_fn=utf8_byte_length,
            cost_unit="utf8_bytes",
        )
