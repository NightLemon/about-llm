"""Deterministic, authorization-preserving citation context packing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from about_llm.rag.citations import CitationContext, build_citation_context
from about_llm.rag.models import Document, SearchResult

ContextCost = Callable[[str], int]
ChatTokenize = Callable[[tuple[Mapping[str, str], ...]], Iterable[int]]


class PackingReason(str, Enum):
    """Auditable terminal decision for one ranked evidence candidate."""

    SELECTED = "selected"
    DUPLICATE_DOCUMENT = "duplicate_document"
    SOURCE_QUOTA = "source_quota"
    BUDGET = "budget"


@dataclass(frozen=True)
class PackingDecision:
    """Why one ranked document was selected or dropped."""

    document_id: str
    stable_source_id: str
    rank: int
    selected: bool
    reason: PackingReason
    cost_if_selected_units: int | None

    def __post_init__(self) -> None:
        if not self.document_id.strip() or not self.stable_source_id.strip():
            raise ValueError("packing decision ids cannot be empty")
        if self.rank <= 0:
            raise ValueError("packing decision rank must be positive")
        if self.selected is not (self.reason is PackingReason.SELECTED):
            raise ValueError("selected flag must agree with packing reason")
        measured_reasons = {PackingReason.SELECTED, PackingReason.BUDGET}
        if self.reason in measured_reasons and self.cost_if_selected_units is None:
            raise ValueError("selected/budget decisions need prospective cost")
        if self.reason not in measured_reasons and self.cost_if_selected_units is not None:
            raise ValueError("duplicate/quota decisions cannot claim a prospective cost")
        if (
            self.cost_if_selected_units is not None
            and self.cost_if_selected_units < 0
        ):
            raise ValueError("prospective cost cannot be negative")


@dataclass(frozen=True)
class PackedCitationContext:
    """Packed context plus the exact accounting and decision ledger."""

    context: CitationContext
    budget_units: int
    base_cost_units: int
    used_cost_units: int
    cost_unit: str
    max_chunks_per_source: int
    decisions: tuple[PackingDecision, ...]

    def __post_init__(self) -> None:
        integer_fields = (
            self.budget_units,
            self.base_cost_units,
            self.used_cost_units,
            self.max_chunks_per_source,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("packing budget, costs, and quota must be integers")
        if self.budget_units < 0 or self.base_cost_units < 0:
            raise ValueError("budget and base cost cannot be negative")
        if not 0 <= self.used_cost_units <= self.budget_units:
            raise ValueError("used cost must be within the packing budget")
        if not isinstance(self.cost_unit, str) or not self.cost_unit.strip():
            raise ValueError("cost_unit cannot be empty")
        if self.max_chunks_per_source <= 0:
            raise ValueError("max_chunks_per_source must be positive")
        context_document_ids = tuple(
            document.document_id for document in self.context.sources.values()
        )
        if self.selected_document_ids != context_document_ids:
            raise ValueError("selected decisions must match citation context documents")

    @property
    def selected_document_ids(self) -> tuple[str, ...]:
        return tuple(decision.document_id for decision in self.decisions if decision.selected)

    @property
    def dropped_document_ids(self) -> tuple[str, ...]:
        return tuple(decision.document_id for decision in self.decisions if not decision.selected)


def pack_citation_context(
    results: Iterable[SearchResult],
    *,
    tenant_id: str,
    budget_units: int,
    cost_fn: ContextCost,
    cost_unit: str,
    principals: Iterable[str] = (),
    max_chunks_per_source: int = 2,
    prefix: str = "S",
) -> PackedCitationContext:
    """Greedily pack ranked evidence while measuring each full candidate context.

    ``cost_fn`` receives the complete rendered context after each proposed
    selection. A production caller can close over the fixed system/query/output
    template and tokenize the full prospective prompt, avoiding the false
    assumption that separately tokenized component lengths are additive.

    Every candidate is authorization-checked before budget or quota decisions,
    including candidates that would otherwise be dropped.
    """
    if isinstance(budget_units, bool) or not isinstance(budget_units, int):
        raise TypeError("budget_units must be an integer")
    if budget_units < 0:
        raise ValueError("budget_units cannot be negative")
    if isinstance(max_chunks_per_source, bool) or not isinstance(
        max_chunks_per_source, int
    ):
        raise TypeError("max_chunks_per_source must be an integer")
    if max_chunks_per_source <= 0:
        raise ValueError("max_chunks_per_source must be positive")
    if not cost_unit.strip():
        raise ValueError("cost_unit cannot be empty")

    candidates = tuple(results)
    principals_tuple = tuple(principals)
    _validate_rank_order(candidates)
    # Validate every candidate before selection. Otherwise an unauthorized
    # result hidden behind a quota/budget drop could escape the security gate.
    for candidate in candidates:
        build_citation_context(
            (candidate,),
            tenant_id=tenant_id,
            principals=principals_tuple,
            prefix=prefix,
        )

    current_context = build_citation_context(
        (),
        tenant_id=tenant_id,
        principals=principals_tuple,
        prefix=prefix,
    )
    base_cost = _measure_cost(cost_fn, current_context.rendered)
    current_cost = base_cost
    selected: list[SearchResult] = []
    decisions: list[PackingDecision] = []
    seen_document_ids: set[str] = set()
    selected_per_source: dict[str, int] = {}

    for result in candidates:
        document = result.document
        stable_source_id = _stable_source_id(document)
        if document.document_id in seen_document_ids:
            decisions.append(
                PackingDecision(
                    document_id=document.document_id,
                    stable_source_id=stable_source_id,
                    rank=result.rank,
                    selected=False,
                    reason=PackingReason.DUPLICATE_DOCUMENT,
                    cost_if_selected_units=None,
                )
            )
            continue
        seen_document_ids.add(document.document_id)
        if selected_per_source.get(stable_source_id, 0) >= max_chunks_per_source:
            decisions.append(
                PackingDecision(
                    document_id=document.document_id,
                    stable_source_id=stable_source_id,
                    rank=result.rank,
                    selected=False,
                    reason=PackingReason.SOURCE_QUOTA,
                    cost_if_selected_units=None,
                )
            )
            continue

        prospective = build_citation_context(
            [*selected, result],
            tenant_id=tenant_id,
            principals=principals_tuple,
            prefix=prefix,
        )
        prospective_cost = _measure_cost(cost_fn, prospective.rendered)
        if prospective_cost > budget_units:
            decisions.append(
                PackingDecision(
                    document_id=document.document_id,
                    stable_source_id=stable_source_id,
                    rank=result.rank,
                    selected=False,
                    reason=PackingReason.BUDGET,
                    cost_if_selected_units=prospective_cost,
                )
            )
            continue
        selected.append(result)
        selected_per_source[stable_source_id] = (
            selected_per_source.get(stable_source_id, 0) + 1
        )
        current_context = prospective
        current_cost = prospective_cost
        decisions.append(
            PackingDecision(
                document_id=document.document_id,
                stable_source_id=stable_source_id,
                rank=result.rank,
                selected=True,
                reason=PackingReason.SELECTED,
                cost_if_selected_units=prospective_cost,
            )
        )

    if current_cost > budget_units:
        raise ValueError(
            f"no packed context fits budget {budget_units} {cost_unit}; "
            f"empty-context base cost is {base_cost}"
        )
    return PackedCitationContext(
        context=current_context,
        budget_units=budget_units,
        base_cost_units=base_cost,
        used_cost_units=current_cost,
        cost_unit=cost_unit,
        max_chunks_per_source=max_chunks_per_source,
        decisions=tuple(decisions),
    )


def utf8_byte_length(text: str) -> int:
    """Return serialized UTF-8 bytes; this is deliberately not a token count."""
    return len(text.encode("utf-8"))


def make_rag_chat_prompt_cost(
    *,
    system_prompt: str,
    query: str,
    user_prompt_template: str,
    tokenize_messages: ChatTokenize,
    reserved_output_tokens: int,
) -> ContextCost:
    """Build a full-chat token cost for prospective RAG contexts.

    The returned cost re-renders and tokenizes the complete system/user chat
    for every candidate context, then adds an explicit output reservation.
    It deliberately does not add separately tokenized component lengths.
    """

    for name, value in (
        ("system_prompt", system_prompt),
        ("query", query),
        ("user_prompt_template", user_prompt_template),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if user_prompt_template.count("{query}") != 1:
        raise ValueError("user_prompt_template must contain {query} exactly once")
    if user_prompt_template.count("{context}") != 1:
        raise ValueError("user_prompt_template must contain {context} exactly once")
    if isinstance(reserved_output_tokens, bool) or not isinstance(
        reserved_output_tokens, int
    ):
        raise TypeError("reserved_output_tokens must be an integer")
    if reserved_output_tokens < 0:
        raise ValueError("reserved_output_tokens cannot be negative")

    def full_prompt_cost(context: str) -> int:
        user_prompt = _render_rag_user_prompt(
            user_prompt_template, query=query, context=context
        )
        messages: tuple[Mapping[str, str], ...] = (
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        )
        token_ids = tuple(tokenize_messages(messages))
        if not token_ids:
            raise ValueError("chat tokenizer returned no token ids")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in token_ids
        ):
            raise ValueError("chat tokenizer must return non-negative integer token ids")
        return len(token_ids) + reserved_output_tokens

    return full_prompt_cost


def _render_rag_user_prompt(template: str, *, query: str, context: str) -> str:
    """Substitute original template slots without reinterpreting inserted text."""

    replacements = {"{query}": query, "{context}": context}
    positions = sorted((template.index(marker), marker) for marker in replacements)
    rendered: list[str] = []
    cursor = 0
    for position, marker in positions:
        rendered.append(template[cursor:position])
        rendered.append(replacements[marker])
        cursor = position + len(marker)
    rendered.append(template[cursor:])
    return "".join(rendered)


def _measure_cost(cost_fn: ContextCost, rendered_context: str) -> int:
    value = cost_fn(rendered_context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("cost_fn must return an integer")
    if value < 0:
        raise ValueError("cost_fn cannot return a negative value")
    return value


def _validate_rank_order(results: tuple[SearchResult, ...]) -> None:
    ranks = [result.rank for result in results]
    if any(left >= right for left, right in pairwise(ranks)):
        raise ValueError("results must be supplied in strictly increasing rank order")


def _stable_source_id(document: Document) -> str:
    source_id = document.metadata.get("source_id")
    return source_id if isinstance(source_id, str) and source_id else document.document_id
