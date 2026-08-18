"""Real-weight, retrieval-to-generation RAG control with explicit failure evidence."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    VerifiedCheckpointSnapshot,
    download_checkpoint_snapshot,
    verify_checkpoint_snapshot,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes
from about_llm.rag.bm25 import (
    BM25_AUTHORIZED_STATISTICS_IMPLEMENTATION,
    BM25_LEGACY_GLOBAL_STATISTICS_IMPLEMENTATION,
    BM25Index,
)
from about_llm.rag.citations import audit_citations, build_citation_context
from about_llm.rag.context_packing import (
    make_rag_chat_prompt_cost,
    pack_citation_context,
)
from about_llm.rag.models import Document, SearchResult

RAG_TRANSFORMERS_CONTROL_VERSION = "about-llm.rag-transformers-control.v1"
RAG_TRANSFORMERS_REPORT_VERSION = "about-llm.rag-transformers-control-report.v1"
RAG_TRANSFORMERS_EVIDENCE_BOUNDARY = (
    "This control verifies selected bytes from one immutable checkpoint revision, "
    "then executes authorization-filtered BM25 retrieval, tokenizer-measured context "
    "packing, and greedy Transformers generation for two fixed cases on CPU FP32. It "
    "records model failures rather than repairing outputs. A local verifier checks "
    "citation syntax or exact abstention only; it does not prove semantic entailment. "
    "Unkeyed hashes do not authenticate the publisher or recorder, verification does "
    "not eliminate loader-reopen TOCTOU, and two authored cases do not establish "
    "general RAG quality, security, licensing, performance, or production readiness."
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_MAX_JSON_BYTES = 2_000_000
_MAX_CORPUS_DOCUMENTS = 32
_MAX_CASES = 8
_MAX_TEXT_CHARACTERS = 16_384
_MAX_GENERATED_TOKENS = 256

_MANIFEST_FIELDS = {
    "control_version",
    "checked_at",
    "checkpoint",
    "runtime",
    "generation",
    "prompts",
    "corpus",
    "cases",
    "evidence_boundary",
}
_CHECKPOINT_MANIFEST_FIELDS = {
    "manifest_fingerprint",
    "model_id",
    "revision",
    "expected_model_class",
    "expected_model_type",
}
_RUNTIME_MANIFEST_FIELDS = {
    "device",
    "dtype",
    "attention_implementation",
    "trust_remote_code",
}
_GENERATION_MANIFEST_FIELDS = {
    "do_sample",
    "use_cache",
    "max_new_tokens",
    "prompt_budget_tokens",
    "max_chunks_per_source",
}
_PROMPT_FIELDS = {
    "system_prompt",
    "user_prompt_template",
    "abstention_text",
}
_DOCUMENT_FIELDS = {
    "document_id",
    "stable_source_id",
    "source_version",
    "text",
    "tenant_id",
    "acl",
}
_CASE_FIELDS = {
    "case_id",
    "query",
    "tenant_id",
    "principals",
    "top_k",
    "expected_retrieved_document_ids",
    "expected_packed_document_ids",
    "expected_behavior",
}
_REPORT_FIELDS = {
    "report_version",
    "manifest_fingerprint",
    "checkpoint_manifest_fingerprint",
    "checked_at",
    "checkpoint",
    "runtime",
    "model",
    "tokenizer",
    "cases",
    "summary",
    "scope",
    "evidence_boundary",
    "report_fingerprint",
}
_REPORT_CHECKPOINT_FIELDS = {
    "model_id",
    "revision",
    "selected_file_count",
    "selected_total_bytes",
    "all_selected_file_bytes_verified_before_load",
    "loader_input",
}
_REPORT_RUNTIME_FIELDS = {
    "python_implementation",
    "python_version",
    "platform",
    "torch_version",
    "transformers_version",
    "device",
    "dtype",
    "attention_implementation",
    "torch_num_threads",
    "cuda_executed",
}
_REPORT_MODEL_FIELDS = {
    "class",
    "model_type",
    "total_parameters",
    "trainable_parameters",
    "parameter_storage_bytes",
    "parameter_dtypes",
    "eval_mode",
}
_REPORT_TOKENIZER_FIELDS = {
    "class",
    "vocabulary_size_with_added_tokens",
    "chat_template_sha256",
    "pad_token_id",
    "eos_token_id",
}
_REPORT_CASE_FIELDS = {
    "case_id",
    "query_sha256",
    "tenant_id",
    "principals",
    "expected_behavior",
    "retrieval",
    "packing",
    "prompt",
    "generation",
    "verification",
    "case_fingerprint",
}
_REPORT_RETRIEVAL_FIELDS = {
    "implementation",
    "top_k",
    "authorization_filtered_before_scoring",
    "document_ids",
    "results",
}
_REPORT_RETRIEVAL_RESULT_FIELDS = {"document_id", "rank", "score", "source"}
_REPORT_PACKING_FIELDS = {
    "budget_units",
    "base_cost_units",
    "used_cost_units",
    "cost_unit",
    "max_chunks_per_source",
    "document_ids",
    "source_short_ids",
    "rendered_context_sha256",
    "decisions",
}
_REPORT_DECISION_FIELDS = {
    "document_id",
    "stable_source_id",
    "rank",
    "selected",
    "reason",
    "cost_if_selected_units",
}
_REPORT_PROMPT_FIELDS = {
    "system_prompt_sha256",
    "user_prompt_template_sha256",
    "chat_template_sha256",
    "prompt_token_count",
    "prompt_token_ids_sha256",
}
_REPORT_GENERATION_FIELDS = {
    "do_sample",
    "use_cache",
    "max_new_tokens",
    "manual_greedy_token_ids",
    "generated_token_ids",
    "greedy_step_logits_sha256",
    "manual_greedy_matches_generate",
    "raw_output",
    "raw_output_sha256",
    "decoded_with_special_tokens",
    "generated_ended_with_eos",
    "stop_reason",
}
_REPORT_VERIFICATION_FIELDS = {
    "verifier",
    "valid_source_ids",
    "cited_source_ids",
    "unknown_source_ids",
    "uncited_paragraphs",
    "citation_syntax_passed",
    "abstention_exact_match",
    "expected_behavior_gate_passed",
}
_REPORT_SUMMARY_FIELDS = {
    "case_count",
    "expected_behavior_gate_passed_count",
    "all_expected_behavior_gates_passed",
}
_REPORT_SCOPE_FIELDS = {
    "target_checkpoint_weights_loaded",
    "selected_checkpoint_files_verified_before_load",
    "authorization_filtered_before_bm25_scoring",
    "tokenizer_measured_context_packing_executed",
    "manual_greedy_logits_executed",
    "framework_generate_executed",
    "model_failures_recorded_without_output_repair",
    "claim_evidence_entailment_verified",
    "general_rag_quality_proven",
    "publisher_or_recorder_authenticated_by_signature",
    "verification_to_loader_reopen_toctou_eliminated",
    "gpu_vllm_or_production_service_executed",
}


@dataclass(frozen=True)
class RAGControlDocument:
    document_id: str
    stable_source_id: str
    source_version: str
    text: str
    tenant_id: str
    acl: tuple[str, ...]

    def to_document(self) -> Document:
        return Document(
            document_id=self.document_id,
            text=self.text,
            tenant_id=self.tenant_id,
            metadata={
                "source_id": self.stable_source_id,
                "source_version": self.source_version,
            },
            acl=self.acl,
        )


@dataclass(frozen=True)
class RAGControlCase:
    case_id: str
    query: str
    tenant_id: str
    principals: tuple[str, ...]
    top_k: int
    expected_retrieved_document_ids: tuple[str, ...]
    expected_packed_document_ids: tuple[str, ...]
    expected_behavior: str


@dataclass(frozen=True)
class RAGTransformersControlSpec:
    checked_at: str
    checkpoint_manifest_fingerprint: str
    model_id: str
    revision: str
    expected_model_class: str
    expected_model_type: str
    device: str
    dtype: str
    attention_implementation: str
    max_new_tokens: int
    prompt_budget_tokens: int
    max_chunks_per_source: int
    system_prompt: str
    user_prompt_template: str
    abstention_text: str
    corpus: tuple[RAGControlDocument, ...]
    cases: tuple[RAGControlCase, ...]
    manifest_fingerprint: str


def load_rag_transformers_control_spec(
    path: Path,
    *,
    expected_control_version: str = RAG_TRANSFORMERS_CONTROL_VERSION,
    expected_evidence_boundary: str = RAG_TRANSFORMERS_EVIDENCE_BOUNDARY,
) -> RAGTransformersControlSpec:
    """Load a strict manifest for one reviewed retrieval-to-generation control."""

    manifest = _load_json_file(path)
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "manifest")
    if manifest.get("control_version") != expected_control_version:
        raise ValueError("manifest.control_version is unsupported")
    if manifest.get("evidence_boundary") != expected_evidence_boundary:
        raise ValueError("manifest.evidence_boundary drift")
    checked_at = _required_string(manifest, "checked_at", "manifest")
    try:
        date.fromisoformat(checked_at)
    except ValueError as error:
        raise ValueError("manifest.checked_at must be an ISO date") from error

    checkpoint = _mapping(manifest.get("checkpoint"), "manifest.checkpoint")
    _require_exact_fields(checkpoint, _CHECKPOINT_MANIFEST_FIELDS, "manifest.checkpoint")
    checkpoint_fingerprint = _sha256(
        checkpoint.get("manifest_fingerprint"),
        "manifest.checkpoint.manifest_fingerprint",
    )
    model_id = _required_string(checkpoint, "model_id", "manifest.checkpoint")
    revision = _required_string(checkpoint, "revision", "manifest.checkpoint")
    if _MODEL_ID.fullmatch(model_id) is None or _REVISION.fullmatch(revision) is None:
        raise ValueError("manifest.checkpoint model_id/revision format mismatch")
    expected_model_class = _required_string(
        checkpoint, "expected_model_class", "manifest.checkpoint"
    )
    expected_model_type = _required_string(
        checkpoint, "expected_model_type", "manifest.checkpoint"
    )

    runtime = _mapping(manifest.get("runtime"), "manifest.runtime")
    _require_exact_fields(runtime, _RUNTIME_MANIFEST_FIELDS, "manifest.runtime")
    device = _required_string(runtime, "device", "manifest.runtime")
    dtype = _required_string(runtime, "dtype", "manifest.runtime")
    attention = _required_string(
        runtime, "attention_implementation", "manifest.runtime"
    )
    if (device, dtype, attention) != ("cpu", "float32", "eager"):
        raise ValueError("manifest.runtime must be cpu/float32/eager")
    _require_boolean(runtime, "trust_remote_code", "manifest.runtime", expected=False)

    generation = _mapping(manifest.get("generation"), "manifest.generation")
    _require_exact_fields(
        generation, _GENERATION_MANIFEST_FIELDS, "manifest.generation"
    )
    _require_boolean(generation, "do_sample", "manifest.generation", expected=False)
    _require_boolean(generation, "use_cache", "manifest.generation", expected=True)
    max_new_tokens = _positive_integer(
        generation.get("max_new_tokens"), "manifest.generation.max_new_tokens"
    )
    if max_new_tokens > _MAX_GENERATED_TOKENS:
        raise ValueError("manifest.generation.max_new_tokens exceeds the limit")
    prompt_budget_tokens = _positive_integer(
        generation.get("prompt_budget_tokens"),
        "manifest.generation.prompt_budget_tokens",
    )
    max_chunks = _positive_integer(
        generation.get("max_chunks_per_source"),
        "manifest.generation.max_chunks_per_source",
    )
    if max_new_tokens >= prompt_budget_tokens:
        raise ValueError("manifest generation reservation must fit prompt budget")

    prompts = _mapping(manifest.get("prompts"), "manifest.prompts")
    _require_exact_fields(prompts, _PROMPT_FIELDS, "manifest.prompts")
    system_prompt = _bounded_string(prompts, "system_prompt", "manifest.prompts")
    user_prompt_template = _bounded_string(
        prompts, "user_prompt_template", "manifest.prompts"
    )
    abstention_text = _bounded_string(
        prompts, "abstention_text", "manifest.prompts"
    )
    if user_prompt_template.count("{query}") != 1:
        raise ValueError("manifest user prompt must contain {query} exactly once")
    if user_prompt_template.count("{context}") != 1:
        raise ValueError("manifest user prompt must contain {context} exactly once")

    raw_corpus = _bounded_array(
        manifest.get("corpus"),
        "manifest.corpus",
        maximum=_MAX_CORPUS_DOCUMENTS,
    )
    corpus: list[RAGControlDocument] = []
    document_ids: set[str] = set()
    for index, raw_document in enumerate(raw_corpus):
        location = f"manifest.corpus[{index}]"
        value = _mapping(raw_document, location)
        _require_exact_fields(value, _DOCUMENT_FIELDS, location)
        document_id = _identifier(value.get("document_id"), f"{location}.document_id")
        if document_id in document_ids:
            raise ValueError(f"{location}.document_id is duplicated")
        document_ids.add(document_id)
        corpus.append(
            RAGControlDocument(
                document_id=document_id,
                stable_source_id=_identifier(
                    value.get("stable_source_id"), f"{location}.stable_source_id"
                ),
                source_version=_identifier(
                    value.get("source_version"), f"{location}.source_version"
                ),
                text=_bounded_string(value, "text", location),
                tenant_id=_identifier(
                    value.get("tenant_id"), f"{location}.tenant_id"
                ),
                acl=tuple(
                    _unique_string_array(
                        value.get("acl"), f"{location}.acl", allow_empty=True
                    )
                ),
            )
        )

    raw_cases = _bounded_array(
        manifest.get("cases"), "manifest.cases", maximum=_MAX_CASES
    )
    cases: list[RAGControlCase] = []
    case_ids: set[str] = set()
    behavior_seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        location = f"manifest.cases[{index}]"
        value = _mapping(raw_case, location)
        _require_exact_fields(value, _CASE_FIELDS, location)
        case_id = _identifier(value.get("case_id"), f"{location}.case_id")
        if case_id in case_ids:
            raise ValueError(f"{location}.case_id is duplicated")
        case_ids.add(case_id)
        expected_behavior = _required_string(value, "expected_behavior", location)
        if expected_behavior not in {"answer_with_citations", "abstain"}:
            raise ValueError(f"{location}.expected_behavior is unsupported")
        behavior_seen.add(expected_behavior)
        retrieved = tuple(
            _unique_string_array(
                value.get("expected_retrieved_document_ids"),
                f"{location}.expected_retrieved_document_ids",
                allow_empty=True,
            )
        )
        packed = tuple(
            _unique_string_array(
                value.get("expected_packed_document_ids"),
                f"{location}.expected_packed_document_ids",
                allow_empty=True,
            )
        )
        if any(document_id not in document_ids for document_id in (*retrieved, *packed)):
            raise ValueError(f"{location} references an unknown corpus document")
        if any(document_id not in retrieved for document_id in packed):
            raise ValueError(f"{location} packed documents must be retrieved")
        cases.append(
            RAGControlCase(
                case_id=case_id,
                query=_bounded_string(value, "query", location),
                tenant_id=_identifier(
                    value.get("tenant_id"), f"{location}.tenant_id"
                ),
                principals=tuple(
                    _unique_string_array(
                        value.get("principals"),
                        f"{location}.principals",
                        allow_empty=True,
                    )
                ),
                top_k=_positive_integer(value.get("top_k"), f"{location}.top_k"),
                expected_retrieved_document_ids=retrieved,
                expected_packed_document_ids=packed,
                expected_behavior=expected_behavior,
            )
        )
    if behavior_seen != {"answer_with_citations", "abstain"}:
        raise ValueError("manifest cases must include answer-with-citations and abstain")

    return RAGTransformersControlSpec(
        checked_at=checked_at,
        checkpoint_manifest_fingerprint=checkpoint_fingerprint,
        model_id=model_id,
        revision=revision,
        expected_model_class=expected_model_class,
        expected_model_type=expected_model_type,
        device=device,
        dtype=dtype,
        attention_implementation=attention,
        max_new_tokens=max_new_tokens,
        prompt_budget_tokens=prompt_budget_tokens,
        max_chunks_per_source=max_chunks,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        abstention_text=abstention_text,
        corpus=tuple(corpus),
        cases=tuple(cases),
        manifest_fingerprint=_canonical_sha256(manifest),
    )


def validate_checkpoint_binding(
    spec: RAGTransformersControlSpec, checkpoint_spec: CheckpointControlSpec
) -> None:
    """Fail before loading if the RAG manifest does not bind the checkpoint spec."""

    observed = (
        checkpoint_spec.manifest_fingerprint,
        checkpoint_spec.model_id,
        checkpoint_spec.revision,
        checkpoint_spec.expected_model_class,
        checkpoint_spec.expected_model_type,
        checkpoint_spec.device,
        checkpoint_spec.dtype,
        checkpoint_spec.attention_implementation,
    )
    expected = (
        spec.checkpoint_manifest_fingerprint,
        spec.model_id,
        spec.revision,
        spec.expected_model_class,
        spec.expected_model_type,
        spec.device,
        spec.dtype,
        spec.attention_implementation,
    )
    if observed != expected:
        raise ValueError("RAG control and checkpoint manifest identity/runtime mismatch")


def execute_loaded_rag_transformers_control(
    spec: RAGTransformersControlSpec,
    *,
    checkpoint_spec: CheckpointControlSpec,
    snapshot: VerifiedCheckpointSnapshot,
    model: Any,
    tokenizer: Any,
) -> dict[str, object]:
    """Execute fixed retrieval, packing, manual greedy logits, and generate paths."""

    try:
        import torch
        import transformers
        from transformers import GenerationConfig
    except ImportError as error:  # pragma: no cover - environment-specific dependency
        raise RuntimeError("torch and transformers are required for RAG control") from error

    validate_checkpoint_binding(spec, checkpoint_spec)
    if type(model).__name__ != spec.expected_model_class:
        raise ValueError("loaded model class does not match RAG control")
    if getattr(model.config, "model_type", None) != spec.expected_model_type:
        raise ValueError("loaded model_type does not match RAG control")
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("RAG control tokenizer must provide a chat template")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError("RAG control tokenizer must provide an integer EOS id")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise ValueError("RAG control tokenizer must provide an integer PAD id")
    if not 0 <= eos_token_id < len(tokenizer) or not 0 <= pad_token_id < len(tokenizer):
        raise ValueError("RAG control tokenizer special id is outside its vocabulary")

    model.to("cpu")
    model.requires_grad_(False)
    model.eval()
    documents = tuple(item.to_document() for item in spec.corpus)
    index = BM25Index(documents)
    case_reports: list[dict[str, object]] = []
    for case in spec.cases:
        results = index.search(
            case.query,
            tenant_id=case.tenant_id,
            principals=case.principals,
            top_k=case.top_k,
        )
        retrieved_ids = tuple(result.document.document_id for result in results)
        if retrieved_ids != case.expected_retrieved_document_ids:
            raise RuntimeError(
                f"case {case.case_id}: reviewed BM25 result identity drift: {retrieved_ids}"
            )

        def tokenize_messages(messages: tuple[Mapping[str, str], ...]) -> Sequence[int]:
            return _tokenize_chat(tokenizer, messages)

        cost_fn = make_rag_chat_prompt_cost(
            system_prompt=spec.system_prompt,
            query=case.query,
            user_prompt_template=spec.user_prompt_template,
            tokenize_messages=tokenize_messages,
            reserved_output_tokens=spec.max_new_tokens,
        )
        packed = pack_citation_context(
            results,
            tenant_id=case.tenant_id,
            principals=case.principals,
            budget_units=spec.prompt_budget_tokens,
            cost_fn=cost_fn,
            cost_unit="chat_tokens_including_output_reservation",
            max_chunks_per_source=spec.max_chunks_per_source,
        )
        if packed.selected_document_ids != case.expected_packed_document_ids:
            raise RuntimeError(
                f"case {case.case_id}: reviewed packed document identity drift"
            )
        user_prompt = _render_user_prompt(
            spec.user_prompt_template,
            query=case.query,
            context=packed.context.rendered,
        )
        messages: tuple[Mapping[str, str], ...] = (
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": user_prompt},
        )
        prompt_ids = _tokenize_chat(tokenizer, messages)
        if len(prompt_ids) + spec.max_new_tokens != packed.used_cost_units:
            raise RuntimeError("packing cost and final prompt tokenization disagree")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
        attention_mask = torch.ones_like(input_ids)
        if torch.any(input_ids < 0) or torch.any(input_ids >= len(tokenizer)):
            raise RuntimeError("RAG chat template emitted an out-of-vocabulary id")

        manual_ids: list[int] = []
        step_hashes: list[str] = []
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            if output.past_key_values is None:
                raise RuntimeError("RAG prefill did not return a KV cache")
            next_logits = output.logits[:, -1, :].to(dtype=torch.float32)
            cache = output.past_key_values
            for _ in range(spec.max_new_tokens):
                if next_logits.ndim != 2 or next_logits.shape[0] != 1:
                    raise RuntimeError("RAG greedy logits have an invalid shape")
                step_hashes.append(_tensor_sha256(next_logits))
                next_id = int(torch.argmax(next_logits, dim=-1).item())
                manual_ids.append(next_id)
                if next_id == eos_token_id:
                    break
                token = torch.tensor([[next_id]], dtype=torch.long, device="cpu")
                cached_mask = torch.ones(
                    (1, len(prompt_ids) + len(manual_ids)),
                    dtype=torch.long,
                    device="cpu",
                )
                output = model(
                    input_ids=token,
                    attention_mask=cached_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                if output.past_key_values is None:
                    raise RuntimeError("RAG cached step did not return a KV cache")
                cache = output.past_key_values
                next_logits = output.logits[:, -1, :].to(dtype=torch.float32)

            generation_config = GenerationConfig(  # type: ignore[no-untyped-call]
                do_sample=False,
                max_new_tokens=spec.max_new_tokens,
                repetition_penalty=1.0,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                bos_token_id=None,
            )
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
        sequences = generated.sequences
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
            raise RuntimeError("RAG generate returned invalid sequences")
        generated_ids = [
            int(value) for value in sequences[0, len(prompt_ids) :].tolist()
        ]
        if not generated_ids or len(generated_ids) > spec.max_new_tokens:
            raise RuntimeError("RAG generate returned an invalid continuation length")
        if generated_ids != manual_ids:
            raise RuntimeError("manual greedy logits and GenerationMixin disagree")
        raw_output = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_with_special = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(raw_output, str) or not isinstance(raw_with_special, str):
            raise RuntimeError("RAG tokenizer.decode returned a non-string")
        short_ids = tuple(packed.context.sources)
        citation_audit = audit_citations(raw_output, short_ids)
        citation_passed = bool(citation_audit.cited_source_ids) and (
            citation_audit.syntactically_valid
        )
        abstention_match = raw_output == spec.abstention_text
        behavior_gate = (
            citation_passed
            if case.expected_behavior == "answer_with_citations"
            else abstention_match
        )
        retrieval_projection = {
            "implementation": BM25_AUTHORIZED_STATISTICS_IMPLEMENTATION,
            "top_k": case.top_k,
            "authorization_filtered_before_scoring": True,
            "document_ids": list(retrieved_ids),
            "results": [
                {
                    "document_id": result.document.document_id,
                    "rank": result.rank,
                    "score": result.score,
                    "source": result.source,
                }
                for result in results
            ],
        }
        packing_projection = {
            "budget_units": packed.budget_units,
            "base_cost_units": packed.base_cost_units,
            "used_cost_units": packed.used_cost_units,
            "cost_unit": packed.cost_unit,
            "max_chunks_per_source": packed.max_chunks_per_source,
            "document_ids": list(packed.selected_document_ids),
            "source_short_ids": list(short_ids),
            "rendered_context_sha256": _text_sha256(packed.context.rendered),
            "decisions": [
                {
                    "document_id": decision.document_id,
                    "stable_source_id": decision.stable_source_id,
                    "rank": decision.rank,
                    "selected": decision.selected,
                    "reason": decision.reason.value,
                    "cost_if_selected_units": decision.cost_if_selected_units,
                }
                for decision in packed.decisions
            ],
        }
        prompt_projection = {
            "system_prompt_sha256": _text_sha256(spec.system_prompt),
            "user_prompt_template_sha256": _text_sha256(spec.user_prompt_template),
            "chat_template_sha256": _text_sha256(chat_template),
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids_sha256": _canonical_sha256({"token_ids": prompt_ids}),
        }
        generation_projection = {
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": spec.max_new_tokens,
            "manual_greedy_token_ids": manual_ids,
            "generated_token_ids": generated_ids,
            "greedy_step_logits_sha256": step_hashes,
            "manual_greedy_matches_generate": True,
            "raw_output": raw_output,
            "raw_output_sha256": _text_sha256(raw_output),
            "decoded_with_special_tokens": raw_with_special,
            "generated_ended_with_eos": generated_ids[-1] == eos_token_id,
            "stop_reason": (
                "eos" if generated_ids[-1] == eos_token_id else "max_new_tokens"
            ),
        }
        verification_projection = {
            "verifier": "about_llm.rag.audit_citations+exact-abstention.v1",
            "valid_source_ids": list(short_ids),
            "cited_source_ids": list(citation_audit.cited_source_ids),
            "unknown_source_ids": list(citation_audit.unknown_source_ids),
            "uncited_paragraphs": list(citation_audit.uncited_paragraphs),
            "citation_syntax_passed": citation_passed,
            "abstention_exact_match": abstention_match,
            "expected_behavior_gate_passed": behavior_gate,
        }
        case_projection: dict[str, object] = {
            "case_id": case.case_id,
            "query_sha256": _text_sha256(case.query),
            "tenant_id": case.tenant_id,
            "principals": list(case.principals),
            "expected_behavior": case.expected_behavior,
            "retrieval": retrieval_projection,
            "packing": packing_projection,
            "prompt": prompt_projection,
            "generation": generation_projection,
            "verification": verification_projection,
        }
        case_projection["case_fingerprint"] = _canonical_sha256(case_projection)
        case_reports.append(case_projection)

    parameters = tuple(model.parameters())
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    parameter_storage_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    gates_passed = sum(
        bool(cast(Mapping[str, Any], case["verification"])["expected_behavior_gate_passed"])
        for case in case_reports
    )
    projection: dict[str, object] = {
        "report_version": RAG_TRANSFORMERS_REPORT_VERSION,
        "manifest_fingerprint": spec.manifest_fingerprint,
        "checkpoint_manifest_fingerprint": checkpoint_spec.manifest_fingerprint,
        "checked_at": spec.checked_at,
        "checkpoint": {
            "model_id": checkpoint_spec.model_id,
            "revision": checkpoint_spec.revision,
            "selected_file_count": len(snapshot.files),
            "selected_total_bytes": sum(
                cast(int, item["size_bytes"]) for item in snapshot.files
            ),
            "all_selected_file_bytes_verified_before_load": all(
                item.get("verified") is True for item in snapshot.files
            ),
            "loader_input": "verified_local_snapshot_directory",
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "transformers_version": str(transformers.__version__),
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": "eager",
            "torch_num_threads": torch.get_num_threads(),
            "cuda_executed": False,
        },
        "model": {
            "class": type(model).__name__,
            "model_type": str(model.config.model_type),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "parameter_storage_bytes": parameter_storage_bytes,
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
            "eval_mode": not bool(model.training),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocabulary_size_with_added_tokens": len(tokenizer),
            "chat_template_sha256": _text_sha256(chat_template),
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        },
        "cases": case_reports,
        "summary": {
            "case_count": len(case_reports),
            "expected_behavior_gate_passed_count": gates_passed,
            "all_expected_behavior_gates_passed": gates_passed == len(case_reports),
        },
        "scope": {
            "target_checkpoint_weights_loaded": True,
            "selected_checkpoint_files_verified_before_load": True,
            "authorization_filtered_before_bm25_scoring": True,
            "tokenizer_measured_context_packing_executed": True,
            "manual_greedy_logits_executed": True,
            "framework_generate_executed": True,
            "model_failures_recorded_without_output_repair": True,
            "claim_evidence_entailment_verified": False,
            "general_rag_quality_proven": False,
            "publisher_or_recorder_authenticated_by_signature": False,
            "verification_to_loader_reopen_toctou_eliminated": False,
            "gpu_vllm_or_production_service_executed": False,
        },
        "evidence_boundary": RAG_TRANSFORMERS_EVIDENCE_BOUNDARY,
    }
    projection["report_fingerprint"] = _canonical_sha256(projection)
    return projection


def run_rag_transformers_control(
    spec: RAGTransformersControlSpec,
    *,
    checkpoint_spec: CheckpointControlSpec,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Verify, load, and execute the reviewed Qwen RAG control."""

    try:
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment-specific dependency
        raise RuntimeError("torch and transformers are required for RAG control") from error
    validate_checkpoint_binding(spec, checkpoint_spec)
    directory = download_checkpoint_snapshot(
        checkpoint_spec, local_files_only=local_files_only
    )
    snapshot = verify_checkpoint_snapshot(checkpoint_spec, directory)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        snapshot.directory,
        trust_remote_code=False,
        local_files_only=True,
        use_fast=True,
    )
    dtype_argument = (
        {"dtype": torch.float32}
        if Version(transformers.__version__) >= Version("4.56")
        else {"torch_dtype": torch.float32}
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot.directory,
        trust_remote_code=False,
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        attn_implementation=spec.attention_implementation,
        **dtype_argument,
    )
    return execute_loaded_rag_transformers_control(
        spec,
        checkpoint_spec=checkpoint_spec,
        snapshot=snapshot,
        model=model,
        tokenizer=tokenizer,
    )


def verify_recorded_rag_transformers_report(
    path: Path,
    *,
    spec: RAGTransformersControlSpec,
    checkpoint_spec: CheckpointControlSpec,
) -> Mapping[str, Any]:
    """Verify a recorded report's closed schema, identities, and local invariants."""

    validate_checkpoint_binding(spec, checkpoint_spec)
    report = _load_json_file(path)
    _require_exact_fields(report, _REPORT_FIELDS, "report")
    if report.get("report_version") != RAG_TRANSFORMERS_REPORT_VERSION:
        raise ValueError("report.report_version is unsupported")
    if report.get("evidence_boundary") != RAG_TRANSFORMERS_EVIDENCE_BOUNDARY:
        raise ValueError("report.evidence_boundary drift")
    if report.get("manifest_fingerprint") != spec.manifest_fingerprint:
        raise ValueError("report manifest fingerprint mismatch")
    if report.get("checkpoint_manifest_fingerprint") != (
        checkpoint_spec.manifest_fingerprint
    ):
        raise ValueError("report checkpoint manifest fingerprint mismatch")
    if report.get("checked_at") != spec.checked_at:
        raise ValueError("report.checked_at differs from the reviewed manifest")
    recorded_fingerprint = _sha256(
        report.get("report_fingerprint"), "report.report_fingerprint"
    )
    unsigned = dict(report)
    del unsigned["report_fingerprint"]
    if recorded_fingerprint != _canonical_sha256(unsigned):
        raise ValueError("report fingerprint mismatch")

    checkpoint = _mapping(report.get("checkpoint"), "report.checkpoint")
    _require_exact_fields(checkpoint, _REPORT_CHECKPOINT_FIELDS, "report.checkpoint")
    if (checkpoint.get("model_id"), checkpoint.get("revision")) != (
        checkpoint_spec.model_id,
        checkpoint_spec.revision,
    ):
        raise ValueError("report.checkpoint identity mismatch")
    selected_count = _positive_integer(
        checkpoint.get("selected_file_count"),
        "report.checkpoint.selected_file_count",
    )
    selected_bytes = _positive_integer(
        checkpoint.get("selected_total_bytes"),
        "report.checkpoint.selected_total_bytes",
    )
    if selected_count != len(checkpoint_spec.files):
        raise ValueError("report.checkpoint selected file count mismatch")
    if selected_bytes != sum(item.size_bytes for item in checkpoint_spec.files):
        raise ValueError("report.checkpoint selected byte total mismatch")
    _require_boolean(
        checkpoint,
        "all_selected_file_bytes_verified_before_load",
        "report.checkpoint",
        expected=True,
    )
    if checkpoint.get("loader_input") != "verified_local_snapshot_directory":
        raise ValueError("report.checkpoint.loader_input is unsupported")

    runtime = _mapping(report.get("runtime"), "report.runtime")
    _require_exact_fields(runtime, _REPORT_RUNTIME_FIELDS, "report.runtime")
    for field in (
        "python_implementation",
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
    ):
        _required_string(runtime, field, "report.runtime")
    if (
        runtime.get("device"),
        runtime.get("dtype"),
        runtime.get("attention_implementation"),
    ) != (spec.device, spec.dtype, spec.attention_implementation):
        raise ValueError("report.runtime differs from the reviewed runtime")
    _positive_integer(runtime.get("torch_num_threads"), "report.runtime.torch_num_threads")
    _require_boolean(runtime, "cuda_executed", "report.runtime", expected=False)

    model = _mapping(report.get("model"), "report.model")
    _require_exact_fields(model, _REPORT_MODEL_FIELDS, "report.model")
    if (model.get("class"), model.get("model_type")) != (
        spec.expected_model_class,
        spec.expected_model_type,
    ):
        raise ValueError("report.model class/type mismatch")
    total_parameters = _positive_integer(
        model.get("total_parameters"), "report.model.total_parameters"
    )
    if _non_negative_integer(
        model.get("trainable_parameters"), "report.model.trainable_parameters"
    ) != 0:
        raise ValueError("report.model parameters were not frozen")
    if _positive_integer(
        model.get("parameter_storage_bytes"),
        "report.model.parameter_storage_bytes",
    ) != total_parameters * 4:
        raise ValueError("report.model FP32 parameter storage is inconsistent")
    if _unique_string_array(
        model.get("parameter_dtypes"), "report.model.parameter_dtypes"
    ) != ["torch.float32"]:
        raise ValueError("report.model parameter dtype is not reviewed FP32")
    _require_boolean(model, "eval_mode", "report.model", expected=True)

    tokenizer = _mapping(report.get("tokenizer"), "report.tokenizer")
    _require_exact_fields(tokenizer, _REPORT_TOKENIZER_FIELDS, "report.tokenizer")
    _required_string(tokenizer, "class", "report.tokenizer")
    vocabulary_size = _positive_integer(
        tokenizer.get("vocabulary_size_with_added_tokens"),
        "report.tokenizer.vocabulary_size_with_added_tokens",
    )
    tokenizer_chat_hash = _sha256(
        tokenizer.get("chat_template_sha256"),
        "report.tokenizer.chat_template_sha256",
    )
    pad_token_id = _non_negative_integer(
        tokenizer.get("pad_token_id"), "report.tokenizer.pad_token_id"
    )
    eos_token_id = _non_negative_integer(
        tokenizer.get("eos_token_id"), "report.tokenizer.eos_token_id"
    )
    if max(pad_token_id, eos_token_id) >= vocabulary_size:
        raise ValueError("report.tokenizer special id exceeds vocabulary")

    raw_cases = _bounded_array(report.get("cases"), "report.cases", maximum=_MAX_CASES)
    if len(raw_cases) != len(spec.cases):
        raise ValueError("report.cases count mismatch")
    corpus_by_id = {item.document_id: item.to_document() for item in spec.corpus}
    corpus_documents = tuple(item.to_document() for item in spec.corpus)
    reconstructed_index = BM25Index(corpus_documents)
    legacy_reconstructed_index = BM25Index._for_legacy_global_statistics(
        corpus_documents
    )
    gate_count = 0
    for index, (raw_case, expected_case) in enumerate(
        zip(raw_cases, spec.cases, strict=True)
    ):
        location = f"report.cases[{index}]"
        case = _mapping(raw_case, location)
        _require_exact_fields(case, _REPORT_CASE_FIELDS, location)
        if (
            case.get("case_id"),
            case.get("query_sha256"),
            case.get("tenant_id"),
            case.get("expected_behavior"),
        ) != (
            expected_case.case_id,
            _text_sha256(expected_case.query),
            expected_case.tenant_id,
            expected_case.expected_behavior,
        ):
            raise ValueError(f"{location} reviewed case identity mismatch")
        if _unique_string_array(
            case.get("principals"), f"{location}.principals", allow_empty=True
        ) != list(expected_case.principals):
            raise ValueError(f"{location}.principals mismatch")
        case_fingerprint = _sha256(
            case.get("case_fingerprint"), f"{location}.case_fingerprint"
        )
        case_unsigned = dict(case)
        del case_unsigned["case_fingerprint"]
        if case_fingerprint != _canonical_sha256(case_unsigned):
            raise ValueError(f"{location} fingerprint mismatch")

        retrieval = _mapping(case.get("retrieval"), f"{location}.retrieval")
        _require_exact_fields(
            retrieval, _REPORT_RETRIEVAL_FIELDS, f"{location}.retrieval"
        )
        implementation = retrieval.get("implementation")
        if implementation == BM25_AUTHORIZED_STATISTICS_IMPLEMENTATION:
            case_index = reconstructed_index
        elif implementation == BM25_LEGACY_GLOBAL_STATISTICS_IMPLEMENTATION:
            case_index = legacy_reconstructed_index
        else:
            raise ValueError(f"{location}.retrieval implementation drift")
        if _positive_integer(
            retrieval.get("top_k"), f"{location}.retrieval.top_k"
        ) != expected_case.top_k:
            raise ValueError(f"{location}.retrieval.top_k mismatch")
        _require_boolean(
            retrieval,
            "authorization_filtered_before_scoring",
            f"{location}.retrieval",
            expected=True,
        )
        retrieved_ids = _unique_string_array(
            retrieval.get("document_ids"),
            f"{location}.retrieval.document_ids",
            allow_empty=True,
        )
        if retrieved_ids != list(expected_case.expected_retrieved_document_ids):
            raise ValueError(f"{location}.retrieval document identity mismatch")
        reconstructed_results = case_index.search(
            expected_case.query,
            tenant_id=expected_case.tenant_id,
            principals=expected_case.principals,
            top_k=expected_case.top_k,
        )
        if [result.document.document_id for result in reconstructed_results] != retrieved_ids:
            raise ValueError(f"{location}.retrieval differs from current BM25 reconstruction")
        raw_results = _array(
            retrieval.get("results"), f"{location}.retrieval.results"
        )
        if len(raw_results) != len(retrieved_ids):
            raise ValueError(f"{location}.retrieval result count mismatch")
        for result_index, (raw_result, expected_document_id, reconstructed_result) in enumerate(
            zip(raw_results, retrieved_ids, reconstructed_results, strict=True)
        ):
            result_location = f"{location}.retrieval.results[{result_index}]"
            result = _mapping(raw_result, result_location)
            _require_exact_fields(
                result, _REPORT_RETRIEVAL_RESULT_FIELDS, result_location
            )
            if result.get("document_id") != expected_document_id:
                raise ValueError(f"{result_location}.document_id mismatch")
            if _positive_integer(result.get("rank"), f"{result_location}.rank") != (
                result_index + 1
            ):
                raise ValueError(f"{result_location}.rank is not canonical")
            observed_score = _finite_number(
                result.get("score"), f"{result_location}.score"
            )
            if observed_score <= 0:
                raise ValueError(f"{result_location}.score must be positive")
            if observed_score != reconstructed_result.score:
                raise ValueError(f"{result_location}.score differs from BM25 reconstruction")
            if result.get("source") != "bm25":
                raise ValueError(f"{result_location}.source is unsupported")

        packing = _mapping(case.get("packing"), f"{location}.packing")
        _require_exact_fields(packing, _REPORT_PACKING_FIELDS, f"{location}.packing")
        budget = _positive_integer(
            packing.get("budget_units"), f"{location}.packing.budget_units"
        )
        if budget != spec.prompt_budget_tokens:
            raise ValueError(f"{location}.packing budget mismatch")
        base_cost = _positive_integer(
            packing.get("base_cost_units"), f"{location}.packing.base_cost_units"
        )
        used_cost = _positive_integer(
            packing.get("used_cost_units"), f"{location}.packing.used_cost_units"
        )
        if base_cost > budget or used_cost > budget:
            raise ValueError(f"{location}.packing cost accounting is invalid")
        if packing.get("cost_unit") != "chat_tokens_including_output_reservation":
            raise ValueError(f"{location}.packing cost unit drift")
        if _positive_integer(
            packing.get("max_chunks_per_source"),
            f"{location}.packing.max_chunks_per_source",
        ) != spec.max_chunks_per_source:
            raise ValueError(f"{location}.packing source quota mismatch")
        packed_ids = _unique_string_array(
            packing.get("document_ids"),
            f"{location}.packing.document_ids",
            allow_empty=True,
        )
        if packed_ids != list(expected_case.expected_packed_document_ids):
            raise ValueError(f"{location}.packing document identity mismatch")
        source_short_ids = _unique_string_array(
            packing.get("source_short_ids"),
            f"{location}.packing.source_short_ids",
            allow_empty=True,
        )
        if source_short_ids != [f"S{number}" for number in range(1, len(packed_ids) + 1)]:
            raise ValueError(f"{location}.packing source short ids are not canonical")
        reconstructed = build_citation_context(
            (
                _result_for_reconstruction(corpus_by_id[document_id], rank)
                for rank, document_id in enumerate(packed_ids, 1)
            ),
            tenant_id=expected_case.tenant_id,
            principals=expected_case.principals,
        )
        if _sha256(
            packing.get("rendered_context_sha256"),
            f"{location}.packing.rendered_context_sha256",
        ) != _text_sha256(reconstructed.rendered):
            raise ValueError(f"{location}.packing rendered context mismatch")
        raw_decisions = _array(
            packing.get("decisions"), f"{location}.packing.decisions"
        )
        if len(raw_decisions) != len(retrieved_ids):
            raise ValueError(f"{location}.packing decision ledger count mismatch")
        selected_from_decisions: list[str] = []
        for decision_index, raw_decision in enumerate(raw_decisions):
            decision_location = f"{location}.packing.decisions[{decision_index}]"
            decision = _mapping(raw_decision, decision_location)
            _require_exact_fields(decision, _REPORT_DECISION_FIELDS, decision_location)
            document_id = _identifier(
                decision.get("document_id"), f"{decision_location}.document_id"
            )
            if document_id != retrieved_ids[decision_index]:
                raise ValueError(f"{decision_location}.document_id mismatch")
            expected_document = next(
                item for item in spec.corpus if item.document_id == document_id
            )
            if decision.get("stable_source_id") != expected_document.stable_source_id:
                raise ValueError(f"{decision_location}.stable_source_id mismatch")
            if _positive_integer(
                decision.get("rank"), f"{decision_location}.rank"
            ) != decision_index + 1:
                raise ValueError(f"{decision_location}.rank mismatch")
            selected = _boolean(decision.get("selected"), f"{decision_location}.selected")
            reason = _required_string(decision, "reason", decision_location)
            if reason not in {"selected", "duplicate_document", "source_quota", "budget"}:
                raise ValueError(f"{decision_location}.reason is unsupported")
            cost = decision.get("cost_if_selected_units")
            if reason in {"selected", "budget"}:
                _positive_integer(cost, f"{decision_location}.cost_if_selected_units")
            elif cost is not None:
                raise ValueError(f"{decision_location}.cost_if_selected_units must be null")
            if selected != (reason == "selected"):
                raise ValueError(f"{decision_location} selected/reason mismatch")
            if selected:
                selected_from_decisions.append(document_id)
        if selected_from_decisions != packed_ids:
            raise ValueError(f"{location}.packing decisions disagree with packed ids")

        prompt = _mapping(case.get("prompt"), f"{location}.prompt")
        _require_exact_fields(prompt, _REPORT_PROMPT_FIELDS, f"{location}.prompt")
        if prompt.get("system_prompt_sha256") != _text_sha256(spec.system_prompt):
            raise ValueError(f"{location}.prompt system prompt mismatch")
        if prompt.get("user_prompt_template_sha256") != _text_sha256(
            spec.user_prompt_template
        ):
            raise ValueError(f"{location}.prompt user template mismatch")
        if prompt.get("chat_template_sha256") != tokenizer_chat_hash:
            raise ValueError(f"{location}.prompt chat template mismatch")
        prompt_token_count = _positive_integer(
            prompt.get("prompt_token_count"), f"{location}.prompt.prompt_token_count"
        )
        _sha256(
            prompt.get("prompt_token_ids_sha256"),
            f"{location}.prompt.prompt_token_ids_sha256",
        )
        if prompt_token_count + spec.max_new_tokens != used_cost:
            raise ValueError(f"{location}.prompt token count disagrees with packing")

        generation = _mapping(case.get("generation"), f"{location}.generation")
        _require_exact_fields(
            generation, _REPORT_GENERATION_FIELDS, f"{location}.generation"
        )
        _require_boolean(
            generation, "do_sample", f"{location}.generation", expected=False
        )
        _require_boolean(
            generation, "use_cache", f"{location}.generation", expected=True
        )
        if _positive_integer(
            generation.get("max_new_tokens"),
            f"{location}.generation.max_new_tokens",
        ) != spec.max_new_tokens:
            raise ValueError(f"{location}.generation max_new_tokens mismatch")
        manual_ids = _integer_array(
            generation.get("manual_greedy_token_ids"),
            f"{location}.generation.manual_greedy_token_ids",
            maximum=spec.max_new_tokens,
        )
        generated_ids = _integer_array(
            generation.get("generated_token_ids"),
            f"{location}.generation.generated_token_ids",
            maximum=spec.max_new_tokens,
        )
        if manual_ids != generated_ids:
            raise ValueError(f"{location}.generation manual/generate ids mismatch")
        if any(token_id >= vocabulary_size for token_id in generated_ids):
            raise ValueError(f"{location}.generation token id exceeds vocabulary")
        step_hashes = _unique_or_repeated_sha256_array(
            generation.get("greedy_step_logits_sha256"),
            f"{location}.generation.greedy_step_logits_sha256",
            maximum=spec.max_new_tokens,
        )
        if len(step_hashes) != len(generated_ids):
            raise ValueError(f"{location}.generation logits hash count mismatch")
        _require_boolean(
            generation,
            "manual_greedy_matches_generate",
            f"{location}.generation",
            expected=True,
        )
        raw_output = _plain_string(
            generation.get("raw_output"), f"{location}.generation.raw_output"
        )
        if _sha256(
            generation.get("raw_output_sha256"),
            f"{location}.generation.raw_output_sha256",
        ) != _text_sha256(raw_output):
            raise ValueError(f"{location}.generation raw output hash mismatch")
        _plain_string(
            generation.get("decoded_with_special_tokens"),
            f"{location}.generation.decoded_with_special_tokens",
        )
        ended_with_eos = _boolean(
            generation.get("generated_ended_with_eos"),
            f"{location}.generation.generated_ended_with_eos",
        )
        if ended_with_eos != (generated_ids[-1] == eos_token_id):
            raise ValueError(f"{location}.generation EOS observation mismatch")
        expected_stop = "eos" if ended_with_eos else "max_new_tokens"
        if generation.get("stop_reason") != expected_stop:
            raise ValueError(f"{location}.generation stop reason mismatch")
        if not ended_with_eos and len(generated_ids) != spec.max_new_tokens:
            raise ValueError(f"{location}.generation stopped early without EOS")

        verification = _mapping(
            case.get("verification"), f"{location}.verification"
        )
        _require_exact_fields(
            verification, _REPORT_VERIFICATION_FIELDS, f"{location}.verification"
        )
        if verification.get("verifier") != (
            "about_llm.rag.audit_citations+exact-abstention.v1"
        ):
            raise ValueError(f"{location}.verification verifier drift")
        if _unique_string_array(
            verification.get("valid_source_ids"),
            f"{location}.verification.valid_source_ids",
            allow_empty=True,
        ) != source_short_ids:
            raise ValueError(f"{location}.verification valid source ids mismatch")
        observed_audit = audit_citations(raw_output, source_short_ids)
        citation_passed = bool(observed_audit.cited_source_ids) and (
            observed_audit.syntactically_valid
        )
        abstention_match = raw_output == spec.abstention_text
        expected_gate = (
            citation_passed
            if expected_case.expected_behavior == "answer_with_citations"
            else abstention_match
        )
        expected_verification = {
            "cited_source_ids": list(observed_audit.cited_source_ids),
            "unknown_source_ids": list(observed_audit.unknown_source_ids),
            "uncited_paragraphs": list(observed_audit.uncited_paragraphs),
            "citation_syntax_passed": citation_passed,
            "abstention_exact_match": abstention_match,
            "expected_behavior_gate_passed": expected_gate,
        }
        for field, expected_value in expected_verification.items():
            if verification.get(field) != expected_value:
                raise ValueError(f"{location}.verification.{field} mismatch")
        gate_count += int(expected_gate)

    summary = _mapping(report.get("summary"), "report.summary")
    _require_exact_fields(summary, _REPORT_SUMMARY_FIELDS, "report.summary")
    if _positive_integer(summary.get("case_count"), "report.summary.case_count") != len(
        spec.cases
    ):
        raise ValueError("report.summary case count mismatch")
    if _non_negative_integer(
        summary.get("expected_behavior_gate_passed_count"),
        "report.summary.expected_behavior_gate_passed_count",
    ) != gate_count:
        raise ValueError("report.summary behavior gate count mismatch")
    if summary.get("all_expected_behavior_gates_passed") is not (
        gate_count == len(spec.cases)
    ):
        raise ValueError("report.summary all-gates observation mismatch")

    scope = _mapping(report.get("scope"), "report.scope")
    _require_exact_fields(scope, _REPORT_SCOPE_FIELDS, "report.scope")
    true_scope = {
        "target_checkpoint_weights_loaded",
        "selected_checkpoint_files_verified_before_load",
        "authorization_filtered_before_bm25_scoring",
        "tokenizer_measured_context_packing_executed",
        "manual_greedy_logits_executed",
        "framework_generate_executed",
        "model_failures_recorded_without_output_repair",
    }
    false_scope = _REPORT_SCOPE_FIELDS - true_scope
    for field in true_scope:
        _require_boolean(scope, field, "report.scope", expected=True)
    for field in false_scope:
        _require_boolean(scope, field, "report.scope", expected=False)
    return report


def _result_for_reconstruction(document: Document, rank: int) -> SearchResult:
    return SearchResult(document=document, score=1.0, rank=rank, source="bm25")


def _render_user_prompt(template: str, *, query: str, context: str) -> str:
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


def _tokenize_chat(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    raw_ids = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(raw_ids, "ndim") and hasattr(raw_ids, "tolist"):
        if raw_ids.ndim != 2 or raw_ids.shape[0] != 1:
            raise RuntimeError("chat template must return one [1, tokens] tensor")
        values = raw_ids[0].tolist()
    elif isinstance(raw_ids, list):
        values = raw_ids
    else:
        raise RuntimeError("chat template returned an unsupported token container")
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"chat token {index} is not a non-negative integer")
        result.append(value)
    if not result:
        raise RuntimeError("chat template returned no token ids")
    return result


def _tensor_sha256(tensor: Any) -> str:
    import torch

    contiguous = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return "sha256:" + hashlib.sha256(contiguous.numpy().tobytes(order="C")).hexdigest()


def _text_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json_file(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{path}: cannot read JSON: {error}") from error
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"{path}: JSON exceeds the size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: JSON is not valid UTF-8") from error
    try:
        value: Any = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    canonical_json_bytes(value)
    return cast(dict[str, Any], value)


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], location: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{location}: field set mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return cast(Mapping[str, Any], value)


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _bounded_array(value: Any, location: str, *, maximum: int) -> list[Any]:
    result = _array(value, location)
    if not result or len(result) > maximum:
        raise ValueError(f"{location} must be a bounded non-empty array")
    return result


def _required_string(value: Mapping[str, Any], name: str, location: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{location}.{name} must be a non-empty string")
    return result


def _plain_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string")
    if len(value) > _MAX_TEXT_CHARACTERS:
        raise ValueError(f"{location} exceeds the character limit")
    return value


def _bounded_string(value: Mapping[str, Any], name: str, location: str) -> str:
    result = _required_string(value, name, location)
    if len(result) > _MAX_TEXT_CHARACTERS:
        raise ValueError(f"{location}.{name} exceeds the character limit")
    return result


def _identifier(value: Any, location: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{location} must be a simple identifier")
    return value


def _positive_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return cast(int, value)


def _non_negative_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return cast(int, value)


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    return result


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be a boolean")
    return value


def _require_boolean(
    value: Mapping[str, Any], name: str, location: str, *, expected: bool
) -> None:
    if value.get(name) is not expected:
        raise ValueError(f"{location}.{name} must be {expected}")


def _unique_string_array(
    value: Any, location: str, *, allow_empty: bool = False
) -> list[str]:
    raw = _array(value, location)
    if not raw and not allow_empty:
        raise ValueError(f"{location} must be non-empty")
    if len(raw) > _MAX_CORPUS_DOCUMENTS:
        raise ValueError(f"{location} exceeds the item limit")
    result: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{location}[{index}] must be a non-empty string")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{location} contains duplicate strings")
    return result


def _integer_array(value: Any, location: str, *, maximum: int) -> list[int]:
    raw = _bounded_array(value, location, maximum=maximum)
    return [
        _non_negative_integer(item, f"{location}[{index}]")
        for index, item in enumerate(raw)
    ]


def _unique_or_repeated_sha256_array(
    value: Any, location: str, *, maximum: int
) -> list[str]:
    raw = _bounded_array(value, location, maximum=maximum)
    return [_sha256(item, f"{location}[{index}]") for index, item in enumerate(raw)]


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{location} must be a canonical SHA-256 fingerprint")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return "sha256:" + artifact_fingerprint(cast(Mapping[str, object], value))


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result
