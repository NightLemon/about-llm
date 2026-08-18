"""Real-weight RAG control with a fail-closed policy around model generation."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
from about_llm.rag.citations import CitationContext, build_citation_context
from about_llm.rag.context_packing import (
    make_rag_chat_prompt_cost,
    pack_citation_context,
)
from about_llm.rag.generation_policy import (
    DEFAULT_RAG_PUBLICATION_POLICY,
    PublicationAction,
    evaluate_post_generation,
    evaluate_pre_generation,
    guard_rag_generation,
)
from about_llm.rag.models import Document, SearchResult
from about_llm.rag.transformers_control import (
    RAGControlCase,
    RAGTransformersControlSpec,
    load_rag_transformers_control_spec,
    validate_checkpoint_binding,
)

RAG_GUARDED_TRANSFORMERS_CONTROL_VERSION = (
    "about-llm.rag-guarded-transformers-control.v1"
)
RAG_GUARDED_TRANSFORMERS_REPORT_VERSION = (
    "about-llm.rag-guarded-transformers-control-report.v1"
)
RAG_GUARDED_TRANSFORMERS_EVIDENCE_BOUNDARY = (
    "This control verifies selected bytes from one immutable checkpoint revision, "
    "then executes authorization-filtered BM25 retrieval and tokenizer-measured "
    "packing for two fixed queries distinct from the earlier failure-control queries. "
    "A fail-closed policy wraps the generation callback: the control observes one "
    "GenerationMixin.generate API invocation when packed authorized evidence exists "
    "and zero when it is empty. It records audit and public projections separately. "
    "The offline verifier does not replay model generation, tokenizer token IDs, or "
    "decoding. The control does not count internal model forward calls, execute a "
    "manual-greedy logits cross-check, or prove claim-evidence entailment. The queries "
    "share the earlier authored corpus and checkpoint and are not a representative "
    "quality evaluation. "
    "Unkeyed hashes do not authenticate the publisher or recorder, verification does "
    "not eliminate loader-reopen TOCTOU, and this CPU control does not prove provider "
    "billing/cancellation, GPU/vLLM behavior, performance, or production integration."
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 2_000_000
_MAX_GENERATED_TOKENS = 256

_REPORT_FIELDS = {
    "report_version",
    "manifest_fingerprint",
    "checkpoint_manifest_fingerprint",
    "checked_at",
    "checkpoint",
    "runtime",
    "model",
    "tokenizer",
    "policy",
    "cases",
    "summary",
    "scope",
    "evidence_boundary",
    "report_fingerprint",
}
_CHECKPOINT_FIELDS = {
    "model_id",
    "revision",
    "selected_file_count",
    "selected_total_bytes",
    "all_selected_file_bytes_verified_before_load",
    "loader_input",
}
_RUNTIME_FIELDS = {
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
_MODEL_FIELDS = {
    "class",
    "model_type",
    "total_parameters",
    "trainable_parameters",
    "parameter_storage_bytes",
    "parameter_dtypes",
    "eval_mode",
}
_TOKENIZER_FIELDS = {
    "class",
    "vocabulary_size_with_added_tokens",
    "chat_template_sha256",
    "pad_token_id",
    "eos_token_id",
}
_POLICY_FIELDS = {
    "revision",
    "no_evidence_response",
    "rejected_response",
    "policy_fingerprint",
}
_CASE_FIELDS = {
    "case_id",
    "query_sha256",
    "tenant_id",
    "principals",
    "expected_behavior",
    "retrieval",
    "packing",
    "prompt",
    "generation",
    "decision",
    "public_decision",
    "case_fingerprint",
}
_RETRIEVAL_FIELDS = {
    "implementation",
    "top_k",
    "authorization_filtered_before_scoring",
    "document_ids",
    "results",
}
_RETRIEVAL_RESULT_FIELDS = {"document_id", "rank", "score", "source"}
_PACKING_FIELDS = {
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
_PACKING_DECISION_FIELDS = {
    "document_id",
    "stable_source_id",
    "rank",
    "selected",
    "reason",
    "cost_if_selected_units",
}
_PROMPT_FIELDS = {
    "system_prompt_sha256",
    "user_prompt_template_sha256",
    "chat_template_sha256",
    "prompt_token_count",
    "prompt_token_ids_sha256",
    "prompt_transmitted_to_model",
}
_GENERATION_FIELDS = {
    "do_sample",
    "use_cache",
    "max_new_tokens",
    "generator_callback_invocation_count",
    "framework_generate_invocation_count",
    "generated_token_ids",
    "raw_output",
    "raw_output_sha256",
    "decoded_with_special_tokens",
    "generated_ended_with_eos",
    "stop_reason",
}
_SUMMARY_FIELDS = {
    "case_count",
    "framework_generate_invocation_count",
    "publish_count",
    "pre_generation_abstention_count",
    "post_generation_rejection_count",
    "public_raw_output_field_count",
}
_SCOPE_FIELDS = {
    "target_checkpoint_weights_loaded",
    "selected_checkpoint_files_verified_before_load",
    "authorization_filtered_before_bm25_scoring",
    "tokenizer_measured_context_packing_executed",
    "publication_policy_wrapped_generation_callback",
    "framework_generate_invocation_executed_for_evidence",
    "framework_generate_invocation_suppression_observed_for_empty_evidence",
    "audit_public_projection_separation_executed",
    "manual_greedy_logits_cross_check_executed",
    "claim_evidence_entailment_verified",
    "general_rag_quality_proven",
    "publisher_or_recorder_authenticated_by_signature",
    "verification_to_loader_reopen_toctou_eliminated",
    "gpu_vllm_or_production_service_executed",
    "provider_billing_or_cancellation_verified",
    "production_integration_proven",
}


@dataclass
class _TransformersGenerator:
    model: Any
    tokenizer: Any
    prompt_ids: tuple[int, ...]
    expected_context: str
    eos_token_id: int
    pad_token_id: int
    max_new_tokens: int
    callback_invocations: int = 0
    framework_generate_invocations: int = 0
    generated_ids: tuple[int, ...] = ()
    raw_output: str | None = None
    decoded_with_special_tokens: str | None = None
    ended_with_eos: bool = False
    stop_reason: str = "pre_generation_abstention"
    _executed: bool = field(default=False, init=False, repr=False)

    def __call__(self, rendered_context: str) -> str:
        if self._executed:
            raise RuntimeError("guarded generator is single-use")
        self._executed = True
        self.callback_invocations += 1
        if rendered_context != self.expected_context:
            raise AssertionError("guarded generation context drift")
        try:
            import torch
            from transformers import GenerationConfig
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("torch and transformers are required") from error

        input_ids = torch.tensor([self.prompt_ids], dtype=torch.long, device="cpu")
        attention_mask = torch.ones_like(input_ids)
        generation_config = GenerationConfig(  # type: ignore[no-untyped-call]
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            bos_token_id=None,
        )
        self.framework_generate_invocations += 1
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
        sequences = generated.sequences
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
            raise RuntimeError("guarded generate returned invalid sequences")
        values = tuple(
            int(value) for value in sequences[0, len(self.prompt_ids) :].tolist()
        )
        if not values or len(values) > self.max_new_tokens:
            raise RuntimeError("guarded generate returned an invalid continuation length")
        if any(value < 0 or value >= len(self.tokenizer) for value in values):
            raise RuntimeError("guarded generate returned an out-of-vocabulary id")
        raw_output = self.tokenizer.decode(
            list(values),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        decoded = self.tokenizer.decode(
            list(values),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(raw_output, str) or not isinstance(decoded, str):
            raise RuntimeError("guarded tokenizer.decode returned a non-string")
        self.generated_ids = values
        self.raw_output = raw_output
        self.decoded_with_special_tokens = decoded
        self.ended_with_eos = values[-1] == self.eos_token_id
        self.stop_reason = "eos" if self.ended_with_eos else "max_new_tokens"
        return raw_output


def load_guarded_rag_transformers_control_spec(
    path: Path,
) -> RAGTransformersControlSpec:
    """Load the guarded workload through the shared strict RAG manifest schema."""

    spec = load_rag_transformers_control_spec(
        path,
        expected_control_version=RAG_GUARDED_TRANSFORMERS_CONTROL_VERSION,
        expected_evidence_boundary=RAG_GUARDED_TRANSFORMERS_EVIDENCE_BOUNDARY,
    )
    _validate_guarded_spec(spec)
    return spec


def _validate_guarded_spec(spec: RAGTransformersControlSpec) -> None:
    if len(spec.cases) != 2:
        raise ValueError("guarded control v1 requires exactly two cases")
    behaviors = [case.expected_behavior for case in spec.cases]
    if behaviors.count("answer_with_citations") != 1 or behaviors.count("abstain") != 1:
        raise ValueError("guarded control v1 requires one case for each behavior")
    if spec.abstention_text != DEFAULT_RAG_PUBLICATION_POLICY.no_evidence_response:
        raise ValueError("guarded control abstention text differs from the policy")


def execute_loaded_guarded_rag_transformers_control(
    spec: RAGTransformersControlSpec,
    *,
    checkpoint_spec: CheckpointControlSpec,
    snapshot: VerifiedCheckpointSnapshot,
    model: Any,
    tokenizer: Any,
) -> dict[str, object]:
    """Run retrieval/packing and apply the policy around real framework generation."""

    _validate_guarded_spec(spec)
    try:
        import torch
        import transformers
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("torch and transformers are required") from error

    validate_checkpoint_binding(spec, checkpoint_spec)
    if type(model).__name__ != spec.expected_model_class:
        raise ValueError("loaded model class does not match guarded control")
    if getattr(model.config, "model_type", None) != spec.expected_model_type:
        raise ValueError("loaded model_type does not match guarded control")
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("guarded control tokenizer must provide a chat template")
    eos_token_id = _token_id(tokenizer, "eos_token_id")
    pad_token_id = _token_id(tokenizer, "pad_token_id")
    if eos_token_id >= len(tokenizer) or pad_token_id >= len(tokenizer):
        raise ValueError("guarded control special id is outside tokenizer vocabulary")

    model.to("cpu")
    model.requires_grad_(False)
    model.eval()
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("guarded control model must have parameters")
    if any(
        parameter.device.type != "cpu" or parameter.dtype != torch.float32
        for parameter in parameters
    ):
        raise ValueError("guarded control model parameters must be CPU FP32")
    documents = tuple(item.to_document() for item in spec.corpus)
    index = BM25Index(documents)

    def tokenize_messages(messages: tuple[Mapping[str, str], ...]) -> Sequence[int]:
        return _tokenize_chat(tokenizer, messages)

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
                f"case {case.case_id}: reviewed guarded BM25 identity drift"
            )
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
                f"case {case.case_id}: reviewed guarded packing identity drift"
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
        prompt_ids = tuple(_tokenize_chat(tokenizer, messages))
        if len(prompt_ids) + spec.max_new_tokens != packed.used_cost_units:
            raise RuntimeError("guarded packing and final tokenization disagree")
        if any(value >= len(tokenizer) for value in prompt_ids):
            raise RuntimeError("guarded chat template emitted an invalid token id")

        generator = _TransformersGenerator(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            expected_context=packed.context.rendered,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=spec.max_new_tokens,
        )
        decision = guard_rag_generation(packed.context, generator)
        expected_invocations = int(bool(packed.context.sources))
        expected_from_behavior = int(
            case.expected_behavior == "answer_with_citations"
        )
        if expected_invocations != expected_from_behavior:
            raise RuntimeError(
                f"case {case.case_id}: evidence presence differs from expected behavior"
            )
        if (
            generator.callback_invocations != expected_invocations
            or generator.framework_generate_invocations != expected_invocations
        ):
            raise AssertionError("guarded generation invocation ledger mismatch")

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
            "source_short_ids": list(packed.context.sources),
            "rendered_context_sha256": _text_sha256(packed.context.rendered),
            "decisions": [
                {
                    "document_id": item.document_id,
                    "stable_source_id": item.stable_source_id,
                    "rank": item.rank,
                    "selected": item.selected,
                    "reason": item.reason.value,
                    "cost_if_selected_units": item.cost_if_selected_units,
                }
                for item in packed.decisions
            ],
        }
        prompt_projection = {
            "system_prompt_sha256": _text_sha256(spec.system_prompt),
            "user_prompt_template_sha256": _text_sha256(
                spec.user_prompt_template
            ),
            "chat_template_sha256": _text_sha256(chat_template),
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids_sha256": _canonical_sha256(
                {"token_ids": list(prompt_ids)}
            ),
            "prompt_transmitted_to_model": bool(packed.context.sources),
        }
        generation_projection = {
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": spec.max_new_tokens,
            "generator_callback_invocation_count": generator.callback_invocations,
            "framework_generate_invocation_count": (
                generator.framework_generate_invocations
            ),
            "generated_token_ids": list(generator.generated_ids),
            "raw_output": generator.raw_output,
            "raw_output_sha256": (
                _text_sha256(generator.raw_output)
                if generator.raw_output is not None
                else None
            ),
            "decoded_with_special_tokens": generator.decoded_with_special_tokens,
            "generated_ended_with_eos": generator.ended_with_eos,
            "stop_reason": generator.stop_reason,
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
            "decision": decision.to_dict(),
            "public_decision": decision.to_public_dict(),
        }
        case_projection["case_fingerprint"] = _canonical_sha256(case_projection)
        case_reports.append(case_projection)

    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    parameter_storage_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    actions = [
        cast(Mapping[str, Any], case["decision"])["action"]
        for case in case_reports
    ]
    total_generate_invocations = sum(
        cast(Mapping[str, Any], case["generation"])[
            "framework_generate_invocation_count"
        ]
        for case in case_reports
    )
    if total_generate_invocations != 1:
        raise RuntimeError("guarded control must execute exactly one generate invocation")
    if PublicationAction.ABSTAIN.value not in actions:
        raise RuntimeError("guarded control must observe a pre-generation abstention")

    projection: dict[str, object] = {
        "report_version": RAG_GUARDED_TRANSFORMERS_REPORT_VERSION,
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
        "policy": {
            **DEFAULT_RAG_PUBLICATION_POLICY.to_dict(),
            "policy_fingerprint": DEFAULT_RAG_PUBLICATION_POLICY.fingerprint,
        },
        "cases": case_reports,
        "summary": {
            "case_count": len(case_reports),
            "framework_generate_invocation_count": total_generate_invocations,
            "publish_count": actions.count(PublicationAction.PUBLISH.value),
            "pre_generation_abstention_count": actions.count(
                PublicationAction.ABSTAIN.value
            ),
            "post_generation_rejection_count": actions.count(
                PublicationAction.REJECT.value
            ),
            "public_raw_output_field_count": sum(
                "raw_output" in cast(Mapping[str, Any], case["public_decision"])
                for case in case_reports
            ),
        },
        "scope": {
            "target_checkpoint_weights_loaded": True,
            "selected_checkpoint_files_verified_before_load": True,
            "authorization_filtered_before_bm25_scoring": True,
            "tokenizer_measured_context_packing_executed": True,
            "publication_policy_wrapped_generation_callback": True,
            "framework_generate_invocation_executed_for_evidence": True,
            "framework_generate_invocation_suppression_observed_for_empty_evidence": True,
            "audit_public_projection_separation_executed": True,
            "manual_greedy_logits_cross_check_executed": False,
            "claim_evidence_entailment_verified": False,
            "general_rag_quality_proven": False,
            "publisher_or_recorder_authenticated_by_signature": False,
            "verification_to_loader_reopen_toctou_eliminated": False,
            "gpu_vllm_or_production_service_executed": False,
            "provider_billing_or_cancellation_verified": False,
            "production_integration_proven": False,
        },
        "evidence_boundary": RAG_GUARDED_TRANSFORMERS_EVIDENCE_BOUNDARY,
    }
    projection["report_fingerprint"] = _canonical_sha256(projection)
    return projection


def run_guarded_rag_transformers_control(
    spec: RAGTransformersControlSpec,
    *,
    checkpoint_spec: CheckpointControlSpec,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Verify checkpoint bytes, load the model, and run the guarded control."""

    _validate_guarded_spec(spec)
    try:
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("torch and transformers are required") from error
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
    return execute_loaded_guarded_rag_transformers_control(
        spec,
        checkpoint_spec=checkpoint_spec,
        snapshot=snapshot,
        model=model,
        tokenizer=tokenizer,
    )


def verify_recorded_guarded_rag_transformers_report(
    path: Path,
    *,
    spec: RAGTransformersControlSpec,
    checkpoint_spec: CheckpointControlSpec,
) -> Mapping[str, Any]:
    """Verify the guarded report by rebuilding all offline-checkable semantics."""

    _validate_guarded_spec(spec)
    validate_checkpoint_binding(spec, checkpoint_spec)
    report = _load_json_file(path)
    _require_exact_fields(report, _REPORT_FIELDS, "report")
    if report.get("report_version") != RAG_GUARDED_TRANSFORMERS_REPORT_VERSION:
        raise ValueError("guarded report version is unsupported")
    if report.get("manifest_fingerprint") != spec.manifest_fingerprint:
        raise ValueError("guarded report manifest fingerprint mismatch")
    if (
        report.get("checkpoint_manifest_fingerprint")
        != checkpoint_spec.manifest_fingerprint
    ):
        raise ValueError("guarded report checkpoint fingerprint mismatch")
    if report.get("checked_at") != spec.checked_at:
        raise ValueError("guarded report checked_at mismatch")
    if report.get("evidence_boundary") != RAG_GUARDED_TRANSFORMERS_EVIDENCE_BOUNDARY:
        raise ValueError("guarded report evidence boundary drift")
    observed_fingerprint = _sha256(
        report.get("report_fingerprint"), "report.report_fingerprint"
    )
    unsigned = dict(report)
    unsigned.pop("report_fingerprint", None)
    if _canonical_sha256(unsigned) != observed_fingerprint:
        raise ValueError("guarded report fingerprint mismatch")

    _verify_checkpoint_runtime_model_tokenizer(report, checkpoint_spec, spec)
    tokenizer_projection = _mapping(report.get("tokenizer"), "report.tokenizer")
    tokenizer_chat_template_sha256 = _sha256(
        tokenizer_projection.get("chat_template_sha256"),
        "report.tokenizer.chat_template_sha256",
    )
    policy = _mapping(report.get("policy"), "report.policy")
    _require_exact_fields(policy, _POLICY_FIELDS, "report.policy")
    expected_policy = {
        **DEFAULT_RAG_PUBLICATION_POLICY.to_dict(),
        "policy_fingerprint": DEFAULT_RAG_PUBLICATION_POLICY.fingerprint,
    }
    if policy != expected_policy:
        raise ValueError("guarded report policy drift")

    documents = tuple(item.to_document() for item in spec.corpus)
    document_by_id = {document.document_id: document for document in documents}
    index = BM25Index(documents)
    legacy_index = BM25Index._for_legacy_global_statistics(documents)
    raw_cases = _array(report.get("cases"), "report.cases")
    if len(raw_cases) != len(spec.cases):
        raise ValueError("guarded report case count mismatch")
    actions: list[str] = []
    generate_invocations = 0
    for index_value, (case_spec, raw_case) in enumerate(
        zip(spec.cases, raw_cases, strict=True)
    ):
        location = f"report.cases[{index_value}]"
        case = _mapping(raw_case, location)
        _require_exact_fields(case, _CASE_FIELDS, location)
        if (
            case.get("case_id") != case_spec.case_id
            or case.get("query_sha256") != _text_sha256(case_spec.query)
            or case.get("tenant_id") != case_spec.tenant_id
            or case.get("principals") != list(case_spec.principals)
            or case.get("expected_behavior") != case_spec.expected_behavior
        ):
            raise ValueError(f"{location} workload binding mismatch")

        retrieval = _mapping(case.get("retrieval"), f"{location}.retrieval")
        implementation = retrieval.get("implementation")
        if implementation == BM25_AUTHORIZED_STATISTICS_IMPLEMENTATION:
            case_index = index
        elif implementation == BM25_LEGACY_GLOBAL_STATISTICS_IMPLEMENTATION:
            case_index = legacy_index
        else:
            raise ValueError(f"{location}.retrieval implementation drift")
        actual_results = case_index.search(
            case_spec.query,
            tenant_id=case_spec.tenant_id,
            principals=case_spec.principals,
            top_k=case_spec.top_k,
        )
        _verify_retrieval(case, actual_results, case_spec.top_k, location)
        packed_document_ids = list(case_spec.expected_packed_document_ids)
        actual_by_id = {
            result.document.document_id: result for result in actual_results
        }
        try:
            selected_results = [actual_by_id[value] for value in packed_document_ids]
        except KeyError as error:
            raise ValueError(f"{location}.packing selects an unretrieved document") from error
        context = build_citation_context(
            selected_results,
            tenant_id=case_spec.tenant_id,
            principals=case_spec.principals,
        )
        prompt_token_count = _verify_packing_and_prompt(
            case,
            case_spec=case_spec,
            actual_results=actual_results,
            context=context,
            document_by_id=document_by_id,
            spec=spec,
            tokenizer_chat_template_sha256=tokenizer_chat_template_sha256,
            location=location,
        )
        generation = _mapping(case.get("generation"), f"{location}.generation")
        _require_exact_fields(generation, _GENERATION_FIELDS, f"{location}.generation")
        expected_calls = int(bool(context.sources))
        expected_from_behavior = int(
            case_spec.expected_behavior == "answer_with_citations"
        )
        if expected_calls != expected_from_behavior:
            raise ValueError(f"{location}.packing evidence/behavior mismatch")
        for field_name in (
            "generator_callback_invocation_count",
            "framework_generate_invocation_count",
        ):
            if _non_negative_integer(
                generation.get(field_name), f"{location}.generation.{field_name}"
            ) != expected_calls:
                raise ValueError(f"{location}.generation invocation count mismatch")
        _require_boolean(
            generation, "do_sample", f"{location}.generation", expected=False
        )
        _require_boolean(
            generation, "use_cache", f"{location}.generation", expected=True
        )
        if generation.get("max_new_tokens") != spec.max_new_tokens:
            raise ValueError(f"{location}.generation token cap mismatch")
        generated_ids = _integer_array(
            generation.get("generated_token_ids"),
            f"{location}.generation.generated_token_ids",
            allow_empty=not bool(context.sources),
        )
        if len(generated_ids) > spec.max_new_tokens:
            raise ValueError(f"{location}.generation exceeds token cap")
        tokenizer = _mapping(report.get("tokenizer"), "report.tokenizer")
        vocabulary_size = _positive_integer(
            tokenizer.get("vocabulary_size_with_added_tokens"),
            "report.tokenizer.vocabulary_size_with_added_tokens",
        )
        if any(value >= vocabulary_size for value in generated_ids):
            raise ValueError(f"{location}.generation token exceeds vocabulary")
        if expected_calls:
            raw_output = _plain_string(
                generation.get("raw_output"), f"{location}.generation.raw_output"
            )
            if generation.get("raw_output_sha256") != _text_sha256(raw_output):
                raise ValueError(f"{location}.generation raw output hash mismatch")
            _plain_string(
                generation.get("decoded_with_special_tokens"),
                f"{location}.generation.decoded_with_special_tokens",
            )
            eos_token_id = _non_negative_integer(
                tokenizer.get("eos_token_id"), "report.tokenizer.eos_token_id"
            )
            ended = generated_ids[-1] == eos_token_id
            if generation.get("generated_ended_with_eos") is not ended:
                raise ValueError(f"{location}.generation EOS observation mismatch")
            expected_stop = "eos" if ended else "max_new_tokens"
            if generation.get("stop_reason") != expected_stop:
                raise ValueError(f"{location}.generation stop reason mismatch")
            decision = evaluate_post_generation(context, raw_output)
        else:
            if not (
                generated_ids == []
                and generation.get("raw_output") is None
                and generation.get("raw_output_sha256") is None
                and generation.get("decoded_with_special_tokens") is None
                and generation.get("generated_ended_with_eos") is False
                and generation.get("stop_reason") == "pre_generation_abstention"
            ):
                raise ValueError(f"{location}.generation abstention ledger mismatch")
            early_decision = evaluate_pre_generation(context)
            if early_decision is None:
                raise AssertionError("empty context did not produce abstention")
            decision = early_decision

        observed_decision = _mapping(case.get("decision"), f"{location}.decision")
        observed_public = _mapping(
            case.get("public_decision"), f"{location}.public_decision"
        )
        if observed_decision != decision.to_dict():
            raise ValueError(f"{location}.decision differs from local reconstruction")
        if observed_public != decision.to_public_dict():
            raise ValueError(
                f"{location}.public_decision differs from local reconstruction"
            )
        if "raw_output" in observed_public or "uncited_paragraphs" in observed_public:
            raise ValueError(f"{location}.public_decision exposes audit fields")
        actions.append(decision.action.value)
        generate_invocations += expected_calls
        if prompt_token_count + spec.max_new_tokens > spec.prompt_budget_tokens:
            raise ValueError(f"{location}.prompt exceeds guarded budget")
        case_fingerprint = _sha256(
            case.get("case_fingerprint"), f"{location}.case_fingerprint"
        )
        case_unsigned = dict(case)
        case_unsigned.pop("case_fingerprint", None)
        if _canonical_sha256(case_unsigned) != case_fingerprint:
            raise ValueError(f"{location}.case_fingerprint mismatch")

    summary = _mapping(report.get("summary"), "report.summary")
    _require_exact_fields(summary, _SUMMARY_FIELDS, "report.summary")
    expected_summary = {
        "case_count": len(raw_cases),
        "framework_generate_invocation_count": generate_invocations,
        "publish_count": actions.count(PublicationAction.PUBLISH.value),
        "pre_generation_abstention_count": actions.count(
            PublicationAction.ABSTAIN.value
        ),
        "post_generation_rejection_count": actions.count(
            PublicationAction.REJECT.value
        ),
        "public_raw_output_field_count": 0,
    }
    if summary != expected_summary:
        raise ValueError("guarded report summary mismatch")
    scope = _mapping(report.get("scope"), "report.scope")
    _require_exact_fields(scope, _SCOPE_FIELDS, "report.scope")
    true_scope = {
        "target_checkpoint_weights_loaded",
        "selected_checkpoint_files_verified_before_load",
        "authorization_filtered_before_bm25_scoring",
        "tokenizer_measured_context_packing_executed",
        "publication_policy_wrapped_generation_callback",
        "framework_generate_invocation_executed_for_evidence",
        "framework_generate_invocation_suppression_observed_for_empty_evidence",
        "audit_public_projection_separation_executed",
    }
    for field_name in true_scope:
        _require_boolean(scope, field_name, "report.scope", expected=True)
    for field_name in _SCOPE_FIELDS - true_scope:
        _require_boolean(scope, field_name, "report.scope", expected=False)
    return report


def _verify_checkpoint_runtime_model_tokenizer(
    report: Mapping[str, Any],
    checkpoint_spec: CheckpointControlSpec,
    spec: RAGTransformersControlSpec,
) -> None:
    checkpoint = _mapping(report.get("checkpoint"), "report.checkpoint")
    _require_exact_fields(checkpoint, _CHECKPOINT_FIELDS, "report.checkpoint")
    if not (
        checkpoint.get("model_id") == checkpoint_spec.model_id
        and checkpoint.get("revision") == checkpoint_spec.revision
        and checkpoint.get("selected_file_count") == len(checkpoint_spec.files)
        and checkpoint.get("selected_total_bytes")
        == sum(item.size_bytes for item in checkpoint_spec.files)
        and checkpoint.get("all_selected_file_bytes_verified_before_load") is True
        and checkpoint.get("loader_input") == "verified_local_snapshot_directory"
    ):
        raise ValueError("guarded report checkpoint ledger mismatch")
    runtime = _mapping(report.get("runtime"), "report.runtime")
    _require_exact_fields(runtime, _RUNTIME_FIELDS, "report.runtime")
    for name in (
        "python_implementation",
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
    ):
        _required_string(runtime, name, "report.runtime")
    if (
        runtime.get("device"),
        runtime.get("dtype"),
        runtime.get("attention_implementation"),
        runtime.get("cuda_executed"),
    ) != ("cpu", "float32", "eager", False):
        raise ValueError("guarded report runtime mismatch")
    _positive_integer(runtime.get("torch_num_threads"), "report.runtime.torch_num_threads")

    model = _mapping(report.get("model"), "report.model")
    _require_exact_fields(model, _MODEL_FIELDS, "report.model")
    total_parameters = _positive_integer(
        model.get("total_parameters"), "report.model.total_parameters"
    )
    storage_bytes = _positive_integer(
        model.get("parameter_storage_bytes"), "report.model.parameter_storage_bytes"
    )
    if not (
        model.get("class") == spec.expected_model_class
        and model.get("model_type") == spec.expected_model_type
        and model.get("trainable_parameters") == 0
        and storage_bytes == total_parameters * 4
        and model.get("parameter_dtypes") == ["torch.float32"]
        and model.get("eval_mode") is True
    ):
        raise ValueError("guarded report model ledger mismatch")

    tokenizer = _mapping(report.get("tokenizer"), "report.tokenizer")
    _require_exact_fields(tokenizer, _TOKENIZER_FIELDS, "report.tokenizer")
    _required_string(tokenizer, "class", "report.tokenizer")
    vocabulary_size = _positive_integer(
        tokenizer.get("vocabulary_size_with_added_tokens"),
        "report.tokenizer.vocabulary_size_with_added_tokens",
    )
    _sha256(tokenizer.get("chat_template_sha256"), "report.tokenizer.chat_template_sha256")
    for name in ("pad_token_id", "eos_token_id"):
        token_id = _non_negative_integer(
            tokenizer.get(name), f"report.tokenizer.{name}"
        )
        if token_id >= vocabulary_size:
            raise ValueError(f"report.tokenizer.{name} exceeds vocabulary")


def _verify_retrieval(
    case: Mapping[str, Any],
    actual_results: Sequence[SearchResult],
    top_k: int,
    location: str,
) -> None:
    retrieval = _mapping(case.get("retrieval"), f"{location}.retrieval")
    _require_exact_fields(retrieval, _RETRIEVAL_FIELDS, f"{location}.retrieval")
    if not (
        retrieval.get("implementation")
        in {
            BM25_AUTHORIZED_STATISTICS_IMPLEMENTATION,
            BM25_LEGACY_GLOBAL_STATISTICS_IMPLEMENTATION,
        }
        and retrieval.get("top_k") == top_k
        and retrieval.get("authorization_filtered_before_scoring") is True
        and retrieval.get("document_ids")
        == [result.document.document_id for result in actual_results]
    ):
        raise ValueError(f"{location}.retrieval identity mismatch")
    rows = _array(retrieval.get("results"), f"{location}.retrieval.results")
    if len(rows) != len(actual_results):
        raise ValueError(f"{location}.retrieval result count mismatch")
    for row_index, (raw_row, actual) in enumerate(zip(rows, actual_results, strict=True)):
        row_location = f"{location}.retrieval.results[{row_index}]"
        row = _mapping(raw_row, row_location)
        _require_exact_fields(row, _RETRIEVAL_RESULT_FIELDS, row_location)
        score = _finite_number(row.get("score"), f"{row_location}.score")
        if not (
            row.get("document_id") == actual.document.document_id
            and row.get("rank") == actual.rank
            and math.isclose(score, actual.score, rel_tol=0.0, abs_tol=1e-12)
            and row.get("source") == actual.source
        ):
            raise ValueError(f"{row_location} differs from BM25 reconstruction")


def _verify_packing_and_prompt(
    case: Mapping[str, Any],
    *,
    case_spec: RAGControlCase,
    actual_results: Sequence[SearchResult],
    context: CitationContext,
    document_by_id: Mapping[str, Document],
    spec: RAGTransformersControlSpec,
    tokenizer_chat_template_sha256: str,
    location: str,
) -> int:
    packing = _mapping(case.get("packing"), f"{location}.packing")
    _require_exact_fields(packing, _PACKING_FIELDS, f"{location}.packing")
    if not (
        packing.get("budget_units") == spec.prompt_budget_tokens
        and packing.get("cost_unit") == "chat_tokens_including_output_reservation"
        and packing.get("max_chunks_per_source") == spec.max_chunks_per_source
        and packing.get("document_ids")
        == list(case_spec.expected_packed_document_ids)
        and packing.get("source_short_ids") == list(context.sources)
        and packing.get("rendered_context_sha256")
        == _text_sha256(context.rendered)
    ):
        raise ValueError(f"{location}.packing identity mismatch")
    base_cost = _positive_integer(
        packing.get("base_cost_units"), f"{location}.packing.base_cost_units"
    )
    used_cost = _positive_integer(
        packing.get("used_cost_units"), f"{location}.packing.used_cost_units"
    )
    if not base_cost <= used_cost <= spec.prompt_budget_tokens:
        raise ValueError(f"{location}.packing cost ledger mismatch")
    if not context.sources and used_cost != base_cost:
        raise ValueError(f"{location}.packing empty-context cost mismatch")

    decisions = _array(packing.get("decisions"), f"{location}.packing.decisions")
    if len(decisions) != len(actual_results):
        raise ValueError(f"{location}.packing decision count mismatch")
    selected_ids: list[str] = []
    seen_document_ids: set[str] = set()
    selected_per_source: dict[str, int] = {}
    final_selected_cost = base_cost
    for decision_index, (raw_decision, result) in enumerate(
        zip(decisions, actual_results, strict=True)
    ):
        decision_location = f"{location}.packing.decisions[{decision_index}]"
        decision = _mapping(raw_decision, decision_location)
        _require_exact_fields(decision, _PACKING_DECISION_FIELDS, decision_location)
        document = document_by_id[result.document.document_id]
        stable_source_id = document.metadata.get("source_id")
        if not isinstance(stable_source_id, str) or not stable_source_id:
            raise AssertionError("reviewed corpus has an invalid stable source id")
        if not (
            decision.get("document_id") == document.document_id
            and decision.get("stable_source_id") == stable_source_id
            and decision.get("rank") == result.rank
        ):
            raise ValueError(f"{decision_location} candidate identity mismatch")
        selected = _boolean(decision.get("selected"), f"{decision_location}.selected")
        reason = decision.get("reason")
        cost = decision.get("cost_if_selected_units")
        if document.document_id in seen_document_ids:
            expected_selected = False
            expected_reason = "duplicate_document"
            if cost is not None:
                raise ValueError(f"{decision_location} duplicate cost must be null")
        else:
            seen_document_ids.add(document.document_id)
            selected_for_source = selected_per_source.get(stable_source_id, 0)
            if selected_for_source >= spec.max_chunks_per_source:
                expected_selected = False
                expected_reason = "source_quota"
                if cost is not None:
                    raise ValueError(f"{decision_location} quota cost must be null")
            else:
                prospective_cost = _positive_integer(
                    cost, f"{decision_location}.cost_if_selected_units"
                )
                expected_selected = prospective_cost <= spec.prompt_budget_tokens
                expected_reason = "selected" if expected_selected else "budget"
                if expected_selected:
                    selected_ids.append(document.document_id)
                    selected_per_source[stable_source_id] = selected_for_source + 1
                    final_selected_cost = prospective_cost
        if selected is not expected_selected or reason != expected_reason:
            raise ValueError(f"{decision_location} packing decision mismatch")
    if selected_ids != list(case_spec.expected_packed_document_ids):
        raise ValueError(f"{location}.packing selected order mismatch")
    if final_selected_cost != used_cost:
        raise ValueError(f"{location}.packing final selected cost mismatch")

    prompt = _mapping(case.get("prompt"), f"{location}.prompt")
    _require_exact_fields(prompt, _PROMPT_FIELDS, f"{location}.prompt")
    prompt_transmitted = _boolean(
        prompt.get("prompt_transmitted_to_model"),
        f"{location}.prompt.prompt_transmitted_to_model",
    )
    if not (
        prompt.get("system_prompt_sha256") == _text_sha256(spec.system_prompt)
        and prompt.get("user_prompt_template_sha256")
        == _text_sha256(spec.user_prompt_template)
        and prompt.get("chat_template_sha256")
        == tokenizer_chat_template_sha256
        and prompt_transmitted == bool(context.sources)
    ):
        raise ValueError(f"{location}.prompt binding mismatch")
    _sha256(
        prompt.get("prompt_token_ids_sha256"),
        f"{location}.prompt.prompt_token_ids_sha256",
    )
    prompt_token_count = _positive_integer(
        prompt.get("prompt_token_count"), f"{location}.prompt.prompt_token_count"
    )
    if prompt_token_count + spec.max_new_tokens != used_cost:
        raise ValueError(f"{location}.prompt token ledger mismatch")
    return prompt_token_count


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
            raise RuntimeError("guarded chat template must return one sequence")
        values = raw_ids[0].tolist()
    elif isinstance(raw_ids, list):
        values = raw_ids
    else:
        raise RuntimeError("guarded chat template returned an unsupported value")
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"guarded chat token {index} is invalid")
        result.append(value)
    if not result:
        raise RuntimeError("guarded chat template returned no token ids")
    return result


def _token_id(tokenizer: Any, name: str) -> int:
    value = getattr(tokenizer, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"guarded tokenizer must provide an integer {name}")
    return value


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


def _required_string(value: Mapping[str, Any], name: str, location: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{location}.{name} must be a non-empty string")
    return result


def _plain_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string")
    if len(value) > 16_384:
        raise ValueError(f"{location} exceeds the character limit")
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


def _integer_array(value: Any, location: str, *, allow_empty: bool) -> list[int]:
    raw = _array(value, location)
    if not raw and not allow_empty:
        raise ValueError(f"{location} must be non-empty")
    if len(raw) > _MAX_GENERATED_TOKENS:
        raise ValueError(f"{location} exceeds the token limit")
    return [
        _non_negative_integer(item, f"{location}[{index}]")
        for index, item in enumerate(raw)
    ]


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{location} must be a canonical SHA-256 fingerprint")
    return value


def _text_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


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
