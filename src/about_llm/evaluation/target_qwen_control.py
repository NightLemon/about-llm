"""Recorded, small-scope behavior evaluation for the reviewed Qwen checkpoint.

The suite is intentionally small and authored.  It connects the repository's
evaluation artifact discipline to real target weights without presenting seven
prompts as a representative benchmark or a statistically powered quality claim.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import platform
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from about_llm.evaluation.runner import EvaluationCase
from about_llm.evaluation.text_metrics import (
    literal_exact_match,
    normalized_exact_match,
    token_f1,
)
from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    download_checkpoint_snapshot,
    verify_checkpoint_snapshot,
)
from about_llm.integrations.transformers_tools import parameter_report
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes
from about_llm.model_config import inspect_decoder_config, load_model_config_json

TARGET_QWEN_EVALUATION_CONTROL_VERSION = (
    "about-llm.target-qwen-behavior-evaluation-control.v1"
)
TARGET_QWEN_EVALUATION_REPORT_VERSION = (
    "about-llm.target-qwen-behavior-evaluation-report.v1"
)
TARGET_QWEN_EVALUATION_CHECKED_AT = "2026-08-15"
TARGET_QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TARGET_QWEN_CHECKPOINT_MANIFEST_FINGERPRINT = (
    "sha256:ddf41f2cff963bc2a8fc186c28369abba8a920b850152fc815e2b17c7d037876"
)
TARGET_QWEN_SELECTED_FILE_COUNT = 7
TARGET_QWEN_SELECTED_FILE_BYTES = 999_586_347
TARGET_QWEN_TOTAL_PARAMETERS = 494_032_768
TARGET_QWEN_PARAMETER_STORAGE_BYTES = 1_976_131_072
TARGET_QWEN_EVALUATION_RECORDED_REPORT_FINGERPRINT = (
    "sha256:dd30a278cbc076c973c0b0babc9e752b1063d8bfb114c852b34ea42b2cd85c43"
)
TARGET_QWEN_EVALUATION_EVIDENCE_BOUNDARY = (
    "This control verifies and loads the reviewed immutable Qwen2.5-0.5B-Instruct "
    "snapshot on CPU in FP32 with trust_remote_code disabled, then performs greedy "
    "Transformers generation for seven fixed authored cases. It records raw decoded "
    "outputs, token identities, literal and normalized exact match, token F1, and deterministic "
    "slice aggregates. The suite is not externally preregistered, independently "
    "sampled, held out from prompt selection, representative of a language, domain, "
    "task distribution, safety policy, or user population, or statistically powered. "
    "It does not use a judge model, estimate uncertainty, compare systems, execute GPU, "
    "CUDA, vLLM, tools, RAG, training, or a performance benchmark, authenticate the "
    "publisher, establish licensing, quality, generalization, calibration, effective "
    "context, production safety, or eliminate verification-to-loader-reopen TOCTOU."
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CASE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_MAX_FILE_BYTES = 1_000_000
_MAX_CASES = 32
_MAX_PROMPT_CHARACTERS = 4096
_MAX_EXPECTED_CHARACTERS = 512
_MAX_OUTPUT_CHARACTERS = 4096
_MAX_NEW_TOKENS = 32
_METRIC_REVISIONS = {
    "literal_exact_match": "about-llm.literal-exact-match.v1",
    "exact_match": "about-llm.normalized-exact-match.v1",
    "token_f1": "about-llm.token-f1.v1",
}
_MANIFEST_FIELDS = {
    "control_version",
    "checked_at",
    "suite_id",
    "checkpoint_manifest_fingerprint",
    "system_prompt",
    "max_new_tokens",
    "metric_revisions",
    "cases",
    "evidence_boundary",
}
_CASE_FIELDS = {"case_id", "input", "expected", "slices"}
_REPORT_FIELDS = {
    "report_version",
    "checked_at",
    "checkpoint_manifest_fingerprint",
    "suite_fingerprint",
    "source",
    "runtime",
    "model",
    "generation",
    "results",
    "aggregates",
    "scope",
    "evidence_boundary",
    "report_fingerprint",
}
_SOURCE_FIELDS = {
    "model_id",
    "revision",
    "selected_file_count",
    "selected_file_bytes",
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
    "verified_raw_config_semantic_fingerprint",
    "total_parameters",
    "trainable_parameters",
    "parameter_storage_bytes",
    "parameter_dtypes",
    "eval_mode",
}
_GENERATION_FIELDS = {
    "framework",
    "chat_template_applied",
    "batch_size",
    "do_sample",
    "max_new_tokens",
    "use_cache",
    "skip_special_tokens",
    "trust_remote_code",
}
_RESULT_FIELDS = {
    "case_id",
    "slices",
    "prompt_token_count",
    "prompt_token_ids_sha256",
    "generated_token_ids",
    "generated_token_count",
    "ended_with_eos",
    "hit_length_limit",
    "output",
    "output_sha256",
    "literal_exact_match",
    "exact_match",
    "token_f1",
}
_AGGREGATE_FIELDS = {
    "case_count",
    "literal_exact_match_pass_count",
    "literal_exact_match_mean",
    "exact_match_pass_count",
    "exact_match_mean",
    "token_f1_mean",
    "by_slice",
}
_SLICE_AGGREGATE_FIELDS = {
    "case_count",
    "literal_exact_match_pass_count",
    "literal_exact_match_mean",
    "exact_match_pass_count",
    "exact_match_mean",
    "token_f1_mean",
}
_SCOPE_FIELDS = {
    "target_checkpoint_weights_loaded",
    "all_authored_cases_generated",
    "framework_generate_executed",
    "raw_outputs_recorded",
    "deterministic_metrics_and_slices_recomputed",
    "externally_preregistered_or_held_out_suite",
    "representative_benchmark_or_quality_proven",
    "statistical_uncertainty_estimated",
    "judge_model_or_human_raters_executed",
    "system_comparison_or_release_gate_executed",
    "gpu_cuda_or_vllm_executed",
    "tools_rag_or_training_executed",
    "performance_benchmark_performed",
    "publisher_authenticated_by_signature",
    "license_compatibility_proven",
    "production_safety_proven",
    "verification_to_loader_reopen_toctou_eliminated",
}


@dataclass(frozen=True)
class TargetQwenEvaluationCase:
    case_id: str
    input: str
    expected: str
    slices: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "input": self.input,
            "expected": self.expected,
            "slices": list(self.slices),
        }

    def as_evaluation_case(self) -> EvaluationCase:
        return EvaluationCase(
            case_id=self.case_id,
            input=self.input,
            expected=self.expected,
            slices=self.slices,
        )


@dataclass(frozen=True)
class TargetQwenEvaluationSpec:
    checked_at: str
    suite_id: str
    checkpoint_manifest_fingerprint: str
    system_prompt: str
    max_new_tokens: int
    metric_revisions: Mapping[str, str]
    cases: tuple[TargetQwenEvaluationCase, ...]

    def manifest_projection(self) -> dict[str, object]:
        return {
            "control_version": TARGET_QWEN_EVALUATION_CONTROL_VERSION,
            "checked_at": self.checked_at,
            "suite_id": self.suite_id,
            "checkpoint_manifest_fingerprint": self.checkpoint_manifest_fingerprint,
            "system_prompt": self.system_prompt,
            "max_new_tokens": self.max_new_tokens,
            "metric_revisions": dict(self.metric_revisions),
            "cases": [case.to_dict() for case in self.cases],
            "evidence_boundary": TARGET_QWEN_EVALUATION_EVIDENCE_BOUNDARY,
        }

    @property
    def suite_fingerprint(self) -> str:
        return _object_sha256(self.manifest_projection())


def load_target_qwen_evaluation_spec(path: Path) -> TargetQwenEvaluationSpec:
    """Load the closed authored suite and compute its canonical fingerprint."""

    manifest = _load_json_file(path, label="evaluation suite")
    _require_fields(manifest, _MANIFEST_FIELDS, "evaluation suite")
    if manifest.get("control_version") != TARGET_QWEN_EVALUATION_CONTROL_VERSION:
        raise ValueError("evaluation suite control_version is unsupported")
    if manifest.get("checked_at") != TARGET_QWEN_EVALUATION_CHECKED_AT:
        raise ValueError("evaluation suite checked_at drift")
    if manifest.get("evidence_boundary") != TARGET_QWEN_EVALUATION_EVIDENCE_BOUNDARY:
        raise ValueError("evaluation suite evidence_boundary drift")
    suite_id = _nonempty_string(manifest.get("suite_id"), "evaluation suite.suite_id")
    checkpoint_fingerprint = _sha256(
        manifest.get("checkpoint_manifest_fingerprint"),
        "evaluation suite.checkpoint_manifest_fingerprint",
    )
    if checkpoint_fingerprint != TARGET_QWEN_CHECKPOINT_MANIFEST_FINGERPRINT:
        raise ValueError("evaluation suite checkpoint binding drift")
    system_prompt = _bounded_string(
        manifest.get("system_prompt"),
        "evaluation suite.system_prompt",
        maximum=_MAX_PROMPT_CHARACTERS,
    )
    max_new_tokens = _positive_integer(
        manifest.get("max_new_tokens"), "evaluation suite.max_new_tokens"
    )
    if max_new_tokens > _MAX_NEW_TOKENS:
        raise ValueError("evaluation suite max_new_tokens exceeds resource limit")
    metric_revisions = _record(
        manifest.get("metric_revisions"),
        set(_METRIC_REVISIONS),
        "evaluation suite.metric_revisions",
    )
    if metric_revisions != _METRIC_REVISIONS:
        raise ValueError("evaluation suite metric revisions drift")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= _MAX_CASES:
        raise ValueError("evaluation suite cases must be a bounded non-empty array")
    cases: list[TargetQwenEvaluationCase] = []
    seen_case_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _record(raw_case, _CASE_FIELDS, f"evaluation suite.cases[{index}]")
        case_id = _nonempty_string(
            case.get("case_id"), f"evaluation suite.cases[{index}].case_id"
        )
        if _CASE_ID.fullmatch(case_id) is None:
            raise ValueError("evaluation suite case_id has an invalid format")
        if case_id in seen_case_ids:
            raise ValueError("evaluation suite contains duplicate case_id")
        seen_case_ids.add(case_id)
        input_text = _bounded_string(
            case.get("input"),
            f"evaluation suite.cases[{index}].input",
            maximum=_MAX_PROMPT_CHARACTERS,
        )
        expected = _bounded_string(
            case.get("expected"),
            f"evaluation suite.cases[{index}].expected",
            maximum=_MAX_EXPECTED_CHARACTERS,
        )
        raw_slices = case.get("slices")
        if (
            not isinstance(raw_slices, list)
            or not 1 <= len(raw_slices) <= 8
            or not all(
                isinstance(item, str)
                and item
                and item.strip() == item
                and len(item) <= 64
                for item in raw_slices
            )
            or len(raw_slices) != len(set(raw_slices))
        ):
            raise ValueError("evaluation suite slices must be unique bounded strings")
        cases.append(
            TargetQwenEvaluationCase(
                case_id=case_id,
                input=input_text,
                expected=expected,
                slices=tuple(cast(list[str], raw_slices)),
            )
        )
    return TargetQwenEvaluationSpec(
        checked_at=TARGET_QWEN_EVALUATION_CHECKED_AT,
        suite_id=suite_id,
        checkpoint_manifest_fingerprint=checkpoint_fingerprint,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        metric_revisions=dict(_METRIC_REVISIONS),
        cases=tuple(cases),
    )


def execute_loaded_target_qwen_evaluation(
    model: Any,
    tokenizer: Any,
    spec: TargetQwenEvaluationSpec,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Generate every authored case and compute deterministic local metrics."""

    try:
        import torch
        from transformers import GenerationConfig
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("torch and transformers are required for evaluation") from error
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("target tokenizer must provide a chat template")
    if not spec.cases:
        raise ValueError("evaluation spec must contain cases")
    model.to("cpu")
    model.requires_grad_(False)
    model.eval()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    generation_config = GenerationConfig(  # type: ignore[no-untyped-call]
        do_sample=False,
        max_new_tokens=spec.max_new_tokens,
        repetition_penalty=1.0,
        use_cache=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=None,
    )
    results: list[dict[str, object]] = []
    for case in spec.cases:
        messages = [
            {"role": "system", "content": spec.system_prompt},
            {"role": "user", "content": case.input},
        ]
        raw_input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(raw_input_ids, torch.Tensor):
            raise RuntimeError("chat template did not return a tensor")
        input_ids = raw_input_ids.to(device="cpu", dtype=torch.long)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise RuntimeError("chat template returned invalid input shape")
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
        sequences = getattr(generated, "sequences", None)
        if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
            raise RuntimeError("generate returned invalid sequences")
        continuation = sequences[0, input_ids.shape[1] :].to(dtype=torch.long)
        generated_ids = [int(value) for value in continuation.tolist()]
        if not 1 <= len(generated_ids) <= spec.max_new_tokens:
            raise RuntimeError("generate returned an invalid continuation length")
        output = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(output, str) or len(output) > _MAX_OUTPUT_CHARACTERS:
            raise RuntimeError("decoded output is invalid or exceeds the resource limit")
        evaluation_case = case.as_evaluation_case()
        literal_score = float(literal_exact_match(evaluation_case, output))
        exact_score = float(normalized_exact_match(evaluation_case, output))
        f1_score = float(token_f1(evaluation_case, output))
        input_id_list = [int(value) for value in input_ids[0].tolist()]
        ended_with_eos = bool(
            generated_ids and _token_is_eos(generated_ids[-1], eos_token_id)
        )
        results.append(
            {
                "case_id": case.case_id,
                "slices": list(case.slices),
                "prompt_token_count": len(input_id_list),
                "prompt_token_ids_sha256": _object_sha256(
                    {"token_ids": input_id_list}
                ),
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "ended_with_eos": ended_with_eos,
                "hit_length_limit": (
                    len(generated_ids) == spec.max_new_tokens and not ended_with_eos
                ),
                "output": output,
                "output_sha256": _bytes_sha256(output.encode("utf-8")),
                "literal_exact_match": literal_score,
                "exact_match": exact_score,
                "token_f1": f1_score,
            }
        )
    return results, _aggregate_results(spec, results)


def run_target_qwen_evaluation_control(
    checkpoint_spec: CheckpointControlSpec,
    evaluation_spec: TargetQwenEvaluationSpec,
    *,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Verify, load, and execute the reviewed checkpoint on the authored suite."""

    _require_target_checkpoint_spec(checkpoint_spec, evaluation_spec)
    try:
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("torch and transformers are required for evaluation") from error
    snapshot_directory = download_checkpoint_snapshot(
        checkpoint_spec, local_files_only=local_files_only
    )
    snapshot = verify_checkpoint_snapshot(checkpoint_spec, snapshot_directory)
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
        attn_implementation="eager",
        **dtype_argument,
    )
    if (
        type(model).__name__ != "Qwen2ForCausalLM"
        or getattr(model.config, "model_type", None) != "qwen2"
    ):
        raise ValueError("loaded model is not the reviewed Qwen2 target")
    results, aggregates = execute_loaded_target_qwen_evaluation(
        model, tokenizer, evaluation_spec
    )
    parameters = parameter_report(model)
    raw_config = load_model_config_json(snapshot.directory / "config.json")
    config_inspection = inspect_decoder_config(raw_config)
    projection: dict[str, object] = {
        "report_version": TARGET_QWEN_EVALUATION_REPORT_VERSION,
        "checked_at": TARGET_QWEN_EVALUATION_CHECKED_AT,
        "checkpoint_manifest_fingerprint": checkpoint_spec.manifest_fingerprint,
        "suite_fingerprint": evaluation_spec.suite_fingerprint,
        "source": {
            "model_id": checkpoint_spec.model_id,
            "revision": checkpoint_spec.revision,
            "selected_file_count": len(snapshot.files),
            "selected_file_bytes": sum(
                cast(int, item["size_bytes"]) for item in snapshot.files
            ),
            "all_selected_file_bytes_verified_before_load": True,
            "loader_input": "verified_local_snapshot_directory",
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": "eager",
            "torch_num_threads": torch.get_num_threads(),
            "cuda_executed": False,
        },
        "model": {
            "class": type(model).__name__,
            "model_type": getattr(model.config, "model_type", None),
            "verified_raw_config_semantic_fingerprint": (
                config_inspection.config_fingerprint
            ),
            "total_parameters": parameters["total_parameters"],
            "trainable_parameters": parameters["trainable_parameters"],
            "parameter_storage_bytes": parameters["parameter_storage_bytes"],
            "parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in model.parameters()}
            ),
            "eval_mode": model.training is False,
        },
        "generation": {
            "framework": "transformers.GenerationMixin.generate",
            "chat_template_applied": True,
            "batch_size": 1,
            "do_sample": False,
            "max_new_tokens": evaluation_spec.max_new_tokens,
            "use_cache": True,
            "skip_special_tokens": True,
            "trust_remote_code": False,
        },
        "results": results,
        "aggregates": aggregates,
        "scope": {
            "target_checkpoint_weights_loaded": True,
            "all_authored_cases_generated": True,
            "framework_generate_executed": True,
            "raw_outputs_recorded": True,
            "deterministic_metrics_and_slices_recomputed": True,
            "externally_preregistered_or_held_out_suite": False,
            "representative_benchmark_or_quality_proven": False,
            "statistical_uncertainty_estimated": False,
            "judge_model_or_human_raters_executed": False,
            "system_comparison_or_release_gate_executed": False,
            "gpu_cuda_or_vllm_executed": False,
            "tools_rag_or_training_executed": False,
            "performance_benchmark_performed": False,
            "publisher_authenticated_by_signature": False,
            "license_compatibility_proven": False,
            "production_safety_proven": False,
            "verification_to_loader_reopen_toctou_eliminated": False,
        },
        "evidence_boundary": TARGET_QWEN_EVALUATION_EVIDENCE_BOUNDARY,
    }
    projection["report_fingerprint"] = _object_sha256(projection)
    return projection


def verify_recorded_target_qwen_evaluation_report(
    path: Path,
    spec: TargetQwenEvaluationSpec,
    *,
    expected_report_fingerprint: str = (
        TARGET_QWEN_EVALUATION_RECORDED_REPORT_FINGERPRINT
    ),
) -> Mapping[str, Any]:
    """Verify schema, reviewed identity, metrics, aggregates, scope, and hash."""

    report = _load_json_file(path, label="evaluation report")
    _require_fields(report, _REPORT_FIELDS, "evaluation report")
    if report.get("report_version") != TARGET_QWEN_EVALUATION_REPORT_VERSION:
        raise ValueError("evaluation report version drift")
    if report.get("checked_at") != TARGET_QWEN_EVALUATION_CHECKED_AT:
        raise ValueError("evaluation report checked_at drift")
    if report.get("evidence_boundary") != TARGET_QWEN_EVALUATION_EVIDENCE_BOUNDARY:
        raise ValueError("evaluation report evidence_boundary drift")
    if report.get("checkpoint_manifest_fingerprint") != (
        TARGET_QWEN_CHECKPOINT_MANIFEST_FINGERPRINT
    ):
        raise ValueError("evaluation report checkpoint binding drift")
    if report.get("suite_fingerprint") != spec.suite_fingerprint:
        raise ValueError("evaluation report suite binding drift")
    fingerprint = _sha256(
        report.get("report_fingerprint"), "evaluation report.report_fingerprint"
    )
    projection = dict(report)
    del projection["report_fingerprint"]
    if not hmac.compare_digest(fingerprint, _object_sha256(projection)):
        raise ValueError("evaluation report fingerprint mismatch")
    _sha256(expected_report_fingerprint, "expected report fingerprint")
    if not hmac.compare_digest(fingerprint, expected_report_fingerprint):
        raise ValueError("evaluation report is not the reviewed recording")

    source = _record(report.get("source"), _SOURCE_FIELDS, "evaluation report.source")
    if source != {
        "model_id": TARGET_QWEN_MODEL_ID,
        "revision": TARGET_QWEN_REVISION,
        "selected_file_count": TARGET_QWEN_SELECTED_FILE_COUNT,
        "selected_file_bytes": TARGET_QWEN_SELECTED_FILE_BYTES,
        "all_selected_file_bytes_verified_before_load": True,
        "loader_input": "verified_local_snapshot_directory",
    }:
        raise ValueError("evaluation report source drift")
    runtime = _record(
        report.get("runtime"), _RUNTIME_FIELDS, "evaluation report.runtime"
    )
    for name in (
        "python_implementation",
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
    ):
        _nonempty_string(runtime.get(name), f"evaluation report.runtime.{name}")
    if not (
        runtime.get("device") == "cpu"
        and runtime.get("dtype") == "float32"
        and runtime.get("attention_implementation") == "eager"
        and runtime.get("cuda_executed") is False
    ):
        raise ValueError("evaluation report runtime is not CPU FP32 eager")
    _positive_integer(runtime.get("torch_num_threads"), "runtime.torch_num_threads")
    model = _record(report.get("model"), _MODEL_FIELDS, "evaluation report.model")
    if not (
        model.get("class") == "Qwen2ForCausalLM"
        and model.get("model_type") == "qwen2"
        and model.get("verified_raw_config_semantic_fingerprint")
        == "sha256:ee6f9831a4c4729cf094af9a76a53dfe1dde8e34a8251889f527d2179c7d918d"
        and model.get("total_parameters") == TARGET_QWEN_TOTAL_PARAMETERS
        and model.get("trainable_parameters") == 0
        and model.get("parameter_storage_bytes")
        == TARGET_QWEN_PARAMETER_STORAGE_BYTES
        and model.get("parameter_dtypes") == ["torch.float32"]
        and model.get("eval_mode") is True
    ):
        raise ValueError("evaluation report model identity/state drift")
    generation = _record(
        report.get("generation"),
        _GENERATION_FIELDS,
        "evaluation report.generation",
    )
    if generation != {
        "framework": "transformers.GenerationMixin.generate",
        "chat_template_applied": True,
        "batch_size": 1,
        "do_sample": False,
        "max_new_tokens": spec.max_new_tokens,
        "use_cache": True,
        "skip_special_tokens": True,
        "trust_remote_code": False,
    }:
        raise ValueError("evaluation report generation contract drift")
    results = _validate_and_rescore_results(report.get("results"), spec)
    expected_aggregates = _aggregate_results(spec, results)
    aggregates = _record(
        report.get("aggregates"),
        _AGGREGATE_FIELDS,
        "evaluation report.aggregates",
    )
    if aggregates != expected_aggregates:
        raise ValueError("evaluation report aggregate arithmetic drift")
    scope = _record(report.get("scope"), _SCOPE_FIELDS, "evaluation report.scope")
    expected_true = {
        "target_checkpoint_weights_loaded",
        "all_authored_cases_generated",
        "framework_generate_executed",
        "raw_outputs_recorded",
        "deterministic_metrics_and_slices_recomputed",
    }
    for name, value in scope.items():
        if value is not (name in expected_true):
            raise ValueError(f"evaluation report.scope.{name} drift")
    return report


def _validate_and_rescore_results(
    value: Any, spec: TargetQwenEvaluationSpec
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(spec.cases):
        raise ValueError("evaluation report results length drift")
    checked: list[dict[str, object]] = []
    for index, (raw_result, case) in enumerate(zip(value, spec.cases, strict=True)):
        result = _record(
            raw_result, _RESULT_FIELDS, f"evaluation report.results[{index}]"
        )
        if result.get("case_id") != case.case_id or result.get("slices") != list(
            case.slices
        ):
            raise ValueError("evaluation report result case identity drift")
        _positive_integer(result.get("prompt_token_count"), "result.prompt_token_count")
        _sha256(result.get("prompt_token_ids_sha256"), "result.prompt_token_ids")
        token_ids = result.get("generated_token_ids")
        if (
            not isinstance(token_ids, list)
            or not 1 <= len(token_ids) <= spec.max_new_tokens
            or not all(
                isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and 0 <= token_id < 151_936
                for token_id in token_ids
            )
            or result.get("generated_token_count") != len(token_ids)
        ):
            raise ValueError("evaluation report generated-token contract drift")
        ended_with_eos = result.get("ended_with_eos")
        hit_length_limit = result.get("hit_length_limit")
        if not isinstance(ended_with_eos, bool) or not isinstance(hit_length_limit, bool):
            raise ValueError("evaluation report terminal flags must be boolean")
        if hit_length_limit is not (
            len(token_ids) == spec.max_new_tokens and not ended_with_eos
        ):
            raise ValueError("evaluation report hit_length_limit drift")
        output = result.get("output")
        if not isinstance(output, str) or len(output) > _MAX_OUTPUT_CHARACTERS:
            raise ValueError("evaluation report output is invalid")
        if result.get("output_sha256") != _bytes_sha256(output.encode("utf-8")):
            raise ValueError("evaluation report output hash drift")
        evaluation_case = case.as_evaluation_case()
        literal_score = float(literal_exact_match(evaluation_case, output))
        exact_score = float(normalized_exact_match(evaluation_case, output))
        f1_score = float(token_f1(evaluation_case, output))
        if (
            result.get("literal_exact_match") != literal_score
            or result.get("exact_match") != exact_score
            or result.get("token_f1") != f1_score
        ):
            raise ValueError("evaluation report deterministic score drift")
        checked.append(dict(result))
    return checked


def _aggregate_results(
    spec: TargetQwenEvaluationSpec,
    results: list[dict[str, object]],
) -> dict[str, object]:
    if len(results) != len(spec.cases):
        raise ValueError("cannot aggregate incomplete evaluation results")
    literal_scores = [
        float(cast(float, result["literal_exact_match"])) for result in results
    ]
    exact_scores = [float(cast(float, result["exact_match"])) for result in results]
    f1_scores = [float(cast(float, result["token_f1"])) for result in results]
    by_slice: dict[str, dict[str, object]] = {}
    slice_names = sorted({slice_name for case in spec.cases for slice_name in case.slices})
    for slice_name in slice_names:
        selected = [
            result
            for result, case in zip(results, spec.cases, strict=True)
            if slice_name in case.slices
        ]
        selected_exact = [
            float(cast(float, result["exact_match"])) for result in selected
        ]
        selected_literal = [
            float(cast(float, result["literal_exact_match"])) for result in selected
        ]
        selected_f1 = [float(cast(float, result["token_f1"])) for result in selected]
        by_slice[slice_name] = {
            "case_count": len(selected),
            "literal_exact_match_pass_count": sum(
                score == 1.0 for score in selected_literal
            ),
            "literal_exact_match_mean": math.fsum(selected_literal)
            / len(selected_literal),
            "exact_match_pass_count": sum(score == 1.0 for score in selected_exact),
            "exact_match_mean": math.fsum(selected_exact) / len(selected_exact),
            "token_f1_mean": math.fsum(selected_f1) / len(selected_f1),
        }
    return {
        "case_count": len(results),
        "literal_exact_match_pass_count": sum(
            score == 1.0 for score in literal_scores
        ),
        "literal_exact_match_mean": math.fsum(literal_scores) / len(literal_scores),
        "exact_match_pass_count": sum(score == 1.0 for score in exact_scores),
        "exact_match_mean": math.fsum(exact_scores) / len(exact_scores),
        "token_f1_mean": math.fsum(f1_scores) / len(f1_scores),
        "by_slice": by_slice,
    }


def _require_target_checkpoint_spec(
    checkpoint_spec: CheckpointControlSpec,
    evaluation_spec: TargetQwenEvaluationSpec,
) -> None:
    if not isinstance(checkpoint_spec, CheckpointControlSpec):
        raise TypeError("checkpoint_spec must be CheckpointControlSpec")
    if (
        checkpoint_spec.model_id != TARGET_QWEN_MODEL_ID
        or checkpoint_spec.revision != TARGET_QWEN_REVISION
        or checkpoint_spec.manifest_fingerprint
        != TARGET_QWEN_CHECKPOINT_MANIFEST_FINGERPRINT
        or checkpoint_spec.expected_model_class != "Qwen2ForCausalLM"
        or checkpoint_spec.expected_model_type != "qwen2"
        or checkpoint_spec.device != "cpu"
        or checkpoint_spec.dtype != "float32"
        or checkpoint_spec.attention_implementation != "eager"
        or evaluation_spec.checkpoint_manifest_fingerprint
        != checkpoint_spec.manifest_fingerprint
    ):
        raise ValueError("control is not bound to the reviewed Qwen checkpoint")


def _token_is_eos(token_id: int, eos_token_id: Any) -> bool:
    if isinstance(eos_token_id, int) and not isinstance(eos_token_id, bool):
        return token_id == eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        return token_id in eos_token_id
    return False


def _load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    with path.open("rb") as stream:
        payload = stream.read(_MAX_FILE_BYTES + 1)
    if len(payload) > _MAX_FILE_BYTES:
        raise ValueError(f"{label} exceeds byte limit")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _require_fields(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name}: field set mismatch")


def _record(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    _require_fields(value, fields, name)
    return cast(dict[str, Any], value)


def _nonempty_string(value: Any, name: str) -> str:
    return _bounded_string(value, name, maximum=_MAX_PROMPT_CHARACTERS)


def _bounded_string(value: Any, name: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > maximum
    ):
        raise ValueError(f"{name} must be a bounded non-empty trimmed string")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return cast(int, value)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256 fingerprint")
    return value


def _bytes_sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _object_sha256(value: Any) -> str:
    return "sha256:" + artifact_fingerprint(value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def target_qwen_evaluation_canonical_json(report: Mapping[str, Any]) -> bytes:
    """Expose canonical bytes for tests and artifact tooling."""

    return canonical_json_bytes(dict(report)) + b"\n"
