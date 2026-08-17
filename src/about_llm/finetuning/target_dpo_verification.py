"""Recorded-evidence verification for the target-checkpoint DPO control."""

from __future__ import annotations

import hmac
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from about_llm.finetuning.target_dpo_contract import (
    TARGET_DPO_EVIDENCE_BOUNDARY,
    TARGET_DPO_REPORT_VERSION,
    TargetDPOControlSpec,
    _digest,
    _exact_keys,
    _mapping,
    _number,
    _strict_json_object,
)
from about_llm.finetuning.target_dpo_control import (
    _verified_preference_inputs,
)
from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    verify_recorded_checkpoint_report,
)
from about_llm.llmops import artifact_fingerprint


def verify_recorded_target_dpo_report(
    path: Path,
    *,
    spec: TargetDPOControlSpec,
    checkpoint_spec: CheckpointControlSpec,
    checkpoint_report_path: Path,
    training_path: Path,
    readiness_path: Path,
) -> Mapping[str, Any]:
    """Verify recorded identities, semantics, and non-overclaiming without replaying DPO."""

    if (
        spec.model_id,
        spec.revision,
        spec.source_checkpoint_manifest_fingerprint,
    ) != (
        checkpoint_spec.model_id,
        checkpoint_spec.revision,
        checkpoint_spec.manifest_fingerprint,
    ):
        raise ValueError("recorded target DPO checkpoint binding mismatch")
    checkpoint_report = verify_recorded_checkpoint_report(
        checkpoint_report_path,
        expected_manifest_fingerprint=checkpoint_spec.manifest_fingerprint,
    )
    if checkpoint_report.get("report_fingerprint") != (
        spec.source_checkpoint_report_fingerprint
    ):
        raise ValueError("recorded target DPO source report fingerprint mismatch")
    training_file, readiness_file, records, audit, readiness = (
        _verified_preference_inputs(
            spec, training_path=training_path, readiness_path=readiness_path
        )
    )

    report = _strict_json_object(path, label="recorded target DPO report")
    _exact_keys(
        report,
        {
            "report_version",
            "control_manifest_fingerprint",
            "checked_at",
            "source",
            "data",
            "runtime",
            "trainer_contract",
            "result",
            "scope",
            "evidence_boundary",
            "report_fingerprint",
        },
        "recorded target DPO report",
    )
    supplied = _digest(report["report_fingerprint"], "report.report_fingerprint")
    unsigned = {key: value for key, value in report.items() if key != "report_fingerprint"}
    expected = "sha256:" + artifact_fingerprint(unsigned)
    if not hmac.compare_digest(supplied, expected):
        raise ValueError("recorded target DPO report fingerprint mismatch")
    if (
        report["report_version"],
        report["control_manifest_fingerprint"],
        report["checked_at"],
        report["evidence_boundary"],
    ) != (
        TARGET_DPO_REPORT_VERSION,
        spec.manifest_fingerprint,
        spec.checked_at,
        TARGET_DPO_EVIDENCE_BOUNDARY,
    ):
        raise ValueError("recorded target DPO top-level identity drifted")

    checkpoint_artifacts = _mapping(
        checkpoint_report.get("artifacts"), "checkpoint report artifacts"
    )
    _exact_keys(
        checkpoint_artifacts,
        {"files", "selected_file_count", "selected_total_bytes"},
        "checkpoint report artifacts",
    )
    source = _mapping(report["source"], "report.source")
    expected_source = {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "checkpoint_manifest_fingerprint": (
            spec.source_checkpoint_manifest_fingerprint
        ),
        "checkpoint_report_fingerprint": spec.source_checkpoint_report_fingerprint,
        "loader_input": "verified_local_snapshot_directory",
        "all_selected_checkpoint_file_bytes_verified_before_load": True,
        "selected_checkpoint_files": checkpoint_artifacts["files"],
        "selected_checkpoint_total_bytes": checkpoint_artifacts[
            "selected_total_bytes"
        ],
    }
    if source != expected_source:
        raise ValueError("recorded target DPO source evidence drifted")

    data = _mapping(report["data"], "report.data")
    expected_data = {
        "training_artifact": training_file,
        "readiness_artifact": readiness_file,
        "record_ids": [record.record_id for record in records],
        "training_ordered_dataset_fingerprint": audit.ordered_dataset_fingerprint,
        "training_manifest_fingerprint": audit.manifest_fingerprint,
        "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
        "readiness_gate_passed": True,
        "held_out_plaintext_embedded": False,
        "authored_fixture_labels": True,
    }
    if readiness.gate_passed is not True or data != expected_data:
        raise ValueError("recorded target DPO data evidence drifted")

    runtime = _mapping(report["runtime"], "report.runtime")
    expected_runtime = {
        "python_implementation": spec.runtime.python_implementation,
        "python_version": spec.runtime.python_version,
        "platform": spec.runtime.platform,
        "torch_version": spec.runtime.torch_version,
        "transformers_version": spec.runtime.transformers_version,
        "peft_version": spec.runtime.peft_version,
        "trl_version": spec.trainer.trl_version,
        "device": spec.runtime.device,
        "dtype": spec.runtime.dtype,
        "attention_implementation": spec.runtime.attention_implementation,
        "torch_num_threads": spec.runtime.torch_num_threads,
        "cuda_executed": False,
    }
    if runtime != expected_runtime:
        raise ValueError("recorded target DPO runtime drifted")

    trainer_contract = _mapping(report["trainer_contract"], "report.trainer_contract")
    expected_trainer_contract = {
        "loss_type": spec.trainer.loss_type,
        "beta": spec.trainer.beta,
        "max_length": spec.trainer.max_length,
        "batch_size": spec.trainer.batch_size,
        "gradient_accumulation_steps": spec.trainer.gradient_accumulation_steps,
        "learning_rate": spec.trainer.learning_rate,
        "max_steps": spec.trainer.max_steps,
        "gradient_checkpointing": spec.trainer.gradient_checkpointing,
        "optimizer": spec.trainer.optimizer,
        "reference_mode": spec.trainer.reference_mode,
        "target_modules": list(spec.adapter.target_modules),
        "rank": spec.adapter.rank,
        "alpha": spec.adapter.alpha,
        "dropout": spec.adapter.dropout,
        "torch_seed": spec.torch_seed,
    }
    if trainer_contract != expected_trainer_contract:
        raise ValueError("recorded target DPO trainer contract drifted")

    result = _mapping(report["result"], "report.result")
    _exact_keys(result, {"tokenization", "model", "execution"}, "report.result")
    tokenization = _mapping(result["tokenization"], "report.result.tokenization")
    expected_pairs = [
        {
            "record_id": item.record_id,
            "prompt_ids": list(item.prompt_ids),
            "chosen_ids": list(item.chosen_ids),
            "rejected_ids": list(item.rejected_ids),
        }
        for item in spec.expected_pairs
    ]
    collated_length = max(
        len(item.prompt_ids) + len(completion)
        for item in spec.expected_pairs
        for completion in (item.chosen_ids, item.rejected_ids)
    )
    prompt_counts = [
        len(item.prompt_ids)
        for item in spec.expected_pairs
        for _ in (item.chosen_ids, item.rejected_ids)
    ]
    completion_counts = [
        len(completion)
        for item in spec.expected_pairs
        for completion in (item.chosen_ids, item.rejected_ids)
    ]
    expected_tokenization = {
        "pair_count": len(spec.expected_pairs),
        "pairs": expected_pairs,
        "collated_input_shape": [len(spec.expected_pairs) * 2, collated_length],
        "collated_attention_token_count": sum(prompt_counts) + sum(completion_counts),
        "prompt_masked_token_counts": prompt_counts,
        "completion_token_counts": completion_counts,
        "chosen_rows_first": True,
        "rejected_rows_second": True,
        "silent_truncation_observed": False,
    }
    if tokenization != expected_tokenization:
        raise ValueError("recorded target DPO tokenization evidence drifted")

    model = _mapping(result["model"], "report.result.model")
    source_special_tokens = {
        "model_config": {
            "bos_token_id": spec.model_contract.config_bos_token_id,
            "eos_token_id": spec.model_contract.config_eos_token_id,
            "pad_token_id": spec.model_contract.config_pad_token_id,
        },
        "generation_config": {
            "bos_token_id": spec.model_contract.generation_config_bos_token_id,
            "eos_token_ids": list(
                spec.model_contract.generation_config_eos_token_ids
            ),
            "pad_token_id": spec.model_contract.generation_config_pad_token_id,
        },
        "tokenizer": {
            "bos_token_id": spec.model_contract.tokenizer_bos_token_id,
            "eos_token_id": spec.model_contract.tokenizer_eos_token_id,
            "pad_token_id": spec.model_contract.tokenizer_pad_token_id,
        },
    }
    normalized_generation_eos = list(
        spec.model_contract.generation_config_eos_token_ids
    )
    tokenizer_eos_token_id = spec.model_contract.tokenizer_eos_token_id
    if (
        tokenizer_eos_token_id is not None
        and tokenizer_eos_token_id not in normalized_generation_eos
    ):
        normalized_generation_eos.insert(
            0, tokenizer_eos_token_id
        )
    normalized_special_tokens = {
        "model_config": {
            "bos_token_id": spec.model_contract.tokenizer_bos_token_id,
            "eos_token_id": spec.model_contract.tokenizer_eos_token_id,
            "pad_token_id": spec.model_contract.tokenizer_pad_token_id,
        },
        "generation_config": {
            "bos_token_id": spec.model_contract.tokenizer_bos_token_id,
            "eos_token_ids": normalized_generation_eos,
            "pad_token_id": spec.model_contract.tokenizer_pad_token_id,
        },
        "tokenizer": source_special_tokens["tokenizer"],
    }
    expected_model = {
        "class": spec.model_contract.expected_model_class,
        "model_type": spec.model_contract.expected_model_type,
        "num_hidden_layers": spec.model_contract.num_hidden_layers,
        "hidden_size": spec.model_contract.hidden_size,
        "base_parameter_count": spec.model_contract.base_parameter_count,
        "adapter_parameter_count": spec.adapter.expected_trainable_parameters,
        "total_parameter_count_with_adapter": (
            spec.model_contract.base_parameter_count
            + spec.adapter.expected_trainable_parameters
        ),
        "trainable_tensor_count": spec.adapter.expected_trainable_tensor_count,
        "parameter_dtypes": ["torch.float32"],
        "special_token_contract": {
            "source_before_normalization": source_special_tokens,
            "normalized_before_baseline": normalized_special_tokens,
            "after_trainer_alignment_preflight": normalized_special_tokens,
            "after_trainer": normalized_special_tokens,
        },
    }
    if model != expected_model:
        raise ValueError("recorded target DPO model evidence drifted")

    execution = _mapping(result["execution"], "report.result.execution")
    _exact_keys(
        execution,
        {
            "initial_policy_logps",
            "initial_reference_logps",
            "initial_policy_reference_max_abs_error",
            "initial_relative_margins",
            "initial_trainer_loss",
            "expected_initial_loss_log_2",
            "final_policy_logps",
            "final_reference_logps",
            "final_relative_margins",
            "final_trainer_loss",
            "trainer_reported_pre_step_loss",
            "global_step",
            "backward_executed",
            "optimizer_step_count",
            "gradient_callback_count",
            "trainable_gradient_tensor_count",
            "finite_gradient_tensor_count",
            "frozen_base_gradient_tensor_count",
            "frozen_base_parameter_fingerprint_before",
            "frozen_base_parameter_fingerprint_after",
            "frozen_base_parameters_unchanged",
            "frozen_non_adapter_state_fingerprint_before",
            "frozen_non_adapter_state_fingerprint_after",
            "frozen_non_adapter_state_unchanged",
            "normalized_model_config_fingerprint_before",
            "normalized_model_config_fingerprint_after",
            "normalized_model_config_unchanged",
            "normalized_generation_config_fingerprint_before",
            "normalized_generation_config_fingerprint_after",
            "normalized_generation_config_unchanged",
            "reference_logps_sha256_before",
            "reference_logps_sha256_after",
            "reference_replay_max_abs_error_after_step",
            "reference_replay_bitwise_equal",
            "reference_replay_drift_reported_not_equated_to_weight_drift",
            "initial_adapter_disable_audit",
            "final_adapter_disable_audit",
            "adapter_before",
            "adapter_after",
        },
        "report.result.execution",
    )

    row_count = len(spec.expected_pairs) * 2

    def number_list(key: str, *, expected_length: int = row_count) -> list[float]:
        value = execution[key]
        if not isinstance(value, list) or len(value) != expected_length:
            raise ValueError(f"report.result.execution.{key} length drifted")
        return [
            _number(item, f"report.result.execution.{key}[{index}]")
            for index, item in enumerate(value)
        ]

    initial_policy = number_list("initial_policy_logps")
    initial_reference = number_list("initial_reference_logps")
    final_policy = number_list("final_policy_logps")
    final_reference = number_list("final_reference_logps")
    initial_margins = number_list(
        "initial_relative_margins", expected_length=len(spec.expected_pairs)
    )
    final_margins = number_list(
        "final_relative_margins", expected_length=len(spec.expected_pairs)
    )
    if initial_policy != initial_reference:
        raise ValueError("recorded target DPO reference invariants drifted")
    recomputed_reference_error = max(
        abs(before - after)
        for before, after in zip(initial_reference, final_reference, strict=True)
    )
    recorded_reference_error = _number(
        execution["reference_replay_max_abs_error_after_step"],
        "reference_replay_max_abs_error_after_step",
    )
    if (
        not math.isclose(
            recorded_reference_error,
            recomputed_reference_error,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or execution["reference_replay_bitwise_equal"]
        is not (recorded_reference_error == 0.0)
        or execution[
            "reference_replay_drift_reported_not_equated_to_weight_drift"
        ]
        is not True
    ):
        raise ValueError("recorded target DPO reference replay evidence drifted")
    expected_adapter_disable_audit = {
        "enabled_before": True,
        "enabled_inside_reference": False,
        "enabled_after": True,
        "active_adapters_before": ["default"],
        "active_adapters_inside_reference": ["default"],
        "active_adapters_after": ["default"],
        "merged_adapters_before": [],
        "merged_adapters_inside_reference": [],
        "merged_adapters_after": [],
    }
    if (
        execution["initial_adapter_disable_audit"]
        != expected_adapter_disable_audit
        or execution["final_adapter_disable_audit"]
        != expected_adapter_disable_audit
    ):
        raise ValueError("recorded target DPO adapter-disable evidence drifted")
    if initial_margins != [0.0] * len(spec.expected_pairs):
        raise ValueError("recorded target DPO initial margins drifted")
    recomputed_margins = [
        (final_policy[index] - final_reference[index])
        - (final_policy[index + len(spec.expected_pairs)]
           - final_reference[index + len(spec.expected_pairs)])
        for index in range(len(spec.expected_pairs))
    ]
    if any(
        not math.isclose(actual, expected_margin, rel_tol=0.0, abs_tol=1e-7)
        or actual <= 0
        for actual, expected_margin in zip(
            final_margins, recomputed_margins, strict=True
        )
    ):
        raise ValueError("recorded target DPO final margins drifted")
    initial_loss = _number(execution["initial_trainer_loss"], "initial_trainer_loss")
    expected_log_2 = _number(
        execution["expected_initial_loss_log_2"], "expected_initial_loss_log_2"
    )
    final_loss = _number(execution["final_trainer_loss"], "final_trainer_loss")
    reported_loss = _number(
        execution["trainer_reported_pre_step_loss"],
        "trainer_reported_pre_step_loss",
    )
    recomputed_final_loss = sum(
        math.log1p(math.exp(-spec.trainer.beta * margin))
        for margin in final_margins
    ) / len(final_margins)
    if (
        execution["initial_policy_reference_max_abs_error"] != 0.0
        or not math.isclose(expected_log_2, math.log(2.0), rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(initial_loss, math.log(2.0), rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(reported_loss, initial_loss, rel_tol=0.0, abs_tol=1e-6)
        or not math.isclose(final_loss, recomputed_final_loss, rel_tol=0.0, abs_tol=1e-6)
        or final_loss >= initial_loss
    ):
        raise ValueError("recorded target DPO loss evidence drifted")

    if (
        execution["global_step"],
        execution["backward_executed"],
        execution["optimizer_step_count"],
        execution["gradient_callback_count"],
        execution["trainable_gradient_tensor_count"],
        execution["finite_gradient_tensor_count"],
        execution["frozen_base_gradient_tensor_count"],
        execution["frozen_base_parameters_unchanged"],
        execution["reference_replay_drift_reported_not_equated_to_weight_drift"],
    ) != (
        spec.trainer.max_steps,
        True,
        spec.trainer.max_steps,
        spec.trainer.max_steps,
        spec.adapter.expected_trainable_tensor_count,
        spec.adapter.expected_trainable_tensor_count,
        0,
        True,
        True,
    ):
        raise ValueError("recorded target DPO optimizer evidence drifted")
    frozen_before = _digest(
        execution["frozen_base_parameter_fingerprint_before"],
        "frozen_base_parameter_fingerprint_before",
    )
    frozen_after = _digest(
        execution["frozen_base_parameter_fingerprint_after"],
        "frozen_base_parameter_fingerprint_after",
    )
    reference_before = _digest(
        execution["reference_logps_sha256_before"], "reference_logps_sha256_before"
    )
    reference_after = _digest(
        execution["reference_logps_sha256_after"], "reference_logps_sha256_after"
    )
    frozen_state_before = _digest(
        execution["frozen_non_adapter_state_fingerprint_before"],
        "frozen_non_adapter_state_fingerprint_before",
    )
    frozen_state_after = _digest(
        execution["frozen_non_adapter_state_fingerprint_after"],
        "frozen_non_adapter_state_fingerprint_after",
    )
    model_config_before = _digest(
        execution["normalized_model_config_fingerprint_before"],
        "normalized_model_config_fingerprint_before",
    )
    model_config_after = _digest(
        execution["normalized_model_config_fingerprint_after"],
        "normalized_model_config_fingerprint_after",
    )
    generation_config_before = _digest(
        execution["normalized_generation_config_fingerprint_before"],
        "normalized_generation_config_fingerprint_before",
    )
    generation_config_after = _digest(
        execution["normalized_generation_config_fingerprint_after"],
        "normalized_generation_config_fingerprint_after",
    )
    if (
        not hmac.compare_digest(frozen_before, frozen_after)
        or not hmac.compare_digest(frozen_state_before, frozen_state_after)
        or not hmac.compare_digest(model_config_before, model_config_after)
        or not hmac.compare_digest(
            generation_config_before, generation_config_after
        )
        or execution["frozen_non_adapter_state_unchanged"] is not True
        or execution["normalized_model_config_unchanged"] is not True
        or execution["normalized_generation_config_unchanged"] is not True
    ):
        raise ValueError("recorded target DPO frozen identity drifted")
    reference_hashes_equal = hmac.compare_digest(reference_before, reference_after)
    if reference_hashes_equal is not (recorded_reference_error == 0.0):
        raise ValueError("recorded target DPO reference replay hashes drifted")
    adapter_before = _mapping(execution["adapter_before"], "execution.adapter_before")
    adapter_after = _mapping(execution["adapter_after"], "execution.adapter_after")
    expected_adapter_before = {
        "a_tensor_count": spec.adapter.expected_a_tensor_count,
        "b_tensor_count": spec.adapter.expected_b_tensor_count,
        "nonzero_a_tensor_count": spec.adapter.expected_a_tensor_count,
        "nonzero_b_tensor_count": 0,
        "nonzero_b_element_count": 0,
    }
    expected_adapter_after = {
        "a_tensor_count": spec.adapter.expected_a_tensor_count,
        "b_tensor_count": spec.adapter.expected_b_tensor_count,
        "nonzero_a_tensor_count": spec.adapter.expected_a_tensor_count,
        "nonzero_b_tensor_count": spec.adapter.expected_b_tensor_count,
        "nonzero_b_element_count": spec.adapter.expected_b_element_count,
    }
    if adapter_before != expected_adapter_before or adapter_after != expected_adapter_after:
        raise ValueError("recorded target DPO adapter evidence drifted")

    scope = _mapping(report["scope"], "report.scope")
    expected_scope = {
        "target_checkpoint_weights_loaded": True,
        "real_trl_dpo_trainer_executed": True,
        "real_backward_and_optimizer_step_executed": True,
        "adapter_disabled_reference_executed": True,
        "target_tokenizer_and_chat_template_executed": True,
        "authored_fixture_label_validity_proven": False,
        "human_preference_or_annotator_evidence": False,
        "alignment_quality_or_generalization_proven": False,
        "safety_alignment_proven": False,
        "convergence_proven": False,
        "qlora_cuda_or_vllm_executed": False,
        "performance_or_memory_benchmark_performed": False,
        "publisher_or_trainer_authenticated": False,
        "verification_to_loader_reopen_toctou_eliminated": False,
        "production_readiness_proven": False,
    }
    if scope != expected_scope:
        raise ValueError("recorded target DPO scope drifted or overclaims evidence")
    return report
