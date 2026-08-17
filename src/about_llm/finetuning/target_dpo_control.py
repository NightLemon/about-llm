"""Strict target-checkpoint TRL DPO execution and recorded-evidence control."""

from __future__ import annotations

import gc
import hashlib
import hmac
import json
import math
import platform
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from about_llm.finetuning.target_dpo_contract import (
    TARGET_DPO_CONTROL_VERSION as TARGET_DPO_CONTROL_VERSION,
)
from about_llm.finetuning.target_dpo_contract import (
    TARGET_DPO_EVIDENCE_BOUNDARY as TARGET_DPO_EVIDENCE_BOUNDARY,
)
from about_llm.finetuning.target_dpo_contract import (
    TARGET_DPO_REPORT_VERSION as TARGET_DPO_REPORT_VERSION,
)
from about_llm.finetuning.target_dpo_contract import (
    TargetDPOAdapterSpec as TargetDPOAdapterSpec,
)
from about_llm.finetuning.target_dpo_contract import (
    TargetDPOArtifactSpec as TargetDPOArtifactSpec,
)
from about_llm.finetuning.target_dpo_contract import (
    TargetDPOControlSpec as TargetDPOControlSpec,
)
from about_llm.finetuning.target_dpo_contract import (
    TargetDPOModelSpec as TargetDPOModelSpec,
)
from about_llm.finetuning.target_dpo_contract import (
    TargetDPOPairSpec as TargetDPOPairSpec,
)
from about_llm.finetuning.target_dpo_contract import (
    TargetDPORuntimeSpec as TargetDPORuntimeSpec,
)
from about_llm.finetuning.target_dpo_contract import (
    TargetDPOTrainerSpec as TargetDPOTrainerSpec,
)
from about_llm.finetuning.target_dpo_contract import (
    load_target_dpo_control_spec as load_target_dpo_control_spec,
)
from about_llm.integrations.transformers_checkpoint_control import (
    CheckpointControlSpec,
    download_checkpoint_snapshot,
    verify_checkpoint_snapshot,
    verify_recorded_checkpoint_report,
)
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes


def _verify_file(path: Path, spec: TargetDPOArtifactSpec) -> dict[str, object]:
    if path.name != spec.filename or path.is_symlink() or not path.is_file():
        raise ValueError(f"target DPO artifact path mismatch: {spec.filename}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    actual = "sha256:" + digest.hexdigest()
    if size != spec.size_bytes or not hmac.compare_digest(actual, spec.sha256):
        raise ValueError(f"target DPO artifact bytes drifted: {spec.filename}")
    return {
        "filename": spec.filename,
        "size_bytes": size,
        "sha256": actual,
        "verified": True,
    }


def _verified_preference_inputs(
    spec: TargetDPOControlSpec,
    *,
    training_path: Path,
    readiness_path: Path,
) -> tuple[dict[str, object], dict[str, object], tuple[Any, ...], Any, Any]:
    from about_llm.finetuning import (
        load_preference_records,
        load_preference_training_readiness,
        validate_preference_training_readiness,
    )

    training_file = _verify_file(training_path, spec.training_artifact)
    readiness_file = _verify_file(readiness_path, spec.readiness_artifact)
    records = load_preference_records(training_path)
    readiness = load_preference_training_readiness(readiness_path)
    audit = validate_preference_training_readiness(records, readiness)
    if (
        audit.ordered_dataset_fingerprint
        != spec.training_artifact.ordered_dataset_fingerprint
        or audit.manifest_fingerprint != spec.training_artifact.manifest_fingerprint
        or readiness.manifest_fingerprint
        != spec.readiness_artifact.manifest_fingerprint
    ):
        raise ValueError("target DPO preference/readiness semantic identity drifted")
    if tuple(record.record_id for record in records) != spec.training_artifact.record_ids:
        raise ValueError("target DPO loaded record identity drifted")
    return training_file, readiness_file, records, audit, readiness


def _parameter_fingerprint(parameters: Any) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(parameters, key=lambda item: item[0]):
        tensor = parameter.detach().to(device="cpu").contiguous()
        descriptor = canonical_json_bytes(
            {
                "name": name,
                "shape": [int(value) for value in tensor.shape],
                "dtype": str(tensor.dtype),
            }
        )
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(memoryview(tensor.numpy()).cast("B"))
        count += 1
    if count == 0:
        raise RuntimeError("target DPO parameter fingerprint received no tensors")
    return "sha256:" + digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
            }
        )
    )
    digest.update(memoryview(value.numpy()).cast("B"))
    return "sha256:" + digest.hexdigest()


def _sequence_logps(model: Any, batch: Mapping[str, Any], torch: Any) -> Any:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    logits = output.logits[:, :-1, :].to(dtype=torch.float32)
    labels = batch["input_ids"][:, 1:]
    mask = batch["completion_mask"][:, 1:].to(dtype=torch.float32)
    selected = torch.log_softmax(logits, dim=-1).gather(
        dim=-1, index=labels.unsqueeze(-1)
    )
    return (selected.squeeze(-1) * mask).sum(dim=1).detach().cpu()


def _policy_reference_logps(
    model: Any, batch: Mapping[str, Any], torch: Any
) -> tuple[Any, Any, dict[str, object]]:
    model.eval()
    get_status = getattr(model, "get_model_status", None)
    if get_status is None:
        raise RuntimeError("PEFT model does not expose adapter status")
    before = get_status()
    if before.enabled is not True or list(before.merged_adapters):
        raise RuntimeError("target DPO policy adapter status is not enabled/unmerged")
    with torch.inference_mode():
        policy = _sequence_logps(model, batch, torch)
        disable_adapter = getattr(model, "disable_adapter", None)
        if disable_adapter is None:
            raise RuntimeError("PEFT model does not expose disable_adapter reference mode")
        with disable_adapter():
            inside = get_status()
            if inside.enabled is not False or list(inside.merged_adapters):
                raise RuntimeError("target DPO reference forward did not disable adapters")
            reference = _sequence_logps(model, batch, torch)
    after = get_status()
    if (
        after.enabled is not True
        or list(after.merged_adapters)
        or list(after.active_adapters) != list(before.active_adapters)
    ):
        raise RuntimeError("target DPO adapter status was not restored after reference")
    audit: dict[str, object] = {
        "enabled_before": True,
        "enabled_inside_reference": False,
        "enabled_after": True,
        "active_adapters_before": list(before.active_adapters),
        "active_adapters_inside_reference": list(inside.active_adapters),
        "active_adapters_after": list(after.active_adapters),
        "merged_adapters_before": list(before.merged_adapters),
        "merged_adapters_inside_reference": list(inside.merged_adapters),
        "merged_adapters_after": list(after.merged_adapters),
    }
    return policy, reference, audit


def _non_adapter_state_fingerprint(model: Any) -> str:
    values = (
        (name, tensor)
        for name, tensor in model.state_dict().items()
        if ".lora_A." not in name and ".lora_B." not in name
    )
    return _parameter_fingerprint(values)


def _config_fingerprint(value: Any, label: str) -> str:
    to_dict = getattr(value, "to_dict", None)
    if to_dict is None:
        raise RuntimeError(f"{label} does not expose to_dict")
    projection = to_dict()
    if not isinstance(projection, dict):
        raise RuntimeError(f"{label}.to_dict did not return an object")
    try:
        payload = json.dumps(
            projection,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label}.to_dict is not strict JSON-compatible") from error
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _relative_margins(policy: Any, reference: Any) -> list[float]:
    pair_count = int(policy.shape[0]) // 2
    policy_chosen, policy_rejected = policy[:pair_count], policy[pair_count:]
    ref_chosen, ref_rejected = reference[:pair_count], reference[pair_count:]
    value = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return [float(item) for item in value.tolist()]


def _adapter_summary(model: Any, torch: Any) -> dict[str, int]:
    a_count = b_count = nonzero_a = nonzero_b = nonzero_b_elements = 0
    for name, parameter in model.named_parameters():
        if ".lora_A." in name:
            a_count += 1
            if int(torch.count_nonzero(parameter.detach()).item()) > 0:
                nonzero_a += 1
        elif ".lora_B." in name:
            b_count += 1
            count = int(torch.count_nonzero(parameter.detach()).item())
            if count > 0:
                nonzero_b += 1
            nonzero_b_elements += count
    return {
        "a_tensor_count": a_count,
        "b_tensor_count": b_count,
        "nonzero_a_tensor_count": nonzero_a,
        "nonzero_b_tensor_count": nonzero_b,
        "nonzero_b_element_count": nonzero_b_elements,
    }


def _generation_eos_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return tuple(value)
    raise ValueError("model generation_config.eos_token_id has an unsupported shape")


def _special_token_state(model: Any, tokenizer: Any) -> dict[str, object]:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        raise ValueError("target DPO model must expose generation_config")
    return {
        "model_config": {
            "bos_token_id": getattr(model.config, "bos_token_id", None),
            "eos_token_id": getattr(model.config, "eos_token_id", None),
            "pad_token_id": getattr(model.config, "pad_token_id", None),
        },
        "generation_config": {
            "bos_token_id": getattr(generation_config, "bos_token_id", None),
            "eos_token_ids": list(
                _generation_eos_ids(
                    getattr(generation_config, "eos_token_id", None)
                )
            ),
            "pad_token_id": getattr(generation_config, "pad_token_id", None),
        },
        "tokenizer": {
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        },
    }


def _normalize_special_tokens_for_trainer(model: Any, tokenizer: Any) -> None:
    """Apply the Transformers 4.57 Trainer alignment before baseline measurement."""

    generation_config = model.generation_config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    generation_eos = list(_generation_eos_ids(generation_config.eos_token_id))
    if tokenizer.eos_token_id not in generation_eos:
        generation_eos.insert(0, tokenizer.eos_token_id)
    generation_config.eos_token_id = generation_eos
    generation_config.bos_token_id = tokenizer.bos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id


def execute_loaded_target_dpo_control(
    spec: TargetDPOControlSpec,
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Any],
) -> dict[str, object]:
    """Run the exact one-step DPO contract on an already loaded model."""

    try:
        import torch
        import trl
        from datasets import Dataset, disable_progress_bars  # type: ignore[import-untyped]
        from peft import LoraConfig
        from transformers import TrainerCallback
        from trl.trainer.dpo_config import DPOConfig
        from trl.trainer.dpo_trainer import DPOTrainer
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("torch, transformers, datasets, peft, and trl are required") from error
    if trl.__version__ != spec.trainer.trl_version:
        raise ValueError("installed TRL version differs from target DPO protocol")
    if type(model).__name__ != spec.model_contract.expected_model_class:
        raise ValueError("loaded model class differs from target DPO protocol")
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != spec.model_contract.expected_model_type:
        raise ValueError("loaded model type differs from target DPO protocol")
    if getattr(config, "hidden_size", None) != spec.model_contract.hidden_size:
        raise ValueError("loaded hidden size differs from target DPO protocol")
    if getattr(config, "num_hidden_layers", None) != spec.model_contract.num_hidden_layers:
        raise ValueError("loaded layer count differs from target DPO protocol")
    if (
        getattr(config, "bos_token_id", None),
        getattr(config, "eos_token_id", None),
        getattr(config, "pad_token_id", None),
    ) != (
        spec.model_contract.config_bos_token_id,
        spec.model_contract.config_eos_token_id,
        spec.model_contract.config_pad_token_id,
    ):
        raise ValueError("loaded special-token ids differ from target DPO protocol")
    if (
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ) != (
        spec.model_contract.tokenizer_bos_token_id,
        spec.model_contract.tokenizer_eos_token_id,
        spec.model_contract.tokenizer_pad_token_id,
    ):
        raise ValueError("loaded tokenizer special-token ids differ from target DPO protocol")
    source_special_token_state = _special_token_state(model, tokenizer)
    source_generation = cast(
        Mapping[str, Any], source_special_token_state["generation_config"]
    )
    if (
        source_generation["bos_token_id"],
        tuple(source_generation["eos_token_ids"]),
        source_generation["pad_token_id"],
    ) != (
        spec.model_contract.generation_config_bos_token_id,
        spec.model_contract.generation_config_eos_token_ids,
        spec.model_contract.generation_config_pad_token_id,
    ):
        raise ValueError("loaded generation special-token ids differ from target protocol")
    base_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if base_parameter_count != spec.model_contract.base_parameter_count:
        raise ValueError("loaded base parameter count differs from target DPO protocol")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("target DPO tokenizer must provide a chat template")
    if tuple(getattr(record, "record_id", None) for record in records) != (
        spec.training_artifact.record_ids
    ):
        raise ValueError("target DPO record order differs from control manifest")

    rows: list[dict[str, Any]] = []
    rendered_pairs: list[dict[str, object]] = []
    for record, expected in zip(records, spec.expected_pairs, strict=True):
        row = cast(dict[str, Any], record.to_dpo_row())
        prompt_ids = tuple(
            tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True
            )
        )
        chosen_full = tuple(
            tokenizer.apply_chat_template(
                row["prompt"] + row["chosen"], tokenize=True
            )
        )
        rejected_full = tuple(
            tokenizer.apply_chat_template(
                row["prompt"] + row["rejected"], tokenize=True
            )
        )
        if chosen_full[: len(prompt_ids)] != prompt_ids or rejected_full[
            : len(prompt_ids)
        ] != prompt_ids:
            raise ValueError("target DPO prompt is not an exact completion prefix")
        chosen_ids = chosen_full[len(prompt_ids) :]
        rejected_ids = rejected_full[len(prompt_ids) :]
        if (
            getattr(record, "record_id", None),
            prompt_ids,
            chosen_ids,
            rejected_ids,
        ) != (
            expected.record_id,
            expected.prompt_ids,
            expected.chosen_ids,
            expected.rejected_ids,
        ):
            raise ValueError("target DPO tokenizer output drifted from reviewed protocol")
        if len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)) > (
            spec.trainer.max_length
        ):
            raise ValueError("target DPO pair would be truncated")
        rows.append(row)
        rendered_pairs.append(
            {
                "record_id": expected.record_id,
                "prompt_ids": list(prompt_ids),
                "chosen_ids": list(chosen_ids),
                "rejected_ids": list(rejected_ids),
            }
        )

    torch.set_num_threads(spec.runtime.torch_num_threads)
    torch.manual_seed(spec.torch_seed)
    model.to("cpu")
    model.requires_grad_(False)
    model.config.use_cache = False
    _normalize_special_tokens_for_trainer(model, tokenizer)
    normalized_special_token_state = _special_token_state(model, tokenizer)
    disable_progress_bars()

    class GradientAuditCallback(TrainerCallback):
        trainable_gradient_tensors = 0
        finite_gradient_tensors = 0
        frozen_gradient_tensors = 0
        calls = 0

        def on_pre_optimizer_step(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            del args, state
            callback_model = kwargs.get("model")
            if callback_model is None:
                raise RuntimeError("DPO gradient callback did not receive the model")
            self.calls += 1
            for parameter in callback_model.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    self.trainable_gradient_tensors += 1
                    if bool(torch.isfinite(parameter.grad).all().item()):
                        self.finite_gradient_tensors += 1
                elif not parameter.requires_grad and parameter.grad is not None:
                    self.frozen_gradient_tensors += 1
            return control

    callback = GradientAuditCallback()
    with tempfile.TemporaryDirectory(prefix="about-llm-target-dpo-") as directory:
        trainer_config = DPOConfig(
            output_dir=directory,
            max_length=spec.trainer.max_length,
            beta=spec.trainer.beta,
            loss_type=[spec.trainer.loss_type],
            per_device_train_batch_size=spec.trainer.batch_size,
            gradient_accumulation_steps=spec.trainer.gradient_accumulation_steps,
            learning_rate=spec.trainer.learning_rate,
            max_steps=spec.trainer.max_steps,
            gradient_checkpointing=spec.trainer.gradient_checkpointing,
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
            use_cpu=True,
            dataloader_pin_memory=False,
            optim=spec.trainer.optimizer,
            seed=spec.torch_seed,
            data_seed=spec.torch_seed,
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=trainer_config,
            train_dataset=Dataset.from_list(rows),
            processing_class=tokenizer,
            peft_config=LoraConfig(
                r=spec.adapter.rank,
                lora_alpha=spec.adapter.alpha,
                lora_dropout=spec.adapter.dropout,
                target_modules=list(spec.adapter.target_modules),
                task_type=spec.adapter.task_type,
                bias=cast(
                    Literal["none", "all", "lora_only"], spec.adapter.bias
                ),
            ),
            callbacks=[callback],
        )
        trainer_any: Any = trainer
        align_special_tokens = getattr(trainer_any, "_align_special_tokens", None)
        if align_special_tokens is None:
            raise RuntimeError("Transformers Trainer special-token alignment API is missing")
        align_special_tokens()
        aligned_special_token_state = _special_token_state(trainer_any.model, tokenizer)
        if aligned_special_token_state != normalized_special_token_state:
            raise RuntimeError("Trainer special-token alignment preflight changed the protocol")
        prepared: Any = trainer_any.train_dataset
        if prepared is None or len(prepared) != len(spec.expected_pairs):
            raise RuntimeError("TRL prepared target DPO pair count drifted")
        features = [dict(prepared[index]) for index in range(len(prepared))]
        for feature, expected in zip(features, spec.expected_pairs, strict=True):
            if (
                tuple(feature["prompt_ids"]),
                tuple(feature["chosen_ids"]),
                tuple(feature["rejected_ids"]),
            ) != (expected.prompt_ids, expected.chosen_ids, expected.rejected_ids):
                raise RuntimeError("TRL prepared token ids differ from reviewed protocol")
        batch = trainer_any.data_collator(features)
        pair_count = len(features)
        expected_collated_length = max(
            len(item.prompt_ids) + len(completion_ids)
            for item in spec.expected_pairs
            for completion_ids in (item.chosen_ids, item.rejected_ids)
        )
        if tuple(batch["input_ids"].shape) != (
            pair_count * 2,
            expected_collated_length,
        ):
            raise RuntimeError("target DPO collated batch shape drifted")
        prompt_masked_counts: list[int] = []
        completion_counts: list[int] = []
        for index, feature in enumerate(features):
            prompt_ids = feature["prompt_ids"]
            for row_index, completion_ids in (
                (index, feature["chosen_ids"]),
                (index + pair_count, feature["rejected_ids"]),
            ):
                expected_ids = prompt_ids + completion_ids
                actual_ids = batch["input_ids"][row_index, : len(expected_ids)].tolist()
                if actual_ids != expected_ids:
                    raise RuntimeError("TRL DPO collator reordered target tokens")
                mask = batch["completion_mask"][row_index]
                prompt_nonzero = int(mask[: len(prompt_ids)].count_nonzero().item())
                completion_nonzero = int(
                    mask[
                        len(prompt_ids) : len(prompt_ids) + len(completion_ids)
                    ].sum().item()
                )
                if prompt_nonzero != 0 or completion_nonzero != len(completion_ids):
                    raise RuntimeError("TRL target DPO completion mask drifted")
                prompt_masked_counts.append(len(prompt_ids))
                completion_counts.append(completion_nonzero)

        peft_model: Any = trainer_any.model
        if peft_model is None:
            raise RuntimeError("TRL target DPO trainer did not retain a model")
        trainable = [
            (name, parameter)
            for name, parameter in peft_model.named_parameters()
            if parameter.requires_grad
        ]
        if (
            sum(parameter.numel() for _, parameter in trainable)
            != spec.adapter.expected_trainable_parameters
            or len(trainable) != spec.adapter.expected_trainable_tensor_count
        ):
            raise RuntimeError("target DPO LoRA trainable parameter contract drifted")
        frozen_before = _parameter_fingerprint(
            (name, parameter)
            for name, parameter in peft_model.named_parameters()
            if not parameter.requires_grad
        )
        adapter_before = _adapter_summary(peft_model, torch)
        if (
            adapter_before["a_tensor_count"] != spec.adapter.expected_a_tensor_count
            or adapter_before["b_tensor_count"] != spec.adapter.expected_b_tensor_count
            or adapter_before["nonzero_a_tensor_count"]
            != spec.adapter.expected_a_tensor_count
            or adapter_before["nonzero_b_tensor_count"] != 0
            or adapter_before["nonzero_b_element_count"] != 0
        ):
            raise RuntimeError("target DPO initial LoRA tensor contract drifted")

        initial_policy, initial_reference, initial_adapter_disable_audit = (
            _policy_reference_logps(
            peft_model, batch, torch
            )
        )
        initial_reference_error = float(
            torch.max(torch.abs(initial_policy - initial_reference)).item()
        )
        if initial_reference_error != 0.0:
            raise RuntimeError("zero-initialized DPO adapter differs from reference")
        initial_margins = _relative_margins(initial_policy, initial_reference)
        if initial_margins != [0.0] * pair_count:
            raise RuntimeError("initial target DPO relative margins are not zero")
        peft_model.eval()
        with torch.no_grad():
            initial_trainer_loss = float(trainer_any.compute_loss(peft_model, batch))
        if not math.isclose(
            initial_trainer_loss, math.log(2.0), rel_tol=0.0, abs_tol=1e-6
        ):
            raise RuntimeError("initial target DPO loss differs from log(2)")
        frozen_state_before = _non_adapter_state_fingerprint(peft_model)
        model_config_before = _config_fingerprint(
            peft_model.config, "target DPO model config"
        )
        generation_config_before = _config_fingerprint(
            peft_model.generation_config, "target DPO generation config"
        )

        train_output = trainer_any.train()
        if train_output.global_step != spec.trainer.max_steps:
            raise RuntimeError("target DPO trainer global step drifted")
        if callback.calls != 1:
            raise RuntimeError("target DPO gradient callback count drifted")
        if (
            callback.trainable_gradient_tensors
            != spec.adapter.expected_trainable_tensor_count
            or callback.finite_gradient_tensors
            != callback.trainable_gradient_tensors
            or callback.frozen_gradient_tensors != 0
        ):
            raise RuntimeError("target DPO gradient contract failed")

        frozen_after = _parameter_fingerprint(
            (name, parameter)
            for name, parameter in peft_model.named_parameters()
            if not parameter.requires_grad
        )
        if not hmac.compare_digest(frozen_before, frozen_after):
            raise RuntimeError("target DPO frozen base changed during optimizer step")
        final_policy, final_reference, final_adapter_disable_audit = (
            _policy_reference_logps(peft_model, batch, torch)
        )
        reference_logps_error = float(
            torch.max(torch.abs(initial_reference - final_reference)).item()
        )
        if not math.isfinite(reference_logps_error):
            raise RuntimeError("target DPO adapter-disabled reference replay is non-finite")
        final_margins = _relative_margins(final_policy, final_reference)
        if any(not math.isfinite(value) or value <= 0 for value in final_margins):
            raise RuntimeError("target DPO chosen-relative margin did not improve per pair")
        peft_model.eval()
        with torch.no_grad():
            final_trainer_loss = float(trainer_any.compute_loss(peft_model, batch))
        if not math.isfinite(final_trainer_loss) or final_trainer_loss >= (
            initial_trainer_loss
        ):
            raise RuntimeError("target DPO one-batch post-step loss did not decrease")
        adapter_after = _adapter_summary(peft_model, torch)
        if (
            adapter_after["a_tensor_count"] != spec.adapter.expected_a_tensor_count
            or adapter_after["b_tensor_count"] != spec.adapter.expected_b_tensor_count
            or adapter_after["nonzero_a_tensor_count"]
            != spec.adapter.expected_a_tensor_count
            or adapter_after["nonzero_b_tensor_count"]
            != spec.adapter.expected_b_tensor_count
            or adapter_after["nonzero_b_element_count"]
            != spec.adapter.expected_b_element_count
        ):
            raise RuntimeError("target DPO trained LoRA tensor contract drifted")
        final_special_token_state = _special_token_state(peft_model, tokenizer)
        if final_special_token_state != normalized_special_token_state:
            raise RuntimeError("Trainer changed normalized target DPO special-token config")
        frozen_state_after = _non_adapter_state_fingerprint(peft_model)
        model_config_after = _config_fingerprint(
            peft_model.config, "target DPO model config"
        )
        generation_config_after = _config_fingerprint(
            peft_model.generation_config, "target DPO generation config"
        )
        if (
            not hmac.compare_digest(frozen_state_before, frozen_state_after)
            or not hmac.compare_digest(model_config_before, model_config_after)
            or not hmac.compare_digest(
                generation_config_before, generation_config_after
            )
        ):
            raise RuntimeError("target DPO frozen non-adapter state/config changed")

    result: dict[str, object] = {
        "tokenization": {
            "pair_count": pair_count,
            "pairs": rendered_pairs,
            "collated_input_shape": [int(value) for value in batch["input_ids"].shape],
            "collated_attention_token_count": int(
                batch["attention_mask"].sum().item()
            ),
            "prompt_masked_token_counts": prompt_masked_counts,
            "completion_token_counts": completion_counts,
            "chosen_rows_first": True,
            "rejected_rows_second": True,
            "silent_truncation_observed": False,
        },
        "model": {
            "class": type(model).__name__,
            "model_type": getattr(model.config, "model_type", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "base_parameter_count": base_parameter_count,
            "adapter_parameter_count": sum(
                parameter.numel() for _, parameter in trainable
            ),
            "total_parameter_count_with_adapter": sum(
                parameter.numel() for parameter in peft_model.parameters()
            ),
            "trainable_tensor_count": len(trainable),
            "parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in peft_model.parameters()}
            ),
            "special_token_contract": {
                "source_before_normalization": source_special_token_state,
                "normalized_before_baseline": normalized_special_token_state,
                "after_trainer_alignment_preflight": aligned_special_token_state,
                "after_trainer": final_special_token_state,
            },
        },
        "execution": {
            "initial_policy_logps": [float(value) for value in initial_policy.tolist()],
            "initial_reference_logps": [
                float(value) for value in initial_reference.tolist()
            ],
            "initial_policy_reference_max_abs_error": initial_reference_error,
            "initial_relative_margins": initial_margins,
            "initial_trainer_loss": initial_trainer_loss,
            "expected_initial_loss_log_2": math.log(2.0),
            "final_policy_logps": [float(value) for value in final_policy.tolist()],
            "final_reference_logps": [
                float(value) for value in final_reference.tolist()
            ],
            "final_relative_margins": final_margins,
            "final_trainer_loss": final_trainer_loss,
            "trainer_reported_pre_step_loss": float(train_output.training_loss),
            "global_step": int(train_output.global_step),
            "backward_executed": True,
            "optimizer_step_count": spec.trainer.max_steps,
            "gradient_callback_count": callback.calls,
            "trainable_gradient_tensor_count": callback.trainable_gradient_tensors,
            "finite_gradient_tensor_count": callback.finite_gradient_tensors,
            "frozen_base_gradient_tensor_count": callback.frozen_gradient_tensors,
            "frozen_base_parameter_fingerprint_before": frozen_before,
            "frozen_base_parameter_fingerprint_after": frozen_after,
            "frozen_base_parameters_unchanged": True,
            "frozen_non_adapter_state_fingerprint_before": frozen_state_before,
            "frozen_non_adapter_state_fingerprint_after": frozen_state_after,
            "frozen_non_adapter_state_unchanged": True,
            "normalized_model_config_fingerprint_before": model_config_before,
            "normalized_model_config_fingerprint_after": model_config_after,
            "normalized_model_config_unchanged": True,
            "normalized_generation_config_fingerprint_before": (
                generation_config_before
            ),
            "normalized_generation_config_fingerprint_after": generation_config_after,
            "normalized_generation_config_unchanged": True,
            "reference_logps_sha256_before": _tensor_sha256(initial_reference),
            "reference_logps_sha256_after": _tensor_sha256(final_reference),
            "reference_replay_max_abs_error_after_step": reference_logps_error,
            "reference_replay_bitwise_equal": reference_logps_error == 0.0,
            "reference_replay_drift_reported_not_equated_to_weight_drift": True,
            "initial_adapter_disable_audit": initial_adapter_disable_audit,
            "final_adapter_disable_audit": final_adapter_disable_audit,
            "adapter_before": adapter_before,
            "adapter_after": adapter_after,
        },
    }
    del trainer, trainer_any, peft_model, model, batch, train_output
    gc.collect()
    return result


def run_target_dpo_control(
    spec: TargetDPOControlSpec,
    *,
    checkpoint_spec: CheckpointControlSpec,
    checkpoint_report_path: Path,
    training_path: Path,
    readiness_path: Path,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Verify all inputs, load the reviewed Qwen snapshot, and run target DPO."""

    try:
        import torch
        import transformers
        import trl
        from packaging.version import Version
        from peft import __version__ as peft_version
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("target DPO optional dependencies are required") from error
    actual_runtime = (
        platform.python_implementation(),
        platform.python_version(),
        platform.platform(),
        torch.__version__,
        transformers.__version__,
        peft_version,
        trl.__version__,
    )
    expected_runtime = (
        spec.runtime.python_implementation,
        spec.runtime.python_version,
        spec.runtime.platform,
        spec.runtime.torch_version,
        spec.runtime.transformers_version,
        spec.runtime.peft_version,
        spec.trainer.trl_version,
    )
    if actual_runtime != expected_runtime:
        raise ValueError("installed runtime differs from target DPO recorded protocol")

    checkpoint_report = verify_recorded_checkpoint_report(
        checkpoint_report_path,
        expected_manifest_fingerprint=checkpoint_spec.manifest_fingerprint,
    )
    if checkpoint_report.get("report_fingerprint") != (
        spec.source_checkpoint_report_fingerprint
    ):
        raise ValueError("target DPO source checkpoint report fingerprint mismatch")
    training_file, readiness_file, records, audit, readiness = (
        _verified_preference_inputs(
            spec, training_path=training_path, readiness_path=readiness_path
        )
    )

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
        attn_implementation=spec.runtime.attention_implementation,
        **dtype_argument,
    )
    result = execute_loaded_target_dpo_control(
        spec, model=model, tokenizer=tokenizer, records=records
    )
    projection: dict[str, object] = {
        "report_version": TARGET_DPO_REPORT_VERSION,
        "control_manifest_fingerprint": spec.manifest_fingerprint,
        "checked_at": spec.checked_at,
        "source": {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "checkpoint_manifest_fingerprint": (
                spec.source_checkpoint_manifest_fingerprint
            ),
            "checkpoint_report_fingerprint": spec.source_checkpoint_report_fingerprint,
            "loader_input": "verified_local_snapshot_directory",
            "all_selected_checkpoint_file_bytes_verified_before_load": True,
            "selected_checkpoint_files": [dict(item) for item in snapshot.files],
            "selected_checkpoint_total_bytes": sum(
                cast(int, item["size_bytes"]) for item in snapshot.files
            ),
        },
        "data": {
            "training_artifact": training_file,
            "readiness_artifact": readiness_file,
            "record_ids": [record.record_id for record in records],
            "training_ordered_dataset_fingerprint": audit.ordered_dataset_fingerprint,
            "training_manifest_fingerprint": audit.manifest_fingerprint,
            "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
            "readiness_gate_passed": readiness.gate_passed,
            "held_out_plaintext_embedded": False,
            "authored_fixture_labels": True,
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft_version,
            "trl_version": trl.__version__,
            "device": spec.runtime.device,
            "dtype": spec.runtime.dtype,
            "attention_implementation": spec.runtime.attention_implementation,
            "torch_num_threads": torch.get_num_threads(),
            "cuda_executed": False,
        },
        "trainer_contract": {
            "loss_type": spec.trainer.loss_type,
            "beta": spec.trainer.beta,
            "max_length": spec.trainer.max_length,
            "batch_size": spec.trainer.batch_size,
            "gradient_accumulation_steps": (
                spec.trainer.gradient_accumulation_steps
            ),
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
        },
        "result": result,
        "scope": {
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
        },
        "evidence_boundary": TARGET_DPO_EVIDENCE_BOUNDARY,
    }
    projection["report_fingerprint"] = "sha256:" + artifact_fingerprint(projection)
    return projection


def verify_recorded_target_dpo_report(
    path: Path,
    *,
    spec: TargetDPOControlSpec,
    checkpoint_spec: CheckpointControlSpec,
    checkpoint_report_path: Path,
    training_path: Path,
    readiness_path: Path,
) -> Mapping[str, Any]:
    """Verify a recorded report through the isolated evidence verifier."""

    from about_llm.finetuning.target_dpo_verification import (
        verify_recorded_target_dpo_report as verify,
    )

    return verify(
        path,
        spec=spec,
        checkpoint_spec=checkpoint_spec,
        checkpoint_report_path=checkpoint_report_path,
        training_path=training_path,
        readiness_path=readiness_path,
    )
