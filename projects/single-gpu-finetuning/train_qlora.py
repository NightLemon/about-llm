"""面向消费级 GPU 的 QLoRA SFT 入口，并提供完全离线的显存估算模式。

实际训练路径执行 4-bit base 加载、k-bit training 准备、LoRA 注入、assistant-only labels 与
TRL SFTTrainer；estimate-only 路径只根据配置估算显存和 OOM 降级顺序，不下载模型。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from about_llm.finetuning import (
    audit_assistant_label_projection,
    estimate_qlora_memory,
    load_sft_records,
    load_sft_training_readiness,
    oom_degradation_order,
    prepare_assistant_mask_features,
    validate_sft_training_readiness,
)
from about_llm.finetuning.adapter_bundle import (
    bind_peft_adapter_identity,
    publish_sft_adapter_bundle,
)
from about_llm.finetuning.training_runtime import (
    cuda_memory_snapshot,
    normalize_trainer_metrics,
    reset_cuda_peak_memory,
    training_runtime_identity,
    write_strict_json,
)


def parse_args() -> argparse.Namespace:
    """定义估算、数据门禁、量化、LoRA 与训练参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True, help="Commit hash, not a moving branch")
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--readiness-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chat-template-path", type=Path)
    parser.add_argument("--data-preflight-only", action="store_true")
    parser.add_argument("--tokenization-preflight-only", action="store_true")
    parser.add_argument("--num-parameters", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--target-linears-per-layer", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    numeric = (
        args.num_parameters,
        args.num_layers,
        args.hidden_size,
        args.max_length,
        args.micro_batch_size,
        args.gradient_accumulation,
        args.rank,
        args.alpha,
        args.target_linears_per_layer,
        args.learning_rate,
        args.epochs,
    )
    if any(value <= 0 for value in numeric):
        parser.error("all numeric arguments must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-steps must be -1 or a positive integer")
    if args.data_preflight_only and args.tokenization_preflight_only:
        parser.error("choose at most one preflight-only mode")
    if not args.estimate_only and (
        args.train_jsonl is None
        or args.readiness_json is None
        or args.output_dir is None
    ):
        parser.error(
            "training requires --train-jsonl, --readiness-json, and --output-dir"
        )
    return args


def _training_run_report(
    *,
    args: argparse.Namespace,
    readiness_fingerprint: str,
    assistant_mask_fingerprint: str,
    final_labels_fingerprint: str,
    target_modules: list[str],
    compute_dtype: str,
    status: str,
    optimizer_step_count: int,
    trainable_parameter_count: int,
    total_parameter_count: int,
    trainer_metrics: dict[str, object],
    runtime: dict[str, object],
    memory_before: dict[str, object],
    memory_after: dict[str, object],
) -> dict[str, object]:
    """汇总 QLoRA 运行身份、显存、数据和训练指标。"""

    return {
        "report_version": "about-llm.qlora-training-run.v1",
        "status": status,
        "model": {"model_id": args.model_id, "revision": args.revision},
        "data": {
            "readiness_manifest_fingerprint": readiness_fingerprint,
            "assistant_mask_manifest_fingerprint": assistant_mask_fingerprint,
            "final_labels_fingerprint": final_labels_fingerprint,
        },
        "runtime": runtime,
        "training": {
            "quantized_base": "bitsandbytes-nf4-double-quant",
            "bnb_compute_dtype": compute_dtype,
            "target_modules": target_modules,
            "rank": args.rank,
            "alpha": args.alpha,
            "max_length": args.max_length,
            "per_device_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "gradient_checkpointing": True,
            "seed": 42,
        },
        "outcome": {
            "optimizer_step_count": optimizer_step_count,
            "trainable_parameter_count": trainable_parameter_count,
            "total_parameter_count": total_parameter_count,
            "trainer_metrics": trainer_metrics,
        },
        "torch_cuda_allocator": {
            "measurement_window": (
                "Immediately before Trainer.train() until it returned or raised "
                "torch.OutOfMemoryError; model and Trainer initialization happened earlier."
            ),
            "before_train": memory_before,
            "after_train": memory_after,
        },
        "evidence_boundary": (
            "This report records one local Trainer configuration, terminal status, "
            "Trainer metrics, and process-local torch CUDA allocator counters. It does "
            "not include driver-wide GPU use, authenticate model or data provenance, "
            "establish model quality, or predict another workload."
        ),
    }


def main() -> None:
    """选择离线估算，或执行数据→4bit 模型→LoRA→SFT→发布闭环。"""

    args = parse_args()
    estimate = estimate_qlora_memory(
        num_parameters=args.num_parameters,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        sequence_length=args.max_length,
        micro_batch_size=args.micro_batch_size,
        lora_rank=args.rank,
        target_linears_per_layer=args.target_linears_per_layer,
    )
    print(json.dumps(asdict(estimate), indent=2))
    if args.estimate_only:
        print("OOM degradation order:")
        for index, action in enumerate(oom_degradation_order(), start=1):
            print(f"{index}. {action}")
        return

    if (
        args.train_jsonl is None
        or args.readiness_json is None
        or args.output_dir is None
    ):
        raise RuntimeError("training arguments were not validated")
    # 先做数据/readiness 门禁，失败时不浪费显存加载模型。
    records = load_sft_records(args.train_jsonl)
    readiness = load_sft_training_readiness(args.readiness_json)
    audit = validate_sft_training_readiness(records, readiness)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_strict_json(args.output_dir / "sft-data-audit.json", audit.to_dict())
    write_strict_json(
        args.output_dir / "sft-training-readiness.json", readiness.to_dict()
    )
    if args.data_preflight_only:
        return

    from transformers import AutoTokenizer

    # 先用目标 tokenizer 精确投影 assistant labels，再决定是否进入 GPU 训练。
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.revision, trust_remote_code=False
    )
    if args.chat_template_path is not None:
        chat_template = args.chat_template_path.read_text(encoding="utf-8")
        if not chat_template.strip():
            raise ValueError("chat-template-path must not be empty")
        tokenizer.chat_template = chat_template
    if not tokenizer.chat_template:
        raise ValueError("checkpoint tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a PAD or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    mask_preparation = prepare_assistant_mask_features(
        records,
        render=lambda messages, tools: tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        ),
        renderer_identity={
            "model_id": args.model_id,
            "revision": args.revision,
            "transformers_version": version("transformers"),
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template": tokenizer.chat_template,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        max_length=args.max_length,
    )
    mask_audit = mask_preparation.audit_report
    write_strict_json(
        args.output_dir / "sft-template-mask-audit.json", mask_audit.to_dict()
    )
    if args.tokenization_preflight_only:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires CUDA; use --estimate-only on CPU")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    # BitsAndBytes 在加载时把 base linear weights 量化到 4 bit；LoRA 参数仍用可训练高精度。
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map={"": 0},
    )
    # 冻结并整理量化 base，启用 gradient checkpointing 后再注入 LoRA。
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = [value.strip() for value in args.target_modules.split(",") if value.strip()]
    if not targets:
        raise ValueError("target-modules cannot be empty")
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    dataset = Dataset.from_list(mask_preparation.to_training_rows())
    config = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_length,
        # Masks were materialized before Arrow. TRL 0.29.1 rejects its
        # conversational-preprocessing flag on an already-tokenized dataset,
        # while the collator still applies the supplied assistant_masks.
        assistant_only_loss=False,
        completion_only_loss=False,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        bf16=compute_dtype is torch.bfloat16,
        fp16=compute_dtype is torch.float16,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    prepared_dataset = trainer.train_dataset
    if prepared_dataset is None:
        raise RuntimeError("TRL did not preserve the pre-tokenized training dataset")
    label_audit = audit_assistant_label_projection(
        mask_preparation,
        prepared_features=(
            dict(prepared_dataset[index]) for index in range(len(prepared_dataset))
        ),
        collate=trainer.data_collator,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    write_strict_json(
        args.output_dir / "sft-final-label-audit.json", label_audit.to_dict()
    )

    total_parameter_count = sum(
        parameter.numel() for parameter in trainer.model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    if not 0 < trainable_parameter_count < total_parameter_count:
        raise RuntimeError("QLoRA training requires a non-empty strict trainable subset")

    device = trainer.args.device
    runtime = training_runtime_identity(
        torch, ("transformers", "trl", "peft", "accelerate", "bitsandbytes")
    )
    reset_cuda_peak_memory(torch, device)
    memory_before = cuda_memory_snapshot(torch, device)
    try:
        train_output = trainer.train()
    except torch.OutOfMemoryError:
        write_strict_json(
            args.output_dir / "sft-training-run.json",
            _training_run_report(
                args=args,
                readiness_fingerprint=readiness.manifest_fingerprint,
                assistant_mask_fingerprint=mask_audit.manifest_fingerprint,
                final_labels_fingerprint=label_audit.labels_fingerprint,
                target_modules=targets,
                compute_dtype=str(compute_dtype),
                status="cuda_out_of_memory",
                optimizer_step_count=int(trainer.state.global_step),
                trainable_parameter_count=trainable_parameter_count,
                total_parameter_count=total_parameter_count,
                trainer_metrics={},
                runtime=runtime,
                memory_before=memory_before,
                memory_after=cuda_memory_snapshot(torch, device),
            ),
        )
        raise
    adapter_directory = args.output_dir / "adapter"
    tokenizer_directory = args.output_dir / "tokenizer"
    for path in (adapter_directory, tokenizer_directory):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace completed training artifact: {path}")
    bind_peft_adapter_identity(
        trainer.model, model_id=args.model_id, revision=args.revision
    )
    trainer.model.save_pretrained(
        adapter_directory,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    tokenizer.save_pretrained(tokenizer_directory)
    write_strict_json(
        args.output_dir / "sft-training-run.json",
        _training_run_report(
            args=args,
            readiness_fingerprint=readiness.manifest_fingerprint,
            assistant_mask_fingerprint=mask_audit.manifest_fingerprint,
            final_labels_fingerprint=label_audit.labels_fingerprint,
            target_modules=targets,
            compute_dtype=str(compute_dtype),
            status="completed",
            optimizer_step_count=int(trainer.state.global_step),
            trainable_parameter_count=trainable_parameter_count,
            total_parameter_count=total_parameter_count,
            trainer_metrics=normalize_trainer_metrics(train_output.metrics),
            runtime=runtime,
            memory_before=memory_before,
            memory_after=cuda_memory_snapshot(torch, device),
        ),
    )
    # 发布前将 adapter、tokenizer、训练报告和身份清单绑定成完整 bundle。
    publish_sft_adapter_bundle(args.output_dir)


if __name__ == "__main__":
    main()
