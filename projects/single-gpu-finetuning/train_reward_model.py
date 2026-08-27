"""训练 LoRA/QLoRA Transformer reward model，并保证 held-out preference 不进优化器。

脚本先校验数据/readiness 和 chosen/rejected tokenization，再加载 sequence-classification base，
注入 LoRA 后由 RewardTrainer 优化 pairwise margin。训练报告保留数据与运行身份。
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from about_llm.finetuning import (
    audit_preference_tokenization,
    load_preference_records,
    load_preference_training_readiness,
    validate_preference_training_readiness,
)


def parse_args() -> argparse.Namespace:
    """定义数据、模型、量化、LoRA、长度和训练输出参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--readiness-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chat-template-path", type=Path)
    parser.add_argument("--data-preflight-only", action="store_true")
    parser.add_argument("--tokenization-preflight-only", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--modules-to-save", default="score")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--center-rewards-coefficient", type=float, default=0.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    positive = (
        args.rank,
        args.alpha,
        args.max_length,
        args.batch_size,
        args.gradient_accumulation,
        args.learning_rate,
        args.epochs,
    )
    if any(value <= 0 for value in positive):
        parser.error("all numeric training arguments except centering must be positive")
    if args.center_rewards_coefficient < 0:
        parser.error("center-rewards-coefficient must be non-negative")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-steps must be -1 or a positive integer")
    if args.data_preflight_only and args.tokenization_preflight_only:
        parser.error("choose at most one preflight-only mode")
    if args.bf16 and args.fp16:
        parser.error("bf16 and fp16 are mutually exclusive")
    return args


def _write_json(path: Path, payload: object) -> None:
    """写入保留中文且禁止 NaN 的 JSON 报告。"""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _comma_separated(value: str, label: str) -> list[str]:
    """解析不能为空的逗号分隔模块名列表。"""

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items or len(items) != len(set(items)):
        raise ValueError(f"{label} must contain unique non-empty names")
    return items


def main() -> None:
    """执行数据门禁、token 审计、reward model 训练与 artifact 保存。"""

    args = parse_args()
    # 先验证训练数据和 readiness，避免加载大模型后才发现 split 或治理问题。
    records = load_preference_records(args.train_jsonl)
    readiness = load_preference_training_readiness(args.readiness_json)
    audit = validate_preference_training_readiness(records, readiness)
    targets = _comma_separated(args.target_modules, "target-modules")
    modules_to_save = _comma_separated(args.modules_to_save, "modules-to-save")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "preference-train-audit.json", audit.to_dict())
    _write_json(
        args.output_dir / "preference-training-readiness.json", readiness.to_dict()
    )
    _write_json(
        args.output_dir / "reward-model-data-contract.json",
        {
            "schema_version": 1,
            "model_id": args.model_id,
            "revision": args.revision,
            "train_manifest_fingerprint": audit.manifest_fingerprint,
            "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
            "binary_train_pair_count": len(records),
            "held_out_dataset_passed_to_trainer": False,
            "target_modules": targets,
            "modules_to_save": modules_to_save,
            "scalar_reward_count_per_sequence": 1,
        },
    )
    if args.data_preflight_only:
        return

    from transformers import AutoTokenizer

    # 用目标 tokenizer/template 预先确认 chosen/rejected 不会被截断成无效训练对。
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=False,
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

    def render(row: dict[str, Any]) -> dict[str, Any]:
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]
        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        prompt_chosen = tokenizer.apply_chat_template(
            prompt + chosen,
            tokenize=True,
            return_dict=True,
        )
        prompt_rejected = tokenizer.apply_chat_template(
            prompt + rejected,
            tokenize=True,
            return_dict=True,
        )
        return {
            "prompt_ids": prompt_ids,
            "prompt_chosen_ids": prompt_chosen["input_ids"],
            "prompt_rejected_ids": prompt_rejected["input_ids"],
        }

    tokenization_audit = audit_preference_tokenization(
        records,
        render=render,
        renderer_identity={
            "model_id": args.model_id,
            "revision": args.revision,
            "transformers_version": version("transformers"),
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template": tokenizer.chat_template,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "algorithm": "full-prompt-plus-response-sequence-classification",
        },
        max_length=args.max_length,
    )
    _write_json(
        args.output_dir / "preference-tokenization-audit.json",
        tokenization_audit.to_dict(),
    )
    if args.tokenization_preflight_only:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForSequenceClassification,
        BitsAndBytesConfig,
    )
    from trl import RewardConfig, RewardTrainer

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=targets,
        modules_to_save=modules_to_save,
        task_type="SEQ_CLS",
    )
    model: Any = args.model_id
    model_init_kwargs: dict[str, Any] | None = {
        "revision": args.revision,
        "trust_remote_code": False,
        "use_cache": False,
    }
    use_bf16 = args.bf16
    use_fp16 = args.fp16
    if args.qlora:
        from peft import prepare_model_for_kbit_training

        if not torch.cuda.is_available():
            raise RuntimeError("reward-model QLoRA requires CUDA; omit --qlora for LoRA")
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_id,
            revision=args.revision,
            trust_remote_code=False,
            num_labels=1,
            quantization_config=quantization,
            device_map={"": 0},
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
        model_init_kwargs = None
        use_bf16 = compute_dtype is torch.bfloat16
        use_fp16 = compute_dtype is torch.float16

    center_coefficient = (
        args.center_rewards_coefficient
        if args.center_rewards_coefficient > 0
        else None
    )
    config = RewardConfig(
        output_dir=str(args.output_dir),
        model_init_kwargs=model_init_kwargs,
        max_length=args.max_length,
        center_rewards_coefficient=center_coefficient,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=42,
        data_seed=42,
    )
    _write_json(
        args.output_dir / "reward-model-run-contract.json",
        {
            "schema_version": 1,
            "model_id": args.model_id,
            "revision": args.revision,
            "trl_version": version("trl"),
            "transformers_version": version("transformers"),
            "qlora": args.qlora,
            "target_modules": targets,
            "modules_to_save": modules_to_save,
            "rank": args.rank,
            "alpha": args.alpha,
            "max_length": args.max_length,
            "max_steps": args.max_steps,
            "center_rewards_coefficient": center_coefficient,
            "gradient_checkpointing": args.gradient_checkpointing,
            "training_readiness_manifest_fingerprint": (
                readiness.manifest_fingerprint
            ),
            "tokenization_manifest_fingerprint": (
                tokenization_audit.manifest_fingerprint
            ),
            "pairwise_loss": "-mean(logsigmoid(chosen_reward-rejected_reward))",
            "held_out_dataset_passed_to_trainer": False,
            "target_model_or_cuda_verified_by_repository": False,
        },
    )
    # 只将训练 split 转给 RewardTrainer；held-out 评测应在独立命令中运行。
    dataset = Dataset.from_list([record.to_dpo_row() for record in records])
    trainer = RewardTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    prepared = trainer.train_dataset
    if prepared is None or len(prepared) != len(records):
        raise AssertionError(
            "RewardTrainer filtered a pair after strict tokenization preflight"
        )
    total_parameter_count = sum(parameter.numel() for parameter in trainer.model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameter_count <= 0 or trainable_parameter_count >= total_parameter_count:
        raise AssertionError("LoRA reward model must have a strict trainable subset")
    # Trainer 对每一对计算 chosen score 与 rejected score 的 pairwise loss。
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    _write_json(
        args.output_dir / "reward-model-train-result.json",
        {
            "schema_version": 1,
            "global_step": trainer.state.global_step,
            "prepared_pair_count": len(prepared),
            "trainable_parameter_count": trainable_parameter_count,
            "total_parameter_count": total_parameter_count,
            "metrics": dict(train_result.metrics),
            "scope": {
                "trl_reward_trainer_executed": True,
                "held_out_dataset_passed_to_trainer": False,
                "target_model_quality_proved": False,
                "cuda_or_qlora_executed": args.qlora,
            },
        },
    )
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
