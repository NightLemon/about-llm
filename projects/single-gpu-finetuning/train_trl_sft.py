"""Single-GPU LoRA SFT entry point for chat-formatted JSONL data."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from about_llm.finetuning import (
    audit_assistant_label_projection,
    load_sft_records,
    load_sft_training_readiness,
    prepare_assistant_mask_features,
    validate_sft_training_readiness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--readiness-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chat-template-path", type=Path)
    parser.add_argument("--data-preflight-only", action="store_true")
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--bf16", action="store_true")
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
        parser.error("all numeric training arguments must be positive")
    return args


def main() -> None:
    args = parse_args()
    records = load_sft_records(args.train_jsonl)
    readiness = load_sft_training_readiness(args.readiness_json)
    audit = validate_sft_training_readiness(records, readiness)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sft-data-audit.json").write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output_dir / "sft-training-readiness.json").write_text(
        json.dumps(readiness.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if args.data_preflight_only:
        return

    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

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
    (args.output_dir / "sft-template-mask-audit.json").write_text(
        json.dumps(mask_audit.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    dataset = Dataset.from_list(mask_preparation.to_training_rows())

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if not target_modules:
        raise ValueError("target-modules cannot be empty")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    training_config = SFTConfig(
        output_dir=str(args.output_dir),
        model_init_kwargs={
            "revision": args.revision,
            "trust_remote_code": False,
        },
        max_length=args.max_length,
        # Masks were materialized before Arrow. TRL 0.29.1 rejects its
        # conversational-preprocessing flag on an already-tokenized dataset,
        # while the collator still applies the supplied assistant_masks.
        assistant_only_loss=False,
        completion_only_loss=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        bf16=args.bf16,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=args.model_id,
        args=training_config,
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
    (args.output_dir / "sft-final-label-audit.json").write_text(
        json.dumps(label_audit.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
