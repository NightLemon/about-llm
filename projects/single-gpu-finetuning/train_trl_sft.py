"""Single-GPU LoRA SFT entry point for chat-formatted JSONL data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def validate_row(row: dict[str, Any], index: int) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {index} must contain a non-empty messages list")
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(
                f"row {index} message {message_index} must contain exactly role and content"
            )
        if message["role"] not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"row {index} message {message_index} has unsupported role")
        if not isinstance(message["content"], str) or not message["content"]:
            raise ValueError(f"row {index} message {message_index} content must be non-empty")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError(f"row {index} has no assistant message to supervise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    dataset = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    for index, row in enumerate(dataset):
        validate_row(row, index)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=False,
    )
    if not tokenizer.chat_template:
        raise ValueError("checkpoint tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a PAD or EOS token")
        tokenizer.pad_token = tokenizer.eos_token

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
        assistant_only_loss=True,
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
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
