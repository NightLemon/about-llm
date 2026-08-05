"""Consumer-GPU QLoRA SFT entry point with an offline estimate-only mode."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from about_llm.finetuning import estimate_qlora_memory, oom_degradation_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True, help="Commit hash, not a moving branch")
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
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
    if not args.estimate_only and (args.train_jsonl is None or args.output_dir is None):
        parser.error("training requires --train-jsonl and --output-dir")
    return args


def main() -> None:
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

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.revision, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise ValueError("checkpoint tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map={"": 0},
    )
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
    dataset = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    config = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_length,
        assistant_only_loss=True,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
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
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
