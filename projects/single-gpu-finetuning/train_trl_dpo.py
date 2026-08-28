"""单 GPU TRL DPO 入口，先通过 preference 数据门禁再更新 LoRA/QLoRA。

脚本验证 chosen/rejected 结构、近重复与 readiness，使用固定 chat template 做 tokenization 审计，
随后加载 policy/reference，计算 DPO log-ratio loss 并训练；held-out 数据不会进入优化器。
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path
from typing import Any

from about_llm.finetuning import (
    audit_preference_tokenization,
    load_preference_records,
    load_preference_training_readiness,
    validate_preference_training_readiness,
)
from about_llm.finetuning.training_runtime import normalize_trainer_metrics


def parse_args() -> argparse.Namespace:
    """定义模型、preference 数据、LoRA/QLoRA、DPO beta 与训练参数。"""

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
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    positive = (
        args.rank,
        args.alpha,
        args.beta,
        args.max_length,
        args.batch_size,
        args.gradient_accumulation,
        args.learning_rate,
        args.epochs,
    )
    if any(value <= 0 for value in positive):
        parser.error("all numeric training arguments must be positive")
    if args.data_preflight_only and args.tokenization_preflight_only:
        parser.error("choose at most one preflight-only mode")
    return args


def _write_json(path: Path, payload: object) -> None:
    """以 UTF-8 严格 JSON 写入 preflight 或训练报告。"""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _print_report(payload: object) -> None:
    """把完成状态作为单个严格 JSON 对象写到 stdout。"""

    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    """执行 preference 门禁、tokenization 审计和 DPOTrainer 训练。"""

    args = parse_args()
    # 数据/readiness 检查先于模型加载，确保训练集没有混入 held-out 或阻断级问题。
    records = load_preference_records(args.train_jsonl)
    readiness = load_preference_training_readiness(args.readiness_json)
    audit = validate_preference_training_readiness(records, readiness)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "preference-train-audit.json", audit.to_dict())
    _write_json(
        args.output_dir / "preference-training-readiness.json", readiness.to_dict()
    )
    if args.data_preflight_only:
        _print_report(
            {
                "report_version": "about-llm.dpo-preflight.v1",
                "mode": "data_preflight",
                "status": "completed",
                "model": {"model_id": args.model_id, "revision": args.revision},
                "pair_count": len(records),
                "model_loaded": False,
                "data": {
                    "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
                    "audit_manifest_fingerprint": audit.manifest_fingerprint,
                },
                "artifacts": [
                    str(args.output_dir / "preference-train-audit.json"),
                    str(args.output_dir / "preference-training-readiness.json"),
                ],
                "evidence_boundary": (
                    "Preference structure, split, and readiness were checked; tokenizer, "
                    "policy, reference computation, and optimization were not executed."
                ),
            }
        )
        return

    from transformers import AutoTokenizer

    # chosen/rejected 必须由同一个固定 tokenizer/template 渲染，才能比较条件 log-prob。
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
            "trl_version": version("trl"),
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template": tokenizer.chat_template,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "algorithm": (
                "trl-0.29-conversational-prompt-plus-completion-prefix-slicing"
            ),
        },
        max_length=args.max_length,
    )
    _write_json(
        args.output_dir / "preference-tokenization-audit.json",
        tokenization_audit.to_dict(),
    )
    if args.tokenization_preflight_only:
        _print_report(
            {
                "report_version": "about-llm.dpo-preflight.v1",
                "mode": "tokenization_preflight",
                "status": "completed",
                "model": {"model_id": args.model_id, "revision": args.revision},
                "pair_count": len(records),
                "model_loaded": False,
                "tokenizer_loaded": True,
                "data": {
                    "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
                    "tokenization_manifest_fingerprint": (
                        tokenization_audit.manifest_fingerprint
                    ),
                },
                "artifacts": [
                    str(args.output_dir / "preference-tokenization-audit.json"),
                ],
                "evidence_boundary": (
                    "Preference data and DPO prompt/chosen/rejected tokenization were "
                    "checked; policy and reference weights, optimization, and held-out "
                    "evaluation were not executed."
                ),
            }
        )
        return

    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    targets = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if not targets:
        raise ValueError("target-modules cannot be empty")
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    # Dataset 只包含已通过门禁的 train records；reference 不会看到评测 split。
    dataset = Dataset.from_list([record.to_dpo_row() for record in records])
    model: Any = args.model_id
    model_init_kwargs: dict[str, Any] | None = {
        "revision": args.revision,
        "trust_remote_code": False,
        "use_cache": False,
    }
    use_bf16 = args.bf16
    use_fp16 = False
    if args.qlora:
        import torch
        from peft import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("DPO QLoRA requires CUDA; omit --qlora for LoRA")
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.revision,
            trust_remote_code=False,
            quantization_config=quantization,
            device_map={"": 0},
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        model_init_kwargs = None
        use_bf16 = compute_dtype is torch.bfloat16
        use_fp16 = compute_dtype is torch.float16

    config = DPOConfig(
        output_dir=str(args.output_dir),
        model_init_kwargs=model_init_kwargs,
        max_length=args.max_length,
        truncation_mode="keep_start",
        beta=args.beta,
        loss_type="sigmoid",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        seed=42,
        data_seed=42,
    )
    _write_json(
        args.output_dir / "dpo-run-contract.json",
        {
            "model_id": args.model_id,
            "revision": args.revision,
            "qlora": args.qlora,
            "target_modules": targets,
            "rank": args.rank,
            "alpha": args.alpha,
            "beta": args.beta,
            "max_length": args.max_length,
            "loss_type": "sigmoid",
            "training_readiness_manifest_fingerprint": (
                readiness.manifest_fingerprint
            ),
            "tokenization_manifest_fingerprint": (
                tokenization_audit.manifest_fingerprint
            ),
            "reference_mode": "disable_current_peft_adapter",
            "held_out_dataset_passed_to_trainer": False,
        },
    )
    # DPOTrainer 内部同时计算 policy 与 reference 的 chosen/rejected log probabilities。
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    with redirect_stdout(sys.stderr):
        train_output = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint
        )
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    run_report = {
        "report_version": "about-llm.dpo-training-run.v1",
        "status": "completed",
        "model": {"model_id": args.model_id, "revision": args.revision},
        "data": {
            "pair_count": len(records),
            "readiness_manifest_fingerprint": readiness.manifest_fingerprint,
            "tokenization_manifest_fingerprint": (
                tokenization_audit.manifest_fingerprint
            ),
            "held_out_dataset_passed_to_trainer": False,
        },
        "training": {
            "qlora": args.qlora,
            "target_modules": targets,
            "rank": args.rank,
            "alpha": args.alpha,
            "beta": args.beta,
            "max_length": args.max_length,
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "mixed_precision_mode": "bf16" if use_bf16 else "fp16" if use_fp16 else "disabled",
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "reference_mode": "disable_current_peft_adapter",
        },
        "outcome": {
            "optimizer_step_count": int(trainer.state.global_step),
            "trainer_metrics": normalize_trainer_metrics(train_output.metrics),
        },
        "artifacts": {
            "output_directory": str(args.output_dir),
            "run_contract": str(args.output_dir / "dpo-run-contract.json"),
            "model_and_tokenizer_saved": True,
        },
        "evidence_boundary": (
            "This report records one local DPO training run and its authored training "
            "pairs. It does not evaluate held-out preference quality, human-label validity, "
            "safety alignment, or production convergence."
        ),
    }
    _write_json(args.output_dir / "dpo-training-run.json", run_report)
    _print_report(run_report)


if __name__ == "__main__":
    main()
