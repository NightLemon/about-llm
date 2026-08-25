"""Publish, verify, or fresh-reload one completed SFT PEFT adapter bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from about_llm.finetuning.adapter_bundle import (
    publish_sft_adapter_bundle,
    verify_sft_adapter_bundle,
)
from about_llm.finetuning.training_runtime import (
    training_runtime_identity,
    write_strict_json,
)
from about_llm.llmops import artifact_fingerprint

RELOAD_REPORT_VERSION = "about-llm.sft-adapter-reload.v1"
PROBE_ID = "about-llm-safe-kv-cache-definition-v1"
PROBE_TEXT = "请用一句话解释 KV Cache。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--training-output", type=Path, required=True)
    publish.add_argument("--bundle-directory", type=Path)

    for name in ("verify", "reload"):
        command = subparsers.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--expected-model-id", required=True)
        command.add_argument("--expected-revision", required=True)
        if name == "reload":
            command.add_argument("--report", type=Path, required=True)
            command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
            command.add_argument(
                "--dtype",
                choices=("float32", "float16", "bfloat16"),
                default="float16",
            )
            command.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.command == "reload" and args.device == "cpu" and args.dtype != "float32":
        parser.error("CPU reload uses --dtype float32")
    return args


def _tensor_fingerprint(tensor: Any) -> str:
    snapshot = tensor.detach().to(device="cpu").float().contiguous()
    return "sha256:" + hashlib.sha256(snapshot.numpy().tobytes()).hexdigest()


def _reload(args: argparse.Namespace) -> dict[str, object]:
    bundle = args.bundle.resolve()
    report = args.report.resolve()
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"refusing to replace reload report: {report}")
    if report.is_relative_to(bundle):
        raise ValueError("reload report must be written outside the verified bundle")
    verification = verify_sft_adapter_bundle(
        bundle,
        expected_model_id=args.expected_model_id,
        expected_revision=args.expected_revision,
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires an available CUDA device")
    if (
        args.device == "cuda"
        and args.dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("--dtype bfloat16 requires CUDA BF16 support")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        bundle / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROBE_TEXT}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        name: value.to(args.device)
        for name, value in rendered.items()
        if hasattr(value, "to")
    }
    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("fixed reload probe did not produce one token sequence")

    base = AutoModelForCausalLM.from_pretrained(
        args.expected_model_id,
        revision=args.expected_revision,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    base.eval()
    with torch.no_grad():
        base_logits = base(**inputs, use_cache=False).logits[:, -1, :]
    if not bool(torch.isfinite(base_logits).all().item()):
        raise RuntimeError("base probe logits contain non-finite values")

    adapter = PeftModel.from_pretrained(
        base,
        bundle / "adapter",
        is_trainable=False,
        local_files_only=True,
    ).eval()
    with torch.no_grad():
        adapter_logits = adapter(**inputs, use_cache=False).logits[:, -1, :]
    if not bool(torch.isfinite(adapter_logits).all().item()):
        raise RuntimeError("adapter probe logits contain non-finite values")
    maximum_delta = float(torch.max(torch.abs(adapter_logits - base_logits)).item())
    if maximum_delta <= 0:
        raise RuntimeError("freshly loaded adapter did not change the fixed probe logits")

    result: dict[str, object] = {
        "report_version": RELOAD_REPORT_VERSION,
        "status": "completed",
        "bundle": verification.to_dict(),
        "runtime": training_runtime_identity(
            torch, ("transformers", "peft", "accelerate", "safetensors")
        ),
        "execution": {
            "device": args.device,
            "dtype": args.dtype,
            "probe_id": PROBE_ID,
            "probe_text_persisted": False,
            "input_token_count": int(input_ids.shape[1]),
            "input_ids_fingerprint": "sha256:"
            + artifact_fingerprint(
                {"input_ids": [int(value) for value in input_ids[0].tolist()]}
            ),
            "base_last_logits_fingerprint": _tensor_fingerprint(base_logits),
            "adapter_last_logits_fingerprint": _tensor_fingerprint(adapter_logits),
            "maximum_last_logit_delta": maximum_delta,
            "adapter_loaded_with_peft": True,
            "full_generation_executed": False,
        },
        "evidence_boundary": (
            "This report verifies the complete adapter bundle, loads the expected base "
            "revision in a fresh process, and observes a finite non-zero last-logit change "
            "for one fixed non-sensitive probe. It does not establish response quality, "
            "generalization, safe behavior, merged-weight equivalence, or serving support."
        ),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    write_strict_json(report, result)
    return result


def main() -> None:
    args = parse_args()
    if args.command == "publish":
        result = publish_sft_adapter_bundle(
            args.training_output, bundle_directory=args.bundle_directory
        ).to_dict()
    elif args.command == "verify":
        result = verify_sft_adapter_bundle(
            args.bundle,
            expected_model_id=args.expected_model_id,
            expected_revision=args.expected_revision,
        ).to_dict()
    else:
        result = _reload(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
