"""离线跑通 PEFT LoRA 的训练、保存、重载、merge 与完整文件验证。

实验用随机微型 GPT-2，本地训练 LoRA 后分别保存 adapter、merged model 和 tokenizer；发布清单
绑定全部文件哈希与 base identity，只有先通过 verifier 才重新加载并比较 logits。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, cast

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from about_llm.finetuning.peft_export import (
    PEFTExportIdentity,
    write_peft_export_manifest_new,
)
from about_llm.integrations.transformers_tools import (
    parameter_report,
)
from about_llm.llmops import canonical_json_bytes

_BASE_MODEL_ID = "authored-random-gpt2"
_BASE_REVISION = "fixture-seed-31"
_TOKENIZER_REVISION = "authored-wordlevel-v1"


def _sha256_file(path: Path) -> str:
    """分块读取文件并计算 SHA-256，避免把大 artifact 全部载入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _maximum_error(left: torch.Tensor, right: torch.Tensor) -> float:
    """返回两个 logits 张量的最大绝对误差。"""

    return float((left - right).abs().max())


def _tokenizer() -> PreTrainedTokenizerFast:
    """构建与微型模型配套的离线 WordLevel tokenizer。"""

    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        **{f"tok{index}": index for index in range(4, 32)},
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=16,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['content'] }} {{ eos_token }}"
        "{% endfor %}"
    )
    return tokenizer


def _load_base(path: Path) -> GPT2LMHeadModel:
    """只从本地 safetensors 目录加载一份干净 base model。"""

    model = cast(
        GPT2LMHeadModel,
        GPT2LMHeadModel.from_pretrained(path, local_files_only=True),
    )
    model.loss_type = "ForCausalLM"
    return model


def _run_smoke(*, steps: int, root: Path, persisted: bool) -> dict[str, Any]:
    """在给定目录完成 LoRA 训练、三类 artifact 发布与重载比较。"""

    torch.manual_seed(31)
    base = GPT2LMHeadModel(  # type: ignore[no-untyped-call]
        GPT2Config(  # type: ignore[no-untyped-call]
            vocab_size=32,
            n_positions=16,
            n_embd=32,
            n_layer=2,
            n_head=4,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            loss_type="ForCausalLM",
        )
    )
    base.loss_type = "ForCausalLM"
    base_dir = root / "base"
    adapter_dir = root / "adapter"
    merged_dir = root / "merged"
    tokenizer_dir = root / "tokenizer"
    # 先保存随机 base，后续每条验证路径都从相同 bytes 重新加载，避免共享内存参数。
    base.save_pretrained(base_dir, safe_serialization=True)
    base_weight_path = base_dir / "model.safetensors"

    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            lora_dropout=0,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
        ),
    )
    model.peft_config["default"].base_model_name_or_path = str(base_dir.resolve())
    parameters = parameter_report(model)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    input_ids = torch.tensor([[1, 5, 7, 9, 2], [1, 4, 6, 8, 2]])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    losses: list[float] = []
    # 只有 LoRA 参数可训练；base 参数保持冻结。
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=input_ids)
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))

    base_parameters_unchanged = all(
        torch.equal(parameter, frozen_before[name])
        for name, parameter in model.named_parameters()
        if name in frozen_before
    )
    model.eval()
    with torch.no_grad():
        trained_adapter_logits = model(input_ids).logits
    adapter_keys = sorted(get_peft_model_state_dict(model, save_embedding_layers=False))
    # adapter 保存低秩增量；merged 模型则把增量合回 base 权重。
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    adapter_weight_path = adapter_dir / "adapter_model.safetensors"
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_config: dict[str, Any] = json.loads(
        adapter_config_path.read_text(encoding="utf-8")
    )
    adapter_config["base_model_name_or_path"] = _BASE_MODEL_ID
    adapter_config_path.write_bytes(canonical_json_bytes(adapter_config))

    builder_base = _load_base(base_dir)
    builder_adapter = PeftModel.from_pretrained(
        builder_base,
        adapter_dir,
        is_trainable=False,
        local_files_only=True,
    ).eval()
    with torch.no_grad():
        builder_adapter_logits = builder_adapter(input_ids).logits
    merged = builder_adapter.merge_and_unload(safe_merge=True).eval()
    with torch.no_grad():
        merged_logits = merged(input_ids).logits
    merge_error = _maximum_error(builder_adapter_logits, merged_logits)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    merged_weight_path = merged_dir / "model.safetensors"

    tokenizer = _tokenizer()
    tokenizer.save_pretrained(tokenizer_dir)
    messages = [
        {"role": "user", "content": "tok5 tok7"},
        {"role": "assistant", "content": "tok9"},
    ]
    expected_chat_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    identity = PEFTExportIdentity(
        artifact_id="authored-peft-export-control",
        architecture="GPT2LMHeadModel",
        base_model_id=_BASE_MODEL_ID,
        base_revision=_BASE_REVISION,
        tokenizer_revision=_TOKENIZER_REVISION,
    )
    # 清单最后发布并覆盖全部预期文件，防止“目录存在但文件不完整”被当作成功。
    verification = write_peft_export_manifest_new(
        root,
        identity=identity,
        target_modules=("c_attn",),
    )

    verified_base = _load_base(base_dir)
    # verifier 通过后才用公开加载 API 重载 adapter 与 merged model 并比较输出。
    verified_adapter = PeftModel.from_pretrained(
        verified_base,
        adapter_dir,
        is_trainable=False,
        local_files_only=True,
    ).eval()
    with torch.no_grad():
        verified_adapter_logits = verified_adapter(input_ids).logits
    adapter_reload_error = _maximum_error(
        trained_adapter_logits, verified_adapter_logits
    )
    reloaded_merged = _load_base(merged_dir)
    reloaded_merged.eval()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        reloaded_merged_logits = reloaded_merged(input_ids).logits
    merged_reload_error = _maximum_error(merged_logits, reloaded_merged_logits)
    reloaded_tokenizer: Any = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        tokenizer_dir, local_files_only=True
    )
    reloaded_chat_ids = reloaded_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    weights = {
        "base": {
            "relative_path": "base/model.safetensors",
            "bytes": base_weight_path.stat().st_size,
            "sha256": _sha256_file(base_weight_path),
        },
        "adapter": {
            "relative_path": "adapter/adapter_model.safetensors",
            "bytes": adapter_weight_path.stat().st_size,
            "sha256": _sha256_file(adapter_weight_path),
        },
        "merged": {
            "relative_path": "merged/model.safetensors",
            "bytes": merged_weight_path.stat().st_size,
            "sha256": _sha256_file(merged_weight_path),
        },
    }
    pickle_weight_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.suffix in {".bin", ".pt", ".pth"}
    )
    return {
        "schema_version": 2,
        "fixture": {
            "seed": 31,
            "steps": steps,
            "input_ids": input_ids.tolist(),
        },
        "parameter_report": parameters,
        "training": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "base_parameters_unchanged": base_parameters_unchanged,
            "adapter_tensor_count": len(adapter_keys),
            "adapter_tensor_keys": adapter_keys,
        },
        "round_trip": {
            "builder_adapter_matches_trained_maximum_logit_error": _maximum_error(
                trained_adapter_logits, builder_adapter_logits
            ),
            "verified_adapter_reload_maximum_logit_error": adapter_reload_error,
            "merge_maximum_logit_error": merge_error,
            "verified_merged_reload_maximum_logit_error": merged_reload_error,
            "tokenizer_chat_template_token_ids": list(expected_chat_ids),
            "verified_tokenizer_chat_template_token_ids": list(reloaded_chat_ids),
            "tokenizer_chat_template_exact": reloaded_chat_ids == expected_chat_ids,
        },
        "artifacts": {
            "persisted": persisted,
            "root": str(root) if persisted else None,
            "weights": weights,
            "strict_verification": verification.to_dict(),
            "adapter_config_base_reference_kind": "immutable-id-string",
            "adapter_config_base_model_id": adapter_config[
                "base_model_name_or_path"
            ],
            "pickle_weight_files": pickle_weight_files,
        },
        "scope": {
            "network_used": False,
            "cpu_random_tiny_gpt2_peft_training_executed": True,
            "frozen_base_unchanged": base_parameters_unchanged,
            "base_full_safetensors_saved": True,
            "adapter_safetensors_saved_and_reloaded": True,
            "merge_and_unload_executed": True,
            "merged_full_safetensors_saved_and_reloaded": True,
            "strict_manifest_enforced_before_published_artifact_reload": True,
            "complete_directory_file_set_size_and_hash_bound": True,
            "safetensors_payloads_parsed_before_reload": True,
            "base_merged_config_payload_and_tensor_signature_match": True,
            "lora_target_a_b_tensor_coverage_validated": True,
            "tokenizer_and_chat_template_included_and_reloaded": True,
            "peft_loader_itself_enforces_repo_manifest": False,
            "adapter_config_path_or_id_authenticates_base_content": False,
            "optimizer_scheduler_rng_or_training_resume_state_included": False,
            "quantized_base_or_qlora_merge_executed": False,
            "target_checkpoint_or_cuda_executed": False,
            "task_quality_or_production_compatibility_proved": False,
            "cryptographic_origin_authenticated": False,
            "atomic_or_power_loss_safe_publication_proved": False,
            "concurrent_mutation_or_verify_load_toctou_prevented": False,
        },
    }


def run_smoke(
    steps: int = 10, *, artifact_root: Path | None = None
) -> dict[str, Any]:
    """在临时目录或调用者指定目录运行 PEFT 闭环。"""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if artifact_root is None:
        with tempfile.TemporaryDirectory(prefix="about-llm-peft-export-") as directory:
            return _run_smoke(steps=steps, root=Path(directory), persisted=False)
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=False)
    return _run_smoke(steps=steps, root=root, persisted=True)


def parse_args() -> argparse.Namespace:
    """定义训练步数和可选持久化 artifact 目录。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="optional new output directory; an existing target is rejected",
    )
    return parser.parse_args()


def main() -> None:
    """运行 PEFT smoke 并打印训练、文件与数值一致性证据。"""

    args = parse_args()
    print(
        json.dumps(
            run_smoke(args.steps, artifact_root=args.artifact_root),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
