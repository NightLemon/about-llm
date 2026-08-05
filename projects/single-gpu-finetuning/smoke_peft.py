"""Offline PEFT smoke test using a randomly initialized tiny GPT-2."""

from __future__ import annotations

import json

import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from transformers import GPT2Config, GPT2LMHeadModel

from about_llm.integrations.transformers_tools import parameter_report


def run_smoke(steps: int = 10) -> dict[str, object]:
    torch.manual_seed(31)
    base = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=32,
            n_layer=2,
            n_head=4,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
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
    report = parameter_report(model)
    input_ids = torch.tensor([[1, 5, 7, 9, 2], [1, 4, 6, 8, 2]])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=input_ids)
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))

    model.eval()
    with torch.no_grad():
        adapter_logits = model(input_ids).logits
    adapter_keys = sorted(get_peft_model_state_dict(model, save_embedding_layers=False))
    merged = model.merge_and_unload().eval()
    with torch.no_grad():
        merged_logits = merged(input_ids).logits
    maximum_merge_error = float((adapter_logits - merged_logits).abs().max())

    return {
        "parameter_report": report,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "adapter_tensor_count": len(adapter_keys),
        "maximum_merge_error": maximum_merge_error,
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
