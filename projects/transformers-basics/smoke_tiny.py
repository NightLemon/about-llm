"""Offline Transformers smoke test: build, train, and generate a tiny GPT-2."""

from __future__ import annotations

import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from about_llm.integrations.transformers_tools import parameter_report


def run_smoke(steps: int = 12) -> dict[str, object]:
    torch.manual_seed(23)
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.train()
    input_ids = torch.tensor([[1, 5, 7, 9, 2], [1, 4, 6, 8, 2]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(input_ids=input_ids, labels=input_ids)
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))

    model.eval()
    generated = model.generate(
        input_ids=input_ids[:, :2],
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=config.pad_token_id,
    )
    return {
        "parameter_report": parameter_report(model),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "generated_shape": list(generated.shape),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
