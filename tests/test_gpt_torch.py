from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gpt_torch = pytest.importorskip("about_llm.from_scratch.gpt_torch")
GPTConfig = gpt_torch.GPTConfig
MiniGPT = gpt_torch.MiniGPT


def tiny_model() -> MiniGPT:
    torch.manual_seed(11)
    return MiniGPT(
        GPTConfig(
            vocab_size=32,
            context_length=8,
            model_dim=16,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2,
        )
    ).eval()


def test_forward_shape_loss_and_weight_tying() -> None:
    model = tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])
    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 4, 32)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_causality_future_tokens_do_not_change_past_logits() -> None:
    model = tiny_model()
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 7, 8]])
    first_logits, _ = model(first)
    second_logits, _ = model(second)
    torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2])


def test_greedy_generation_is_deterministic_and_respects_length() -> None:
    model = tiny_model()
    prompt = torch.tensor([[1, 2]])
    first = model.generate(prompt, max_new_tokens=3, temperature=0)
    second = model.generate(prompt, max_new_tokens=3, temperature=0)
    assert first.shape == (1, 5)
    torch.testing.assert_close(first, second)
