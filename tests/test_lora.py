from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
lora_module = pytest.importorskip("about_llm.finetuning.lora")
LoRALinear = lora_module.LoRALinear


def test_zero_initialized_adapter_preserves_base_and_freezes_it() -> None:
    torch.manual_seed(1)
    base = nn.Linear(5, 3)
    x = torch.randn(4, 5)
    expected = base(x).detach()
    layer = LoRALinear(base, rank=2, alpha=4)

    torch.testing.assert_close(layer(x), expected)
    assert not layer.base.weight.requires_grad
    assert layer.lora_a.requires_grad and layer.lora_b.requires_grad


def test_optimizer_updates_adapter_but_not_base() -> None:
    torch.manual_seed(2)
    layer = LoRALinear(nn.Linear(4, 2), rank=2)
    original_base = layer.base.weight.detach().clone()
    optimizer = torch.optim.SGD(
        [parameter for parameter in layer.parameters() if parameter.requires_grad], lr=0.1
    )
    x, target = torch.randn(8, 4), torch.randn(8, 2)

    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(layer(x), target)
        loss.backward()
        optimizer.step()

    torch.testing.assert_close(layer.base.weight, original_base)
    assert torch.count_nonzero(layer.lora_b).item() > 0


def test_merged_linear_matches_eval_output_and_is_independent() -> None:
    torch.manual_seed(3)
    layer = LoRALinear(nn.Linear(6, 4), rank=3, alpha=6).eval()
    with torch.no_grad():
        layer.lora_b.normal_()
    x = torch.randn(2, 6)
    merged = layer.merged()

    torch.testing.assert_close(merged(x), layer(x), rtol=1e-5, atol=1e-6)
    assert merged.weight.data_ptr() != layer.base.weight.data_ptr()
