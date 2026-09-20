from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
lora_module = pytest.importorskip("about_llm.finetuning.lora")
LoRALinear = lora_module.LoRALinear
pytestmark = [pytest.mark.formula, pytest.mark.integration]


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
    base = nn.Linear(2, 2, bias=True, dtype=torch.float64)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[2.0, -1.0], [0.5, 3.0]]))
        base.bias.copy_(torch.tensor([0.25, -0.5]))
    layer = LoRALinear(base, rank=1, alpha=2).eval()
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[1.5, -2.0]], dtype=torch.float64))
        layer.lora_b.copy_(torch.tensor([[4.0], [-0.25]], dtype=torch.float64))
    x = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    merged = layer.merged()

    expected_weight = torch.tensor([[14.0, -17.0], [-0.25, 4.0]], dtype=torch.float64)
    # Hand calculation: W + 2BA is expected_weight; [2, -1] W^T + bias
    # gives [28 + 17 + 0.25, -0.5 - 4 - 0.5].
    expected = torch.tensor([[45.25, -5.0]], dtype=torch.float64)
    torch.testing.assert_close(merged.weight, expected_weight)
    torch.testing.assert_close(layer(x), expected)
    torch.testing.assert_close(merged(x), expected)
    assert merged.weight.data_ptr() != layer.base.weight.data_ptr()
    assert merged.bias is not None and merged.bias.data_ptr() != layer.base.bias.data_ptr()
    assert merged.weight.dtype is torch.float64 and merged.weight.device == base.weight.device


def test_adapter_parameters_follow_base_dtype_and_meta_device() -> None:
    layer = LoRALinear(nn.Linear(3, 2, dtype=torch.float64), rank=1)
    assert layer.lora_a.dtype is torch.float64
    assert layer.lora_b.dtype is torch.float64

    meta_layer = LoRALinear(nn.Linear(3, 2, device="meta"), rank=1)
    assert meta_layer.lora_a.device.type == "meta"
    assert meta_layer.lora_b.device.type == "meta"
