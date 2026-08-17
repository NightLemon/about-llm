from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
patching = pytest.importorskip("about_llm.from_scratch.activation_patching")
gpt_torch = pytest.importorskip("about_llm.from_scratch.gpt_torch")

GPTConfig = gpt_torch.GPTConfig
MiniGPT = gpt_torch.MiniGPT
capture_block_residual = patching.capture_block_residual
normalized_patch_recovery = patching.normalized_patch_recovery
patch_block_residual = patching.patch_block_residual
run_residual_patch_experiment = patching.run_residual_patch_experiment

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "projects" / "transformers-basics" / "activation_patching.py"


def _fixture() -> tuple[MiniGPT, object, object]:
    torch.manual_seed(23)
    model = MiniGPT(
        GPTConfig(
            vocab_size=32,
            context_length=8,
            model_dim=16,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2,
        )
    ).eval()
    clean = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    corrupted = torch.tensor([[5, 2, 3, 4]], dtype=torch.long)
    return model, clean, corrupted


def _run(model: MiniGPT, clean: object, corrupted: object, positions: tuple[int, ...]):
    return run_residual_patch_experiment(
        model,
        clean,
        corrupted,
        layer_index=0,
        positions=positions,
        metric_position=1,
        positive_token_id=27,
        negative_token_id=19,
    )


def test_joint_prefix_patch_restores_metric_and_future_control_cannot_affect_past() -> None:
    model, clean, corrupted = _fixture()

    changed_source = _run(model, clean, corrupted, (0,))
    joint = _run(model, clean, corrupted, (0, 1))
    future_control = _run(model, clean, corrupted, (2,))

    assert changed_source.normalized_recovery == pytest.approx(0.43268338)
    assert joint.normalized_recovery == pytest.approx(1.0)
    assert joint.patched_metric == pytest.approx(joint.clean_metric)
    assert future_control.normalized_recovery == pytest.approx(0.0)
    assert future_control.patched_metric == pytest.approx(
        future_control.corrupted_metric
    )


def test_hook_is_removed_and_capture_is_a_detached_clone() -> None:
    model, clean, corrupted = _fixture()
    _, clean_activation = capture_block_residual(model, clean, layer_index=0)
    with torch.inference_mode():
        baseline, _ = model(corrupted)
    patched = patch_block_residual(
        model,
        corrupted,
        layer_index=0,
        clean_activation=clean_activation,
        positions=(0,),
    )
    with torch.inference_mode():
        after, _ = model(corrupted)

    assert not clean_activation.requires_grad
    assert not torch.equal(patched, baseline)
    torch.testing.assert_close(after, baseline)


def test_recovery_is_unclipped_and_small_denominator_fails_closed() -> None:
    assert normalized_patch_recovery(
        clean_metric=1.0,
        corrupted_metric=0.0,
        patched_metric=2.0,
    ) == pytest.approx(2.0)
    assert normalized_patch_recovery(
        clean_metric=1.0,
        corrupted_metric=0.0,
        patched_metric=-1.0,
    ) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="denominator is too small"):
        normalized_patch_recovery(
            clean_metric=1.0,
            corrupted_metric=1.0 + 1e-10,
            patched_metric=1.2,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("training", "model.eval"),
        ("shape", "identical shape"),
        ("duplicate_position", "must not contain duplicates"),
        ("future_out_of_range", r"must be in \[0, 4\)"),
        ("token_out_of_range", "outside the model vocabulary"),
    ],
)
def test_experiment_rejects_ambiguous_tensor_contracts(
    mutation: str, message: str
) -> None:
    model, clean, corrupted = _fixture()
    positions = (0,)
    if mutation == "training":
        model.train()
    elif mutation == "shape":
        corrupted = corrupted[:, :3]
    elif mutation == "duplicate_position":
        positions = (0, 0)
    elif mutation == "future_out_of_range":
        positions = (4,)
    else:
        corrupted = corrupted.clone()
        corrupted[0, 0] = 32

    with pytest.raises((ValueError, TypeError), match=message):
        _run(model, clean, corrupted, positions)


