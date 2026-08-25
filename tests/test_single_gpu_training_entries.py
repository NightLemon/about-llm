from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.smoke]

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
SFT = PROJECT / "train_trl_sft.py"
QLORA = PROJECT / "train_qlora.py"


def _load_entry(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import training entry {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", [SFT, QLORA])
def test_training_entries_expose_preflight_and_step_limit(script: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert "--tokenization-preflight-only" in completed.stdout
    assert "--max-steps" in completed.stdout


def test_sft_entry_rejects_conflicting_precision_modes() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SFT),
            "--model-id",
            "model",
            "--revision",
            "revision",
            "--train-jsonl",
            "train.jsonl",
            "--readiness-json",
            "readiness.json",
            "--output-dir",
            "output",
            "--bf16",
            "--fp16",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 2
    assert "not allowed with argument" in completed.stderr


def test_sft_run_report_names_mixed_precision_and_memory_scope() -> None:
    module = _load_entry(SFT, "single_gpu_sft_entry")
    args = SimpleNamespace(
        model_id="Qwen/Qwen3-0.6B",
        revision="revision",
        target_modules="q_proj",
        rank=8,
        alpha=16,
        max_length=512,
        batch_size=1,
        gradient_accumulation=1,
        learning_rate=2e-4,
        epochs=1.0,
        max_steps=1,
        bf16=False,
        fp16=True,
        resume_from_checkpoint=None,
    )

    report = module._training_run_report(
        args=args,
        readiness_fingerprint="sha256:readiness",
        assistant_mask_fingerprint="sha256:mask",
        final_labels_fingerprint="sha256:labels",
        target_modules=["q_proj"],
        status="completed",
        optimizer_step_count=1,
        trainable_parameter_count=10,
        total_parameter_count=100,
        trainer_metrics={"train_loss": 1.0},
        runtime={"torch_version": "test"},
        memory_before={"cuda_executed": True},
        memory_after={"peak_allocated_bytes": 123},
    )

    assert report["training"]["trainer_mixed_precision_mode"] == "fp16"
    assert report["data"]["final_labels_fingerprint"] == "sha256:labels"
    assert report["outcome"]["optimizer_step_count"] == 1
    assert report["outcome"]["trainable_parameter_count"] == 10
    assert report["torch_cuda_allocator"]["after_train"] == {
        "peak_allocated_bytes": 123
    }
    assert "model and Trainer initialization happened earlier" in (
        report["torch_cuda_allocator"]["measurement_window"]
    )


@pytest.mark.parametrize(
    ("script", "extra_args"),
    [
        (SFT, []),
        (
            QLORA,
            [
                "--num-parameters",
                "600000000",
                "--num-layers",
                "28",
                "--hidden-size",
                "1024",
            ],
        ),
    ],
)
def test_training_entries_reject_zero_max_steps(
    script: Path, extra_args: list[str]
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model-id",
            "model",
            "--revision",
            "revision",
            "--train-jsonl",
            "train.jsonl",
            "--readiness-json",
            "readiness.json",
            "--output-dir",
            "output",
            "--max-steps",
            "0",
            *extra_args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 2
    assert "max-steps must be -1 or a positive integer" in completed.stderr
