from __future__ import annotations

import importlib.util
import json
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
DPO = PROJECT / "train_trl_dpo.py"
REWARD_MODEL = PROJECT / "train_reward_model.py"


def _load_entry(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import training entry {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", [SFT, QLORA, DPO, REWARD_MODEL])
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
    if script != DPO:
        assert "--max-steps" in completed.stdout


def test_dpo_entry_rejects_conflicting_preflight_modes() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(DPO),
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
            "--data-preflight-only",
            "--tokenization-preflight-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 2
    assert "choose at most one preflight-only mode" in completed.stderr


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


def test_qlora_estimate_only_prints_one_self_contained_json_report() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(QLORA),
            "--model-id",
            "authored/model",
            "--revision",
            "deadbeef",
            "--num-parameters",
            "600000000",
            "--num-layers",
            "28",
            "--hidden-size",
            "1024",
            "--estimate-only",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    report = json.loads(completed.stdout.decode("utf-8"))
    assert report["mode"] == "estimate_only"
    assert report["input"]["num_parameters"] == 600_000_000
    assert report["oom_degradation_order"]
    assert "formula-based memory estimate" in report["evidence_boundary"]


@pytest.mark.parametrize(
    ("script", "train_fixture", "readiness_fixture", "extra_args"),
    [
        (SFT, "train.example.jsonl", "sft-training-readiness.example.json", []),
        (
            QLORA,
            "train.example.jsonl",
            "sft-training-readiness.example.json",
            [
                "--num-parameters",
                "600000000",
                "--num-layers",
                "28",
                "--hidden-size",
                "1024",
            ],
        ),
        (
            DPO,
            "preference.train.example.jsonl",
            "preference-training-readiness.example.json",
            [],
        ),
        (
            REWARD_MODEL,
            "preference.train.example.jsonl",
            "preference-training-readiness.example.json",
            [],
        ),
    ],
)
def test_data_preflight_prints_completion_and_artifact_context(
    tmp_path: Path,
    script: Path,
    train_fixture: str,
    readiness_fixture: str,
    extra_args: list[str],
) -> None:
    output_directory = tmp_path / script.stem
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model-id",
            "authored/model",
            "--revision",
            "deadbeef",
            "--train-jsonl",
            str(PROJECT / train_fixture),
            "--readiness-json",
            str(PROJECT / readiness_fixture),
            "--output-dir",
            str(output_directory),
            "--data-preflight-only",
            *extra_args,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    report = json.loads(completed.stdout.decode("utf-8"))
    assert report["mode"] == "data_preflight"
    assert report["status"] == "completed"
    assert report["model_loaded"] is False
    assert all(Path(path).is_file() for path in report["artifacts"])


def test_dpo_tokenization_preflight_stops_before_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class LocalTokenizer:
        chat_template = "local deterministic test template"
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> LocalTokenizer:
            return cls()

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            add_generation_prompt: bool = False,
            tokenize: bool,
            return_dict: bool,
        ) -> object:
            del tokenize
            token_ids = [self.bos_token_id]
            role_tokens = {"system": 10, "user": 20, "assistant": 30}
            for message in messages:
                token_ids.append(role_tokens[message["role"]])
                token_ids.extend(100 + ord(character) for character in message["content"])
            if add_generation_prompt:
                token_ids.append(role_tokens["assistant"])
            return {"input_ids": token_ids} if return_dict else token_ids

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = LocalTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(DPO),
            "--model-id",
            "local-tokenizer",
            "--revision",
            "deadbeef",
            "--train-jsonl",
            str(PROJECT / "preference.train.example.jsonl"),
            "--readiness-json",
            str(PROJECT / "preference-training-readiness.example.json"),
            "--output-dir",
            str(tmp_path),
            "--tokenization-preflight-only",
        ],
    )
    module = _load_entry(DPO, "single_gpu_dpo_tokenization_preflight")

    module.main()

    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "tokenization_preflight"
    assert report["model_loaded"] is False
    assert report["tokenizer_loaded"] is True
    assert Path(report["artifacts"][0]).is_file()
    assert "policy and reference weights" in report["evidence_boundary"]
