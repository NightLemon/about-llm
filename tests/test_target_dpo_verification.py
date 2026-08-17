from __future__ import annotations

from pathlib import Path

from about_llm.finetuning.target_dpo_control import (
    load_target_dpo_control_spec,
    verify_recorded_target_dpo_report,
)
from about_llm.finetuning.target_dpo_verification import (
    verify_recorded_target_dpo_report as verify_isolated_report,
)
from about_llm.integrations.transformers_checkpoint_control import (
    load_checkpoint_control_spec,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "single-gpu-finetuning"
CHECKPOINT_CONTROL = (
    ROOT
    / "projects"
    / "transformers-basics"
    / "target-checkpoints"
    / "qwen2.5-0.5b-instruct.control.json"
)
CHECKPOINT_REPORT = CHECKPOINT_CONTROL.with_name(
    "qwen2.5-0.5b-instruct.recorded-report.json"
)


def test_public_target_dpo_verifier_delegates_without_projection_drift() -> None:
    checkpoint_spec = load_checkpoint_control_spec(CHECKPOINT_CONTROL)
    spec = load_target_dpo_control_spec(
        PROJECT / "qwen2.5-0.5b-dpo.control.json",
        checkpoint_spec=checkpoint_spec,
    )
    training_path = PROJECT / "preference.train.example.jsonl"
    readiness_path = PROJECT / "preference-training-readiness.example.json"
    report_path = PROJECT / "qwen2.5-0.5b-dpo.recorded-report.json"

    public_report = verify_recorded_target_dpo_report(
        report_path,
        spec=spec,
        checkpoint_spec=checkpoint_spec,
        checkpoint_report_path=CHECKPOINT_REPORT,
        training_path=training_path,
        readiness_path=readiness_path,
    )
    isolated_report = verify_isolated_report(
        report_path,
        spec=spec,
        checkpoint_spec=checkpoint_spec,
        checkpoint_report_path=CHECKPOINT_REPORT,
        training_path=training_path,
        readiness_path=readiness_path,
    )

    assert public_report == isolated_report
