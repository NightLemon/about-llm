from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scripts.check_content_accuracy import (
    check_claude_model_page,
    check_cloud_api_contracts_model_page,
    check_cloud_api_contracts_project_page,
    check_deepseek_model_page,
    check_evaluation_gate_project_page,
    check_gemini_model_page,
    check_gpt_model_page,
    check_inference_serving_project_page,
    check_jax_minigpt_project_page,
    check_jax_minigpt_project_readme,
    check_llama_model_page,
    check_openai_responses_replay,
    check_qwen_model_page,
    check_rag_foundations_project_page,
    check_rag_framework_adapters_project_page,
    check_rag_framework_adapters_project_readme,
    check_safe_agent_project_page,
    check_single_gpu_finetuning_project_page,
    check_synthetic_data_audit_project_page,
    check_transformers_basics_project_page,
)
from scripts.content_quality import check_ledger, text_files


def _write_registry(path: Path, sources: list[dict[str, str]]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "sources": sources}), encoding="utf-8"
    )


def test_text_files_excludes_local_environments_and_artifacts(tmp_path: Path) -> None:
    tracked_doc = tmp_path / "docs" / "guide.md"
    tracked_source = tmp_path / "src" / "module.py"
    local_artifact = tmp_path / "artifacts" / "report.json"
    local_environment = tmp_path / ".venv" / "package.py"
    unrelated_root_file = tmp_path / "notes.md"
    for path in (
        tracked_doc,
        tracked_source,
        local_artifact,
        local_environment,
        unrelated_root_file,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")

    discovered = {path.relative_to(tmp_path).as_posix() for path in text_files(tmp_path)}

    assert discovered == {"docs/guide.md", "src/module.py"}


def test_source_registry_accepts_current_complete_review(tmp_path: Path) -> None:
    url = "https://example.com/official"
    ledger = tmp_path / "accuracy.md"
    registry = tmp_path / "sources.json"
    ledger.write_text(f"# Accuracy\n\n{url}\n", encoding="utf-8")
    _write_registry(
        registry,
        [{"url": url, "checked_at": "2026-08-01", "scope": "test contract"}],
    )

    assert check_ledger(
        accuracy_page=ledger,
        source_registry=registry,
        as_of=date(2026, 8, 11),
        expected_urls={url},
    ) == []


def test_source_registry_rejects_stale_and_duplicate_reviews(tmp_path: Path) -> None:
    url = "https://example.com/official"
    ledger = tmp_path / "accuracy.md"
    registry = tmp_path / "sources.json"
    ledger.write_text(f"# Accuracy\n\n{url}\n", encoding="utf-8")
    source = {"url": url, "checked_at": "2026-01-01", "scope": "test contract"}
    _write_registry(registry, [source, source])

    errors = check_ledger(
        accuracy_page=ledger,
        source_registry=registry,
        as_of=date(2026, 8, 11),
        expected_urls={url},
    )

    assert any("stale" in error for error in errors)
    assert any("duplicate URL" in error for error in errors)


def test_evaluation_gate_page_rejects_navigation_only_summary(tmp_path: Path) -> None:
    page = tmp_path / "evaluation-gate.md"
    page.write_text(
        "# Evaluation Gate\n\nRun one command; see the project README.\n",
        encoding="utf-8",
    )

    errors = check_evaluation_gate_project_page(page)

    assert any("regressed to a navigation summary" in error for error in errors)
    assert any("missing workflow/scope marker" in error for error in errors)


def test_jax_minigpt_readme_rejects_run_only_summary(tmp_path: Path) -> None:
    page = tmp_path / "README.md"
    page.write_text(
        "# JAX MiniGPT\n\nInstall JAX and run train_tiny.py.\n",
        encoding="utf-8",
    )

    errors = check_jax_minigpt_project_readme(page)

    assert any("regressed to a run/control summary" in error for error in errors)
    assert any("README missing workflow/scope marker" in error for error in errors)


def test_inference_serving_page_rejects_control_only_summary(tmp_path: Path) -> None:
    page = tmp_path / "inference-serving.md"
    page.write_text(
        "# Inference Serving\n\nRun one recorded control; see the project README.\n",
        encoding="utf-8",
    )

    errors = check_inference_serving_project_page(page)

    assert any("regressed to a control summary" in error for error in errors)
    assert any("missing workflow/scope marker" in error for error in errors)


def test_rag_page_rejects_retrieval_only_summary(tmp_path: Path) -> None:
    page = tmp_path / "rag-foundations.md"
    page.write_text(
        "# RAG Foundations\n\nRun one BM25 retrieval; see the project README.\n",
        encoding="utf-8",
    )

    errors = check_rag_foundations_project_page(page)

    assert any("regressed to a retrieval/control summary" in error for error in errors)
    assert any("missing workflow/scope marker" in error for error in errors)


def test_rag_framework_adapter_page_rejects_mapping_only_summary(
    tmp_path: Path,
) -> None:
    page = tmp_path / "rag-framework-adapters.md"
    page.write_text(
        "# RAG Framework Adapters\n\nMap one document into two frameworks.\n",
        encoding="utf-8",
    )

    errors = check_rag_framework_adapters_project_page(page)

    assert any("regressed to an adapter summary" in error for error in errors)
    assert any("missing workflow/scope marker" in error for error in errors)


def test_rag_framework_adapter_readme_rejects_quickstart_only(
    tmp_path: Path,
) -> None:
    page = tmp_path / "README.md"
    page.write_text(
        "# RAG Framework Adapters\n\nInstall extras and run parity_control.py.\n",
        encoding="utf-8",
    )

    errors = check_rag_framework_adapters_project_readme(page)

    assert any("regressed to a quickstart" in error for error in errors)
    assert any("missing run/evidence/production/scope marker" in error for error in errors)


def test_cloud_api_page_rejects_demo_only_summary(tmp_path: Path) -> None:
    page = tmp_path / "cloud-api-contracts.md"
    page.write_text(
        "# Cloud API Contracts\n\nRun one offline adapter demo; see the README.\n",
        encoding="utf-8",
    )

    errors = check_cloud_api_contracts_project_page(page)

    assert any("regressed to a demo summary" in error for error in errors)
    assert any("missing workflow/scope marker" in error for error in errors)


def test_cloud_api_model_page_rejects_provider_table_only_summary(
    tmp_path: Path,
) -> None:
    page = tmp_path / "cloud-api-contracts.md"
    page.write_text(
        "# Cloud API\n\nCompare three provider message fields in one table.\n",
        encoding="utf-8",
    )

    errors = check_cloud_api_contracts_model_page(page)

    assert any("regressed to a provider table/control summary" in error for error in errors)
    assert any("missing protocol/production/scope marker" in error for error in errors)


def test_gpt_page_rejects_product_only_summary(tmp_path: Path) -> None:
    page = tmp_path / "gpt.md"
    page.write_text(
        "# GPT\n\nRead the current model catalog and call one API.\n",
        encoding="utf-8",
    )

    errors = check_gpt_model_page(page)

    assert any("regressed to a product/API summary" in error for error in errors)
    assert any("missing research/API/scope marker" in error for error in errors)


def test_llama_page_rejects_family_only_summary(tmp_path: Path) -> None:
    page = tmp_path / "llama.md"
    page.write_text(
        "# Llama\n\nRead one model card and deploy a Llama-family checkpoint.\n",
        encoding="utf-8",
    )

    errors = check_llama_model_page(page)

    assert any("regressed to a family/config summary" in error for error in errors)
    assert any("missing architecture/release/scope marker" in error for error in errors)


def test_qwen_page_rejects_family_or_control_catalog(tmp_path: Path) -> None:
    page = tmp_path / "qwen.md"
    page.write_text(
        "# Qwen\n\nList several model generations and run one control.\n",
        encoding="utf-8",
    )

    errors = check_qwen_model_page(page)

    assert any("regressed to a family/control catalog" in error for error in errors)
    assert any("missing release/runtime/training/scope marker" in error for error in errors)


def test_deepseek_page_rejects_architecture_or_control_summary(
    tmp_path: Path,
) -> None:
    page = tmp_path / "deepseek.md"
    page.write_text(
        "# DeepSeek\n\nExplain MLA, MoE, and R1 in three paragraphs.\n",
        encoding="utf-8",
    )

    errors = check_deepseek_model_page(page)

    assert any("regressed to an architecture/control summary" in error for error in errors)
    assert any("missing config/MLA/MoE/training/scope marker" in error for error in errors)


def test_claude_page_rejects_research_or_api_summary(tmp_path: Path) -> None:
    page = tmp_path / "claude.md"
    page.write_text(
        "# Claude\n\nSummarize Constitutional AI and send one Messages request.\n",
        encoding="utf-8",
    )

    errors = check_claude_model_page(page)

    assert any("regressed to a research/API summary" in error for error in errors)
    assert any(
        "missing Messages/stream/tool/budget/scope marker" in error
        for error in errors
    )


def test_gemini_page_rejects_platform_or_multimodal_summary(tmp_path: Path) -> None:
    page = tmp_path / "gemini.md"
    page.write_text(
        "# Gemini\n\nCompare Gemini API and Vertex AI, then send one image.\n",
        encoding="utf-8",
    )

    errors = check_gemini_model_page(page)

    assert any(
        "regressed to a platform/multimodal summary" in error for error in errors
    )
    assert any(
        "missing Interactions/generateContent/multimodal/production/scope marker"
        in error
        for error in errors
    )


def test_openai_responses_accuracy_gate_accepts_fixed_fixture() -> None:
    assert check_openai_responses_replay() == []


def test_openai_responses_accuracy_gate_rejects_fixture_drift(tmp_path: Path) -> None:
    fixture = tmp_path / "events.jsonl"
    fixture.write_text("{}\n", encoding="utf-8")

    errors = check_openai_responses_replay(fixture)

    assert any("input identity mismatch" in error for error in errors)
    assert any("replay failed" in error for error in errors)


def test_safe_agent_page_rejects_protocol_only_summary(tmp_path: Path) -> None:
    page = tmp_path / "safe-agent.md"
    page.write_text(
        "# Safe Agent\n\nRun one protocol control; see the project README.\n",
        encoding="utf-8",
    )

    errors = check_safe_agent_project_page(page)

    assert any("regressed to a protocol-control summary" in error for error in errors)
    assert any("missing workflow/scope marker" in error for error in errors)


def test_synthetic_data_page_rejects_cli_only_summary(tmp_path: Path) -> None:
    page = tmp_path / "synthetic-data-audit.md"
    page.write_text("# Synthetic Data Audit\n\nRun the CLI.\n", encoding="utf-8")

    errors = check_synthetic_data_audit_project_page(page)

    assert any("regressed to a CLI summary" in error for error in errors)
    assert any("missing workflow/scope marker" in error for error in errors)


def test_single_gpu_finetuning_page_rejects_control_only_catalog(
    tmp_path: Path,
) -> None:
    page = tmp_path / "single-gpu-finetuning.md"
    page.write_text(
        "# Single-GPU Finetuning\n\nRun several CPU controls; see the README.\n",
        encoding="utf-8",
    )

    errors = check_single_gpu_finetuning_project_page(page)

    assert any("regressed to a control catalog" in error for error in errors)
    assert any("missing workflow/scope marker" in error for error in errors)


def test_jax_minigpt_page_rejects_control_only_summary(tmp_path: Path) -> None:
    page = tmp_path / "jax-minigpt.md"
    page.write_text(
        "# JAX MiniGPT\n\n"
        "检查运行设备\N{IDEOGRAPHIC COMMA}初始/最终 loss 和生成结果。\n",
        encoding="utf-8",
    )

    errors = check_jax_minigpt_project_page(page)

    assert any("regressed to a control summary" in error for error in errors)
    assert any("missing workflow/scope marker" in error for error in errors)
    assert any("incorrectly claims" in error for error in errors)


def test_transformers_basics_page_rejects_control_only_summary(
    tmp_path: Path,
) -> None:
    page = tmp_path / "transformers-basics.md"
    page.write_text(
        "# Transformers Basics\n\n"
        "Run one tiny model and several MoE controls; see the project README.\n",
        encoding="utf-8",
    )

    errors = check_transformers_basics_project_page(page)

    assert any("regressed to a control summary" in error for error in errors)
    assert any("missing workflow/scope marker" in error for error in errors)
