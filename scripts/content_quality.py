"""Shared file-discovery and source-freshness checks for documentation quality."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ACCURACY_LEDGER = ROOT / "docs" / "evidence" / "accuracy-ledger.md"
SOURCE_REGISTRY = ROOT / "docs" / "reference" / "official-sources.json"

TEXT_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".toml", ".json", ".jsonl"}
CONTENT_ROOTS = (".github", "docs", "notebooks", "projects", "scripts", "src", "tests")
TOP_LEVEL_TEXT_FILES = (
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "LICENSE-CODE",
    "README.md",
    "mkdocs.yml",
    "pyproject.toml",
)
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "htmlcov",
    "node_modules",
    "outputs",
    "site",
    "site-packages",
}
MAX_SOURCE_AGE_DAYS = 90

OFFICIAL_URLS = {
    "https://developers.openai.com/api/docs/models",
    "https://developers.openai.com/api/reference/resources/responses/methods/create",
    "https://developers.openai.com/api/docs/guides/streaming-responses",
    "https://developers.openai.com/api/reference/resources/responses/streaming-events",
    "https://developers.openai.com/api/docs/guides/structured-outputs",
    "https://developers.openai.com/api/docs/guides/function-calling",
    "https://developers.openai.com/api/docs/guides/evals",
    "https://platform.claude.com/docs/en/api/messages",
    "https://ai.google.dev/gemini-api/docs/interactions-overview",
    "https://ai.google.dev/api/interactions-api",
    "https://ai.google.dev/gemini-api/docs/streaming",
    "https://ai.google.dev/api/generate-content",
    "https://ai.google.dev/gemini-api/docs/text-generation",
    "https://modelcontextprotocol.io/docs/getting-started/intro",
    "https://modelcontextprotocol.io/specification/",
    "https://modelcontextprotocol.io/specification/2025-11-25/basic/transports",
    "https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle",
    "https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/cancellation",
    "https://modelcontextprotocol.io/specification/2025-11-25/server/tools",
    "https://a2a-protocol.org/latest/specification/",
    "https://a2a-protocol.org/v1.0.0/spec/a2a.json",
    "https://github.com/a2aproject/a2a-python",
    "https://raw.githubusercontent.com/meta-llama/llama-models/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3_2/MODEL_CARD.md",
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/config.json",
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors",
    "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/e815299b0bcbac849fa540c768ef21845365c9eb/config.json",
    "https://huggingface.co/docs/trl/en/sft_trainer",
    "https://docs.vllm.ai/en/stable/cli/",
    "https://huggingface.co/docs/transformers/en/chat_templating",
    "https://huggingface.co/papers/week/2026-W32",
    "https://huggingface.co/papers/month/2026-07",
    "https://arxiv.org/abs/2607.24653",
    "https://arxiv.org/abs/2608.05466",
    "https://arxiv.org/abs/2608.01964",
    "https://arxiv.org/abs/2608.02023",
    "https://arxiv.org/abs/2606.30534",
    "https://arxiv.org/abs/2607.19191",
    "https://arxiv.org/abs/2608.10296",
    "https://arxiv.org/abs/2608.09867",
}


def text_files(root: Path = ROOT) -> list[Path]:
    """Return repository-owned text inputs without local environments or artifacts."""
    candidates = [root / name for name in TOP_LEVEL_TEXT_FILES]
    for relative_root in CONTENT_ROOTS:
        directory = root / relative_root
        if directory.exists():
            candidates.extend(directory.rglob("*"))
    return sorted(
        {
            path
            for path in candidates
            if path.is_file()
            and (path.name in TOP_LEVEL_TEXT_FILES or path.suffix.lower() in TEXT_SUFFIXES)
            and not IGNORED_PARTS.intersection(path.relative_to(root).parts)
        }
    )


def check_encoding(files: list[Path], root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for path in files:
        if "\ufffd" in path.read_text(encoding="utf-8"):
            errors.append(f"{path.relative_to(root)}: contains Unicode replacement character")
    return errors


def _load_registry(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], [f"{path}: missing official source registry"]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return [], [f"{path}: invalid official source registry: {error}"]
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return [], [f"{path}: expected schema_version 1 object"]
    sources = payload.get("sources")
    if not isinstance(sources, list) or not all(isinstance(item, dict) for item in sources):
        return [], [f"{path}: sources must be an array of objects"]
    return sources, []


def check_ledger(
    *,
    accuracy_page: Path = ACCURACY_LEDGER,
    source_registry: Path = SOURCE_REGISTRY,
    as_of: date | None = None,
    expected_urls: set[str] = OFFICIAL_URLS,
) -> list[str]:
    """Validate source coverage and force time-sensitive facts through periodic review."""
    if not accuracy_page.exists():
        return [f"{accuracy_page}: missing accuracy ledger"]
    ledger_text = accuracy_page.read_text(encoding="utf-8")
    sources, errors = _load_registry(source_registry)
    if errors:
        return errors

    effective_date = as_of or date.today()
    seen_urls: set[str] = set()
    for index, source in enumerate(sources):
        prefix = f"{source_registry}: sources[{index}]"
        url = source.get("url")
        checked_at_raw = source.get("checked_at")
        scope = source.get("scope")
        if not isinstance(url, str) or not url.startswith("https://"):
            errors.append(f"{prefix}: url must be an HTTPS string")
            continue
        if url in seen_urls:
            errors.append(f"{prefix}: duplicate URL: {url}")
        seen_urls.add(url)
        if not isinstance(scope, str) or not scope.strip():
            errors.append(f"{prefix}: scope must be a non-empty string")
        try:
            checked_at = date.fromisoformat(checked_at_raw)
        except (TypeError, ValueError):
            errors.append(f"{prefix}: checked_at must be an ISO date")
            continue
        age_days = (effective_date - checked_at).days
        if age_days < 0:
            errors.append(f"{prefix}: checked_at is in the future: {checked_at_raw}")
        elif age_days > MAX_SOURCE_AGE_DAYS:
            errors.append(
                f"{prefix}: source review is stale ({age_days} days; "
                f"maximum {MAX_SOURCE_AGE_DAYS}): {url}"
            )
        if url not in ledger_text:
            errors.append(f"accuracy ledger missing official URL: {url}")

    for missing_url in sorted(expected_urls - seen_urls):
        errors.append(f"official source registry missing URL: {missing_url}")
    for unexpected_url in sorted(seen_urls - expected_urls):
        errors.append(f"official source registry has unapproved URL: {unexpected_url}")
    return errors
