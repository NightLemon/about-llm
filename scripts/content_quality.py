"""Shared file-discovery and source-freshness checks for documentation quality."""

from __future__ import annotations

import json
import re
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
SOURCE_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
SOURCE_MARKER_RE = re.compile(r"\[SOURCE:([a-z][a-z0-9]*(?:-[a-z0-9]+)*)\]")
SOURCE_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_STATUSES = {"verified", "pending-review", "rejected"}
SOURCE_VOLATILITIES = {"high", "medium", "low", "immutable"}
SOURCE_FINGERPRINT_KINDS = {"visible-text-v1", "raw-prefix-v1"}

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


def load_registry(path: Path = SOURCE_REGISTRY) -> tuple[list[dict[str, Any]], list[str]]:
    """Load the versioned source registry without performing network access."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], [f"{path}: missing official source registry"]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return [], [f"{path}: invalid official source registry: {error}"]
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        return [], [f"{path}: expected schema_version 2 object"]
    sources = payload.get("sources")
    if not isinstance(sources, list) or not all(isinstance(item, dict) for item in sources):
        return [], [f"{path}: sources must be an array of objects"]
    return sources, []


def effective_source_status(
    source: dict[str, Any],
    *,
    as_of: date | None = None,
    probe: dict[str, Any] | None = None,
) -> str:
    """Combine reviewed state, review deadline, and an optional ephemeral probe result."""
    status = source.get("status")
    if status != "verified":
        return str(status)
    if probe is not None:
        probe_status = probe.get("status")
        if probe_status in {"unknown", "pending-review"}:
            return str(probe_status)
    try:
        next_review_at = date.fromisoformat(str(source["next_review_at"]))
    except (KeyError, ValueError):
        return "unknown"
    if (as_of or date.today()) > next_review_at:
        return "stale"
    return "verified"


def check_ledger(
    *,
    accuracy_page: Path = ACCURACY_LEDGER,
    source_registry: Path = SOURCE_REGISTRY,
    as_of: date | None = None,
    expected_urls: set[str] = OFFICIAL_URLS,
    docs_root: Path | None = None,
) -> list[str]:
    """Validate source coverage and markers using deterministic, offline checks only."""
    if not accuracy_page.exists():
        return [f"{accuracy_page}: missing accuracy ledger"]
    ledger_text = accuracy_page.read_text(encoding="utf-8")
    sources, errors = load_registry(source_registry)
    if errors:
        return errors

    effective_date = as_of or date.today()
    effective_docs_root = docs_root or ROOT / "docs"
    seen_urls: set[str] = set()
    seen_ids: set[str] = set()
    sources_by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources):
        prefix = f"{source_registry}: sources[{index}]"
        source_id = source.get("id")
        url = source.get("url")
        checked_at_raw = source.get("checked_at")
        next_review_at_raw = source.get("next_review_at")
        scope = source.get("scope")
        status = source.get("status")
        volatility = source.get("volatility")
        used_by = source.get("used_by")
        fingerprint = source.get("fingerprint")
        fingerprint_kind = source.get("fingerprint_kind")
        fingerprint_bytes = source.get("fingerprint_bytes")
        revision = source.get("revision")
        if not isinstance(source_id, str) or SOURCE_ID_RE.fullmatch(source_id) is None:
            errors.append(f"{prefix}: id must be a stable lowercase kebab-case identifier")
        elif source_id in seen_ids:
            errors.append(f"{prefix}: duplicate id: {source_id}")
        else:
            seen_ids.add(source_id)
            sources_by_id[source_id] = source
        if not isinstance(url, str) or not url.startswith("https://"):
            errors.append(f"{prefix}: url must be an HTTPS string")
            continue
        if url in seen_urls:
            errors.append(f"{prefix}: duplicate URL: {url}")
        seen_urls.add(url)
        if not isinstance(scope, str) or not scope.strip():
            errors.append(f"{prefix}: scope must be a non-empty string")
        if status not in SOURCE_STATUSES:
            errors.append(f"{prefix}: status must be one of {sorted(SOURCE_STATUSES)}")
        if volatility not in SOURCE_VOLATILITIES:
            errors.append(f"{prefix}: volatility must be one of {sorted(SOURCE_VOLATILITIES)}")
        if fingerprint is not None:
            if (
                not isinstance(fingerprint, str)
                or SOURCE_FINGERPRINT_RE.fullmatch(fingerprint) is None
            ):
                errors.append(f"{prefix}: fingerprint must be sha256 followed by 64 lowercase hex")
            if fingerprint_kind not in SOURCE_FINGERPRINT_KINDS:
                errors.append(
                    f"{prefix}: fingerprint_kind must be one of "
                    f"{sorted(SOURCE_FINGERPRINT_KINDS)}"
                )
            if (
                not isinstance(fingerprint_bytes, int)
                or isinstance(fingerprint_bytes, bool)
                or not 0 < fingerprint_bytes <= 512 * 1024
            ):
                errors.append(f"{prefix}: fingerprint_bytes must be between 1 and 524288")
        elif fingerprint_kind is not None or fingerprint_bytes is not None:
            errors.append(f"{prefix}: fingerprint metadata requires fingerprint")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            errors.append(f"{prefix}: revision must be a non-empty string when present")
        try:
            checked_at = date.fromisoformat(checked_at_raw)
        except (TypeError, ValueError):
            errors.append(f"{prefix}: checked_at must be an ISO date")
            continue
        try:
            next_review_at = date.fromisoformat(next_review_at_raw)
        except (TypeError, ValueError):
            errors.append(f"{prefix}: next_review_at must be an ISO date")
            next_review_at = None
        if checked_at > effective_date:
            errors.append(f"{prefix}: checked_at is in the future: {checked_at_raw}")
        if next_review_at is not None and next_review_at < checked_at:
            errors.append(f"{prefix}: next_review_at precedes checked_at")
        if not isinstance(used_by, list) or not used_by:
            errors.append(f"{prefix}: used_by must be a non-empty array of documentation paths")
        else:
            for raw_path in used_by:
                if not isinstance(raw_path, str) or not raw_path.endswith(".md"):
                    errors.append(f"{prefix}: used_by entries must be Markdown paths")
                    continue
                page = (effective_docs_root / raw_path).resolve()
                try:
                    page.relative_to(effective_docs_root.resolve())
                except ValueError:
                    errors.append(f"{prefix}: used_by path escapes docs root: {raw_path}")
                    continue
                if not page.is_file():
                    errors.append(f"{prefix}: used_by page does not exist: {raw_path}")
        if url not in ledger_text:
            errors.append(f"accuracy ledger missing official URL: {url}")

    for missing_url in sorted(expected_urls - seen_urls):
        errors.append(f"official source registry missing URL: {missing_url}")
    for unexpected_url in sorted(seen_urls - expected_urls):
        errors.append(f"official source registry has unapproved URL: {unexpected_url}")

    if effective_docs_root.is_dir():
        marked_ids: set[str] = set()
        for page in effective_docs_root.rglob("*.md"):
            relative_page = page.relative_to(effective_docs_root).as_posix()
            for source_id in SOURCE_MARKER_RE.findall(page.read_text(encoding="utf-8")):
                source = sources_by_id.get(source_id)
                if source is None:
                    errors.append(f"{relative_page}: unknown source marker: {source_id}")
                    continue
                marked_ids.add(source_id)
                if relative_page not in source.get("used_by", []):
                    errors.append(
                        f"{relative_page}: source marker {source_id} is missing "
                        "from registry used_by"
                    )
        for source_id, source in sources_by_id.items():
            if source.get("volatility") in {"high", "medium"} and source_id not in marked_ids:
                errors.append(f"source {source_id}: high/medium volatility requires a page marker")
    return errors
