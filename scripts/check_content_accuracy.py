"""Check durable fact boundaries and executable numeric claims in the textbook."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ACCURACY_PAGE = ROOT / "docs" / "reference" / "accuracy.md"
TEXT_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".toml", ".jsonl"}
IGNORED_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site"}

OFFICIAL_URLS = {
    "https://developers.openai.com/api/docs/guides/structured-outputs",
    "https://docs.anthropic.com/en/api/messages",
    "https://ai.google.dev/gemini-api/docs/text-generation",
    "https://huggingface.co/docs/trl/en/sft_trainer",
    "https://docs.vllm.ai/en/stable/cli/",
    "https://huggingface.co/docs/transformers/en/chat_templating",
}

MODEL_BOUNDARIES = {
    "gpt.md": ("未披露", "时间敏感"),
    "llama.md": ("以 checkpoint", "config"),
    "qwen.md": ("不能用一个架构", "检查 checkpoint"),
    "deepseek.md": ("具体 checkpoint", "不能"),
    "claude.md": ("保持未知", "不要"),
    "gemini.md": ("2026-08-05", "Interactions API", "generateContent"),
}


def text_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def check_encoding(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if "\ufffd" in path.read_text(encoding="utf-8"):
            errors.append(f"{path.relative_to(ROOT)}: contains Unicode replacement character")
    return errors


def check_ledger() -> list[str]:
    if not ACCURACY_PAGE.exists():
        return ["docs/reference/accuracy.md: missing accuracy ledger"]
    text = ACCURACY_PAGE.read_text(encoding="utf-8")
    errors = [
        f"accuracy ledger missing official URL: {url}"
        for url in OFFICIAL_URLS
        if url not in text
    ]
    if "2026-08-05" not in text:
        errors.append("accuracy ledger missing checked_at date 2026-08-05")
    return errors


def check_model_boundaries() -> list[str]:
    errors: list[str] = []
    model_dir = ROOT / "docs" / "models"
    for filename, markers in MODEL_BOUNDARIES.items():
        text = (model_dir / filename).read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            errors.append(f"docs/models/{filename}: missing boundary marker(s): {missing}")
    return errors


def check_stream_token_accounting() -> list[str]:
    benchmark = ROOT / "projects" / "inference-serving" / "benchmark_openai.py"
    text = benchmark.read_text(encoding="utf-8")
    errors: list[str] = []
    forbidden = ("observed_chunks", 'usage.get("completion_tokens") or')
    for fragment in forbidden:
        if fragment in text:
            errors.append(f"{benchmark.relative_to(ROOT)}: forbidden token fallback: {fragment}")
    required = ("SSE chunks are not tokens", 'usage.get("completion_tokens") is None')
    for fragment in required:
        if fragment not in text:
            errors.append(f"{benchmark.relative_to(ROOT)}: missing strict token check: {fragment}")
    return errors


def check_kv_example() -> list[str]:
    sys.path.insert(0, str(SRC))
    from about_llm.inference import estimate_kv_cache_bytes

    actual = estimate_kv_cache_bytes(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        tokens=8192,
        bytes_per_element=2,
    )
    expected = 1024**3
    if actual != expected:
        return [f"KV example mismatch: expected {expected} bytes, got {actual}"]
    return []


def main() -> int:
    files = text_files()
    errors = (
        check_encoding(files)
        + check_ledger()
        + check_model_boundaries()
        + check_stream_token_accounting()
        + check_kv_example()
    )
    if errors:
        print("Content accuracy checks failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(
        f"OK: checked {len(files)} text files, {len(MODEL_BOUNDARIES)} model boundaries, "
        f"{len(OFFICIAL_URLS)} official sources, strict stream token accounting, and KV math"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
