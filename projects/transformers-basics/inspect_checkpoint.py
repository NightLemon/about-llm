"""Inspect public checkpoint contracts without loading model weights."""

from __future__ import annotations

import argparse
import json
from typing import Any

from transformers import AutoConfig, AutoTokenizer

from about_llm.integrations.transformers_tools import render_chat


def inspect(model_id: str, revision: str) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
    )
    messages = [{"role": "user", "content": "用一句话解释 attention。"}]
    rendered = render_chat(tokenizer, messages)
    return {
        "model_id": model_id,
        "revision": revision,
        "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "tokenizer_class": type(tokenizer).__name__,
        "rendered_prompt": rendered,
        "token_count": len(tokenizer.encode(rendered)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.model_id, args.revision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
