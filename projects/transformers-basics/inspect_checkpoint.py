"""Inspect public checkpoint contracts without loading model weights."""

from __future__ import annotations

import argparse
import json
from typing import Any

from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from about_llm.generation_contract import inspect_generation_protocol
from about_llm.integrations.transformers_tools import render_chat
from about_llm.llmops import artifact_fingerprint
from about_llm.model_config import inspect_decoder_config

NORMALIZED_CONFIG_SNAPSHOT_SOURCE = (
    "AutoConfig.to_dict() after Transformers loading; may include library defaults and "
    "runtime metadata, so its fingerprint is not a hash of raw config.json bytes"
)
NORMALIZED_GENERATION_CONFIG_SNAPSHOT_SOURCE = (
    "GenerationConfig.to_dict() after Transformers loading; may include library "
    "defaults and runtime metadata, so its fingerprint is not a hash of raw "
    "generation_config.json bytes"
)


def inspect(model_id: str, revision: str) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_id,
        revision=revision,
        trust_remote_code=False,
    )
    messages = [{"role": "user", "content": "用一句话解释 attention。"}]
    chat_template_available = bool(getattr(tokenizer, "chat_template", None))
    rendered: str | None = None
    token_ids: list[int] | None = None
    if chat_template_available:
        rendered = render_chat(tokenizer, messages)
        raw_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if not isinstance(raw_token_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in raw_token_ids
        ):
            raise RuntimeError("chat template did not return a flat integer token list")
        token_ids = raw_token_ids
    config_mapping = config.to_dict()
    contract = inspect_decoder_config(config_mapping)
    generation_config_mapping: dict[str, Any] | None
    generation_config_status: str
    generation_config_error_type: str | None
    resolved_generation_config_commit: str | None
    try:
        generation_config = GenerationConfig.from_pretrained(
            model_id,
            revision=revision,
        )
    except OSError as error:
        generation_config_mapping = None
        generation_config_status = "unavailable_or_load_error"
        generation_config_error_type = type(error).__name__
        resolved_generation_config_commit = None
    else:
        generation_config_mapping = generation_config.to_dict()
        generation_config_status = "loaded"
        generation_config_error_type = None
        resolved_generation_config_commit = getattr(
            generation_config, "_commit_hash", None
        )
    chat_template = getattr(tokenizer, "chat_template", None)
    tokenizer_contract = {
        field: getattr(tokenizer, field, None)
        for field in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "decoder_start_token_id",
        )
    }
    tokenizer_contract["chat_template_fingerprint"] = (
        None
        if chat_template is None
        else "sha256:" + artifact_fingerprint({"chat_template": chat_template})
    )
    generation_protocol = inspect_generation_protocol(
        contract_id=f"{model_id}@{revision}",
        tokenizer_size=len(tokenizer),
        model_vocab_size=config_mapping.get("vocab_size"),
        tokenizer=tokenizer_contract,
        model_config=config_mapping,
        generation_config=generation_config_mapping,
    )
    return {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_config_commit": getattr(config, "_commit_hash", None),
        "model_type": config.model_type,
        "architectures": getattr(config, "architectures", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_available": chat_template_available,
        "rendered_prompt": rendered,
        "rendered_token_ids": token_ids,
        "token_count": None if token_ids is None else len(token_ids),
        "normalized_config_snapshot_source": NORMALIZED_CONFIG_SNAPSHOT_SOURCE,
        "normalized_config_contract": contract.to_dict(),
        "generation_config_status": generation_config_status,
        "generation_config_error_type": generation_config_error_type,
        "resolved_generation_config_commit": resolved_generation_config_commit,
        "normalized_generation_config_snapshot_source": (
            NORMALIZED_GENERATION_CONFIG_SNAPSHOT_SOURCE
        ),
        "generation_protocol_contract": generation_protocol.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.model_id, args.revision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
