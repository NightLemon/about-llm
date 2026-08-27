"""追踪一条中文对话怎样经过 Qwen3 chat template 变成 token IDs。

实验只加载固定版本的 tokenizer，不加载模型权重。它把应用层 message、模板渲染文本、
最终 ID、词表 token 和逐 token 解码并排展示，帮助区分“模板加了什么”和“分词器切了什么”。
"""

# ruff: noqa: RUF001 -- Full-width punctuation is intentional in learner-facing text.

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoTokenizer

# 仓库名与完整 commit 共同固定 tokenizer 身份，避免上游分支变化后结果悄悄漂移。
MODEL_REPOSITORY = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
REPORT_SCHEMA = "about-llm.qwen3-tokenizer-trace.v1"
DEFAULT_MESSAGE = "请用一句话解释：为什么生成下一个 token 时可以复用 KV Cache？"


def _flat_integer_ids(value: object, *, source: str) -> list[int]:
    """确认第三方 tokenizer 返回的是一维整数 ID，而不是张量或嵌套 batch。"""

    if not isinstance(value, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in value
    ):
        raise RuntimeError(f"{source} did not return a flat list of integer token IDs")
    return value


def _serializable_special_tokens(value: object) -> object:
    """递归把 tokenizer 元数据转换成 JSON 能稳定保存的基础类型。"""

    if isinstance(value, Mapping):
        return {
            str(key): _serializable_special_tokens(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serializable_special_tokens(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def trace_chat_message(
    tokenizer: Any,
    *,
    message: str = DEFAULT_MESSAGE,
    load_source: str = MODEL_REPOSITORY,
    source_kind: str = "repository_revision",
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """让一条 user message 走过模板、编码、token 映射和解码往返。"""

    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("the loaded tokenizer does not provide a chat template")

    # chat template 接收带 role 的消息列表，而不是已经拼好的 prompt 字符串。
    messages = [{"role": "user", "content": message}]
    template_options = {
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    # 第一次只渲染文本，直接观察模板插入的控制 token、换行和 assistant 起始标记。
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        **template_options,
    )
    if not isinstance(rendered, str):
        raise RuntimeError("chat template rendering did not return text")

    # 第二次让同一模板直接产生 ID，代表正常推理入口最接近的 tokenization 路径。
    direct_ids = _flat_integer_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            **template_options,
        ),
        source="chat template tokenization",
    )
    # 再编码渲染文本作为交叉检查；禁止额外 special token，避免模板标记被重复添加。
    encoded = tokenizer(rendered, add_special_tokens=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise RuntimeError("encoding the rendered prompt did not return input_ids")
    rendered_ids = _flat_integer_ids(
        encoded["input_ids"],
        source="rendered prompt encoding",
    )

    # convert_ids_to_tokens 返回词表内部写法，decode 则尽力恢复面向人的字符片段。
    raw_tokens = tokenizer.convert_ids_to_tokens(direct_ids)
    if not isinstance(raw_tokens, list) or any(
        not isinstance(token, str) for token in raw_tokens
    ):
        raise RuntimeError("token conversion did not return one string per token ID")
    if len(raw_tokens) != len(direct_ids):
        raise RuntimeError("token conversion returned a different number of pieces")

    # special 与 added 是不同概念：模板控制词可能是 added token，但未必被标为 special。
    raw_special_ids = getattr(tokenizer, "all_special_ids", [])
    special_ids = {
        token_id
        for token_id in raw_special_ids
        if isinstance(token_id, int) and not isinstance(token_id, bool)
    }
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    raw_added_vocab = get_added_vocab() if callable(get_added_vocab) else {}
    if not isinstance(raw_added_vocab, Mapping):
        raise RuntimeError("tokenizer.get_added_vocab did not return a mapping")
    added_ids = {
        token_id
        for token_id in raw_added_vocab.values()
        if isinstance(token_id, int) and not isinstance(token_id, bool)
    }
    # 为每个位置保留 ID、词表形式、解码片段和 token 类型，形成可逐行阅读的 trace。
    token_rows = []
    for position, token_id in enumerate(direct_ids):
        decoded_piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded_piece, str):
            raise RuntimeError("single-token decoding did not return text")
        token_rows.append(
            {
                "position": position,
                "token_id": token_id,
                "vocabulary_token": raw_tokens[position],
                "decoded_piece": decoded_piece,
                "tokenizer_kind": (
                    "special"
                    if token_id in special_ids
                    else "added"
                    if token_id in added_ids
                    else "regular"
                ),
            }
        )
    # 最后把整串 ID 一次性解码，检查是否能回到模板渲染文本。
    decoded = tokenizer.decode(
        direct_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(decoded, str):
        raise RuntimeError("tokenizer.decode did not return text")

    # 本地目录方便离线运行，但目录路径本身不能证明文件确实来自目标 commit。
    local_snapshot = source_kind == "local_snapshot"
    source_explanation = (
        "从本地目录读取 tokenizer。报告会保留本实验期望的模型版本，"
        "但仅凭这个目录不能证明文件来自该 commit。"
        if local_snapshot
        else "按模型仓库和完整 commit 加载 tokenizer；可以使用本地缓存，"
        "但不会跟随 branch 或 tag 变化。"
    )
    return {
        "schema": REPORT_SCHEMA,
        "model": {
            "target_repository": MODEL_REPOSITORY,
            "target_revision": MODEL_REVISION,
            "load_source": load_source,
            "source_kind": source_kind,
            "source_explanation": source_explanation,
            "tokenizer_class": type(tokenizer).__name__,
        },
        "runtime": {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
        },
        "request": {
            "messages": messages,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        },
        "rendered_prompt": rendered,
        "tokenization": {
            "token_count": len(direct_ids),
            "token_ids": direct_ids,
            "tokens": token_rows,
            "template_ids_match_rendered_encoding": direct_ids == rendered_ids,
        },
        "tokenizer_metadata": {
            "special_tokens_map": _serializable_special_tokens(
                getattr(tokenizer, "special_tokens_map", {})
            ),
            "added_token_count": len(raw_added_vocab),
        },
        "round_trip": {
            "decoded_with_special_tokens": decoded,
            "matches_rendered_prompt": decoded == rendered,
        },
        "scope": {
            "qwen3_tokenizer_and_chat_template_executed": True,
            "model_weights_loaded": False,
            "nano_vllm_executed": False,
            "gpu_required": False,
        },
    }


def load_tokenizer(
    *,
    model_snapshot: Path | None,
    local_files_only: bool,
) -> tuple[Any, str, str]:
    """从本地快照或固定仓库 revision 加载 tokenizer。"""

    if model_snapshot is not None:
        # 显式目录总是离线读取；调用方需要自行保证其来源可信。
        resolved = model_snapshot.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"model snapshot directory does not exist: {resolved}")
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            str(resolved),
            local_files_only=True,
            trust_remote_code=False,
        )
        return tokenizer, str(resolved), "local_snapshot"

    # 仓库模式固定完整 commit，可选择只使用已存在的 Hugging Face 缓存。
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    return tokenizer, MODEL_REPOSITORY, "repository_revision"


def _visible_token(token: str) -> str:
    """把换行和制表符转义为可在一行表格中看见的文本。"""

    return token.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def format_human_readable(report: Mapping[str, Any]) -> str:
    """把结构化 trace 排成四步中文导览。"""

    model = report["model"]
    request = report["request"]
    runtime = report["runtime"]
    tokenization = report["tokenization"]
    round_trip = report["round_trip"]
    assert isinstance(model, Mapping)
    assert isinstance(request, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(tokenization, Mapping)
    assert isinstance(round_trip, Mapping)
    messages = request["messages"]
    assert isinstance(messages, list) and isinstance(messages[0], Mapping)
    rows = tokenization["tokens"]
    assert isinstance(rows, list)

    # 展示顺序与真实调用链一致：message → template → IDs → 一致性检查。
    lines = [
        "Qwen3 对话分词追踪",
        f"目标模型：{model['target_repository']}@{model['target_revision']}",
        f"加载位置：{model['load_source']}",
        f"Transformers：{runtime['transformers']}",
        f"说明：{model['source_explanation']}",
        "",
        "1. 应用传入的 user message",
        str(messages[0]["content"]),
        "",
        "2. Qwen3 chat template 组成的完整提示词",
        "----- rendered prompt -----",
        str(report["rendered_prompt"]),
        "----- end rendered prompt -----",
        "",
        f"3. Token IDs（共 {tokenization['token_count']} 个）",
        "位置\tID\ttokenizer 类型\t可读片段\t词表中的 token",
    ]
    for row in rows:
        assert isinstance(row, Mapping)
        lines.append(
            f"{row['position']}\t{row['token_id']}\t{row['tokenizer_kind']}\t"
            f"{_visible_token(str(row['decoded_piece']))}\t"
            f"{_visible_token(str(row['vocabulary_token']))}"
        )
    lines.extend(
        [
            "",
            "4. 两个容易混淆的检查",
            "模板直接生成的 IDs 与渲染后再次编码："
            + (
                "一致"
                if tokenization["template_ids_match_rendered_encoding"]
                else "不一致"
            ),
            "保留特殊 token 解码后与渲染文本："
            + ("一致" if round_trip["matches_rendered_prompt"] else "不完全一致"),
            "表中的 special 来自 tokenizer 的 special-token 元数据；added 表示保留的词表项。"
            "像 <think> 这样的模板控制词可能属于 added，而不是 special。",
            "",
            "这次只运行了 tokenizer 和 chat template，没有加载模型权重，"
            "也没有调用 GPU 或 nano-vLLM。",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """定义消息、加载来源、thinking 模板开关和输出格式。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        help="local Qwen3-0.6B snapshot directory; no network is used",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="load the fixed repository revision from the Hugging Face cache only",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="let the Qwen3 chat template open a thinking block",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of the walkthrough",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """加载目标 tokenizer 并打印人类导览或完整 JSON。"""

    # token 片段包含中文和特殊符号，统一为 UTF-8 输出。
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    tokenizer, load_source, source_kind = load_tokenizer(
        model_snapshot=args.model_snapshot,
        local_files_only=args.local_files_only,
    )
    report = trace_chat_message(
        tokenizer,
        message=args.message,
        load_source=load_source,
        source_kind=source_kind,
        enable_thinking=args.enable_thinking,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_human_readable(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
