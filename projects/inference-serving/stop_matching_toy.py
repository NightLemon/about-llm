"""演示流式 stop string 在 UTF-8 分块、部分匹配和重叠规则下的行为。

文本被故意切在 emoji 多字节序列和 ``<END>`` 内部。matcher 必须暂存尚未完成的字符与
潜在 stop 前缀，只输出确认安全的文本；第二组再比较同一位置完成的长短 stop。
"""

from __future__ import annotations

import json
import sys

from about_llm.inference import IncrementalStopMatcher


def _report_dict(matcher: IncrementalStopMatcher) -> dict[str, object]:
    """把 matcher 内部累计状态转换成便于查看的字典。"""

    report = matcher.report()
    return {
        "stop_strings": report.stop_strings,
        "include_stop": report.include_stop,
        "stopped": report.stopped,
        "eof_finished": report.eof_finished,
        "terminal": report.terminal,
        "matched_stop": report.matched_stop,
        "decoded_characters": report.decoded_characters,
        "emitted_characters": report.emitted_characters,
        "held_characters": report.held_characters,
        "discarded_after_stop_characters": (
            report.discarded_after_stop_characters
        ),
        "buffered_utf8_bytes": report.buffered_utf8_bytes,
    }


def main() -> None:
    """逐字节块喂入中文、emoji 和 stop，并检查重叠匹配。"""

    # 严格 UTF-8 输出让无效字节不会被静默替换，便于发现流式解码错误。
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="strict")
    # chunks 的边界有意与 Unicode 字符和 stop string 边界错开。
    payload = "甲🙂乙<END>尾".encode()
    chunks = (payload[:4], payload[4:7], payload[7:12], payload[12:])
    matcher = IncrementalStopMatcher(("<END>", "STOP"))
    updates: list[dict[str, object]] = []
    emitted: list[str] = []
    # 每次 feed 只返回已经确定不可能属于 stop 的安全文本。
    for chunk in chunks:
        update = matcher.feed(chunk)
        emitted.append(update.emitted_text)
        updates.append(
            {
                "chunk_hex": chunk.hex(),
                "emitted_text": update.emitted_text,
                "held_characters": update.held_characters,
                "stopped": update.stopped,
            }
        )
        if update.stopped:
            break

    # 当 BC 与 ABC 在同一字符位置结束时，matcher 应按明确规则选择更长的 ABC。
    overlap = IncrementalStopMatcher(("BC", "ABC"))
    overlap_update = overlap.feed(b"ABCZ")
    artifact = {
        "utf8_split_fixture": {
            "updates": updates,
            "returned_text": "".join(emitted),
            "report": _report_dict(matcher),
        },
        "same_completion_overlap_fixture": {
            "configured_stops": overlap.stop_strings,
            "returned_text": overlap_update.emitted_text,
            "matched_stop": overlap_update.matched_stop,
            "discarded_after_stop_characters": (
                overlap_update.discarded_after_stop_characters
            ),
        },
        "scope": {
            "strict_incremental_utf8_decoding_executed": True,
            "partial_stop_withholding_executed": True,
            "byte_chunk_independent_character_matching_executed": True,
            "tokenizer_or_model_token_ids_decoded": False,
            "provider_usage_or_finish_reason_equivalence_proved": False,
            "server_cancellation_gpu_release_or_billing_proved": False,
            "unicode_normalization_or_case_folding_performed": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
