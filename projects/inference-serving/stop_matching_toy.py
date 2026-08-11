"""Exercise UTF-8 byte boundaries, partial stops, and overlap semantics."""

from __future__ import annotations

import json
import sys

from about_llm.inference import IncrementalStopMatcher


def _report_dict(matcher: IncrementalStopMatcher) -> dict[str, object]:
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
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="strict")
    payload = "甲🙂乙<END>尾".encode()
    chunks = (payload[:4], payload[4:7], payload[7:12], payload[12:])
    matcher = IncrementalStopMatcher(("<END>", "STOP"))
    updates: list[dict[str, object]] = []
    emitted: list[str] = []
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
