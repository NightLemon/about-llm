from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from about_llm.inference import (
    IncrementalStopMatcher,
    StopMatcherStateError,
)

ROOT = Path(__file__).resolve().parents[1]


def test_utf8_and_stop_can_both_cross_arbitrary_byte_chunks_without_leakage() -> None:
    payload = "甲🙂乙<END>尾".encode()
    matcher = IncrementalStopMatcher(("<END>",))
    emitted: list[str] = []
    for byte in payload:
        update = matcher.feed(bytes([byte]))
        emitted.append(update.emitted_text)
        if update.stopped:
            break

    assert "".join(emitted) == "甲🙂乙"
    report = matcher.report()
    assert report.stopped
    assert report.matched_stop == "<END>"
    assert report.held_characters == 0
    assert report.buffered_utf8_bytes == 0


def test_partial_stop_prefix_is_withheld_until_match_or_disproof() -> None:
    matcher = IncrementalStopMatcher(("END",))

    first = matcher.feed(b"hello EN")
    second = matcher.feed(b"X")
    final = matcher.finish()

    assert first.emitted_text == "hello "
    assert first.held_characters == 2
    assert second.emitted_text == "ENX"
    assert second.held_characters == 0
    assert final.emitted_text == ""
    assert final.terminal


def test_completed_stop_is_excluded_by_default_and_trailing_text_is_discarded() -> None:
    matcher = IncrementalStopMatcher(("END",))
    update = matcher.feed(b"answerENDignored")

    assert update.emitted_text == "answer"
    assert update.stopped
    assert update.matched_stop == "END"
    assert update.discarded_after_stop_characters == len("ignored")
    assert matcher.report().decoded_characters == len("answerENDignored")


def test_include_stop_returns_the_selected_delimiter() -> None:
    matcher = IncrementalStopMatcher(("END",), include_stop=True)
    update = matcher.feed(b"answerENDignored")

    assert update.emitted_text == "answerEND"
    assert matcher.report().emitted_characters == len("answerEND")


def test_eof_flushes_an_unfinished_stop_prefix_and_is_idempotent() -> None:
    matcher = IncrementalStopMatcher(("END",))
    first = matcher.feed(b"value EN")
    finished = matcher.finish()
    repeated = matcher.finish()

    assert first.emitted_text == "value "
    assert finished.emitted_text == "EN"
    assert finished.terminal
    assert matcher.report().eof_finished
    assert repeated.emitted_text == ""


def test_same_character_completion_uses_configured_order() -> None:
    first = IncrementalStopMatcher(("BC", "ABC"))
    second = IncrementalStopMatcher(("ABC", "BC"))

    first_update = first.feed(b"ABCZ")
    second_update = second.feed(b"ABCZ")

    assert first_update.matched_stop == "BC"
    assert first_update.emitted_text == "A"
    assert second_update.matched_stop == "ABC"
    assert second_update.emitted_text == ""
    assert first_update.discarded_after_stop_characters == 1
    assert second_update.discarded_after_stop_characters == 1


def test_first_character_completion_is_independent_of_input_chunking() -> None:
    payload = b"prefix ABC suffix"
    whole = IncrementalStopMatcher(("BC", "ABC"))
    split = IncrementalStopMatcher(("BC", "ABC"))

    whole_text = whole.feed(payload).emitted_text
    emitted: list[str] = []
    for chunk in (payload[:3], payload[3:9], payload[9:]):
        update = split.feed(chunk)
        emitted.append(update.emitted_text)
        if update.stopped:
            break
    split_text = "".join(emitted)

    assert whole_text == split_text == "prefix A"
    assert whole.report().matched_stop == split.report().matched_stop == "BC"


def test_shorter_stop_that_completes_first_wins_without_waiting_for_longer_prefix() -> None:
    matcher = IncrementalStopMatcher(("END", "E"))
    update = matcher.feed(b"END")

    assert update.matched_stop == "E"
    assert update.emitted_text == ""
    assert update.discarded_after_stop_characters == 2


def test_matching_is_case_sensitive_and_does_not_normalize_unicode() -> None:
    matcher = IncrementalStopMatcher(("É",))
    first = matcher.feed("e\u0301é".encode())
    final = matcher.finish()

    assert not first.stopped
    assert first.emitted_text == "e\u0301é"
    assert final.emitted_text == ""


def test_invalid_utf8_feed_is_atomic() -> None:
    matcher = IncrementalStopMatcher(("END",))
    before = matcher.report()

    with pytest.raises(UnicodeDecodeError):
        matcher.feed(b"\xff")

    assert matcher.report() == before
    assert matcher.feed(b"ok").emitted_text == "ok"


def test_truncated_utf8_finish_is_atomic_and_can_receive_continuation() -> None:
    matcher = IncrementalStopMatcher(("END",))
    first = matcher.feed(b"\xf0")
    before = matcher.report()

    with pytest.raises(UnicodeDecodeError):
        matcher.finish()

    assert first.emitted_text == ""
    assert matcher.report() == before
    continued = matcher.feed(b"\x9f\x99\x82")
    assert continued.emitted_text == "🙂"
    assert matcher.finish().terminal


def test_feed_after_stop_or_eof_fails_closed() -> None:
    stopped = IncrementalStopMatcher(("X",))
    stopped.feed(b"X")
    with pytest.raises(StopMatcherStateError, match="termination"):
        stopped.feed(b"more")

    finished = IncrementalStopMatcher(("X",))
    finished.finish()
    with pytest.raises(StopMatcherStateError, match="termination"):
        finished.feed(b"more")


@pytest.mark.parametrize(
    ("operation", "error_type", "message"),
    [
        (lambda: IncrementalStopMatcher(()), ValueError, "cannot be empty"),
        (lambda: IncrementalStopMatcher(("",)), ValueError, "cannot be empty"),
        (lambda: IncrementalStopMatcher(("X", "X")), ValueError, "unique"),
        (
            lambda: IncrementalStopMatcher(("X", "Y"), max_stop_strings=1),
            ValueError,
            "max_stop_strings",
        ),
        (
            lambda: IncrementalStopMatcher(("XX",), max_stop_characters=1),
            ValueError,
            "max_stop_characters",
        ),
        (
            lambda: IncrementalStopMatcher(("\ud800",)),
            ValueError,
            "valid Unicode scalar",
        ),
        (
            lambda: IncrementalStopMatcher(("X",), include_stop=1),
            TypeError,
            "boolean",
        ),
        (
            lambda: IncrementalStopMatcher(("X",)).feed("text"),
            TypeError,
            "bytes-like",
        ),
    ],
)
def test_invalid_stop_contracts_fail_closed(
    operation: Callable[[], object], error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        operation()


