"""Bounded incremental UTF-8 stop-string matcher for streamed text bytes."""

from __future__ import annotations

import codecs
from collections.abc import Sequence
from dataclasses import dataclass


class StopMatcherStateError(RuntimeError):
    """Raised when bytes are fed after stop or end-of-stream termination."""


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class StopMatchUpdate:
    emitted_text: str
    stopped: bool
    matched_stop: str | None
    terminal: bool
    decoded_characters: int
    emitted_characters: int
    held_characters: int
    discarded_after_stop_characters: int


@dataclass(frozen=True)
class StopMatcherReport:
    stop_strings: tuple[str, ...]
    include_stop: bool
    stopped: bool
    eof_finished: bool
    terminal: bool
    matched_stop: str | None
    decoded_characters: int
    emitted_characters: int
    held_characters: int
    discarded_after_stop_characters: int
    buffered_utf8_bytes: int


class IncrementalStopMatcher:
    """Match exact Unicode stop strings without leaking a partial stop prefix.

    Input is an arbitrary sequence of byte chunks from one UTF-8 text stream.
    Bytes are decoded strictly and incrementally. Decoded characters are then
    processed one at a time, making the result independent of byte chunking.
    The matcher withholds the longest suffix that is a prefix of any configured
    stop string. A stop fires at the first decoded character where one or more
    strings complete; if several complete at that same character, configuration
    order selects the winner. Matching is exact, case-sensitive, and performs no
    Unicode normalization.

    This class is a transport/text state-machine reference. It does not decode
    model token ids, define provider usage, or prove that server cancellation
    stops GPU work or billing.
    """

    def __init__(
        self,
        stop_strings: Sequence[str],
        *,
        include_stop: bool = False,
        max_stop_strings: int = 32,
        max_stop_characters: int = 256,
    ) -> None:
        max_stop_strings = _positive_integer(max_stop_strings, "max_stop_strings")
        max_stop_characters = _positive_integer(
            max_stop_characters, "max_stop_characters"
        )
        if isinstance(stop_strings, (str, bytes)) or not isinstance(
            stop_strings, Sequence
        ):
            raise TypeError("stop_strings must be a sequence of strings")
        stops = tuple(stop_strings)
        if not stops:
            raise ValueError("stop_strings cannot be empty")
        if len(stops) > max_stop_strings:
            raise ValueError("stop_strings exceeds max_stop_strings")
        if any(not isinstance(stop, str) for stop in stops):
            raise TypeError("every stop string must be a string")
        if any(not stop for stop in stops):
            raise ValueError("stop strings cannot be empty")
        if len(stops) != len(set(stops)):
            raise ValueError("stop strings must be unique")
        if any(len(stop) > max_stop_characters for stop in stops):
            raise ValueError("a stop string exceeds max_stop_characters")
        for stop in stops:
            try:
                stop.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise ValueError("stop strings must be valid Unicode scalar text") from error
        if not isinstance(include_stop, bool):
            raise TypeError("include_stop must be a boolean")

        decoder_type = codecs.getincrementaldecoder("utf-8")
        self._decoder: codecs.IncrementalDecoder = decoder_type(errors="strict")
        self.stop_strings = stops
        self.include_stop = include_stop
        self.max_stop_strings = max_stop_strings
        self.max_stop_characters = max_stop_characters
        self._pending = ""
        self._stopped = False
        self._eof_finished = False
        self._matched_stop: str | None = None
        self._decoded_characters = 0
        self._emitted_characters = 0
        self._discarded_after_stop_characters = 0

    def feed(self, chunk: bytes | bytearray | memoryview) -> StopMatchUpdate:
        """Consume one arbitrary UTF-8 byte chunk and return newly safe text."""

        if self._stopped or self._eof_finished:
            raise StopMatcherStateError("cannot feed bytes after matcher termination")
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("chunk must be bytes-like")
        decoder_state = self._decoder.getstate()
        try:
            decoded = self._decoder.decode(bytes(chunk), final=False)
        except UnicodeDecodeError:
            self._decoder.setstate(decoder_state)
            raise
        emitted = self._process_decoded(decoded)
        return self._update(emitted)

    def finish(self) -> StopMatchUpdate:
        """Declare EOF, validate UTF-8 completeness, and flush safe pending text."""

        if self._stopped or self._eof_finished:
            return self._update("")
        decoder_state = self._decoder.getstate()
        try:
            decoded = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            self._decoder.setstate(decoder_state)
            raise
        emitted = self._process_decoded(decoded)
        if not self._stopped:
            emitted += self._pending
            self._emitted_characters += len(self._pending)
            self._pending = ""
            self._eof_finished = True
        return self._update(emitted)

    def report(self) -> StopMatcherReport:
        buffered, _ = self._decoder.getstate()
        return StopMatcherReport(
            stop_strings=self.stop_strings,
            include_stop=self.include_stop,
            stopped=self._stopped,
            eof_finished=self._eof_finished,
            terminal=self._stopped or self._eof_finished,
            matched_stop=self._matched_stop,
            decoded_characters=self._decoded_characters,
            emitted_characters=self._emitted_characters,
            held_characters=len(self._pending),
            discarded_after_stop_characters=self._discarded_after_stop_characters,
            buffered_utf8_bytes=len(buffered),
        )

    def _process_decoded(self, text: str) -> str:
        self._decoded_characters += len(text)
        emitted_parts: list[str] = []
        for index, character in enumerate(text):
            self._pending += character
            matches = [
                stop for stop in self.stop_strings if self._pending.endswith(stop)
            ]
            if matches:
                matched = matches[0]
                prefix = self._pending[: -len(matched)]
                emitted_parts.append(prefix)
                if self.include_stop:
                    emitted_parts.append(matched)
                newly_emitted = len(prefix) + (
                    len(matched) if self.include_stop else 0
                )
                self._emitted_characters += newly_emitted
                self._discarded_after_stop_characters += len(text) - index - 1
                self._pending = ""
                self._matched_stop = matched
                self._stopped = True
                break

            held = self._longest_partial_stop_suffix()
            safe_characters = len(self._pending) - held
            if safe_characters:
                safe = self._pending[:safe_characters]
                emitted_parts.append(safe)
                self._emitted_characters += len(safe)
                self._pending = self._pending[safe_characters:]

        if not self._stopped and len(self._pending) >= self.max_stop_characters:
            raise RuntimeError("pending stop prefix exceeded configured bound")
        return "".join(emitted_parts)

    def _longest_partial_stop_suffix(self) -> int:
        longest = 0
        for stop in self.stop_strings:
            maximum = min(len(self._pending), len(stop) - 1)
            for length in range(maximum, longest, -1):
                if self._pending.endswith(stop[:length]):
                    longest = length
                    break
        return longest

    def _update(self, emitted_text: str) -> StopMatchUpdate:
        return StopMatchUpdate(
            emitted_text=emitted_text,
            stopped=self._stopped,
            matched_stop=self._matched_stop,
            terminal=self._stopped or self._eof_finished,
            decoded_characters=self._decoded_characters,
            emitted_characters=self._emitted_characters,
            held_characters=len(self._pending),
            discarded_after_stop_characters=self._discarded_after_stop_characters,
        )
