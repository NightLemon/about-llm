from __future__ import annotations

from datetime import datetime, timezone

import pytest

from about_llm.integrations.retry import RetryPolicy, decide_retry, parse_retry_after

pytestmark = pytest.mark.contract

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_retry_after_delta_and_http_date_override_local_backoff() -> None:
    policy = RetryPolicy(base_delay_seconds=8, max_retry_after_seconds=20)

    delta = decide_retry(
        policy=policy,
        attempt=1,
        replay_safe=True,
        outcome_uncertain=False,
        status_code=429,
        response_headers={"Retry-After": "3"},
        now=NOW,
        jitter_fraction=0,
    )
    date = decide_retry(
        policy=policy,
        attempt=1,
        replay_safe=True,
        outcome_uncertain=False,
        status_code=503,
        response_headers={"retry-after": "Wed, 01 Jan 2025 00:00:05 GMT"},
        now=NOW,
    )

    assert delta.retry and delta.delay_seconds == 3
    assert delta.retry_after_source == "delta-seconds"
    assert date.retry and date.delay_seconds == 5
    assert date.retry_after_source == "http-date"


@pytest.mark.parametrize("value", ["-1", "1.5", " 2", "soon", ""])
def test_malformed_retry_after_is_not_silently_coerced(value: str) -> None:
    assert parse_retry_after(value, now=NOW) is None


def test_past_retry_after_date_means_zero_delay() -> None:
    parsed = parse_retry_after("Tue, 31 Dec 2024 23:59:59 GMT", now=NOW)
    assert parsed is not None and parsed.delay_seconds == 0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 501, 505])
def test_non_allowlisted_http_statuses_are_not_retried(status: int) -> None:
    decision = decide_retry(
        policy=RetryPolicy(),
        attempt=1,
        replay_safe=True,
        outcome_uncertain=False,
        status_code=status,
    )
    assert not decision.retry and decision.reason == "not_retryable"


def test_replay_and_uncertain_outcome_guards_fail_closed() -> None:
    unsafe = decide_retry(
        policy=RetryPolicy(),
        attempt=1,
        replay_safe=False,
        outcome_uncertain=False,
        status_code=429,
    )
    uncertain = decide_retry(
        policy=RetryPolicy(),
        attempt=1,
        replay_safe=True,
        outcome_uncertain=True,
        error_category="timeout",
    )
    assert not unsafe.retry and unsafe.reason == "replay_unsafe"
    assert not uncertain.retry and uncertain.reason == "outcome_uncertain"


def test_attempt_retry_after_and_deadline_budgets_stop_retries() -> None:
    exhausted = decide_retry(
        policy=RetryPolicy(max_attempts=2),
        attempt=2,
        replay_safe=True,
        outcome_uncertain=False,
        status_code=500,
    )
    too_long = decide_retry(
        policy=RetryPolicy(max_retry_after_seconds=4),
        attempt=1,
        replay_safe=True,
        outcome_uncertain=False,
        status_code=429,
        response_headers={"Retry-After": "5"},
        now=NOW,
    )
    deadline = decide_retry(
        policy=RetryPolicy(base_delay_seconds=2),
        attempt=1,
        replay_safe=True,
        outcome_uncertain=False,
        status_code=502,
        remaining_seconds=2,
    )
    assert exhausted.reason == "attempts_exhausted"
    assert too_long.reason == "retry_after_too_long"
    assert deadline.reason == "deadline_exhausted"
    assert not exhausted.retry and not too_long.retry and not deadline.retry


def test_injected_jitter_is_deterministic_and_bounded() -> None:
    policy = RetryPolicy(
        max_attempts=4,
        base_delay_seconds=2,
        multiplier=2,
        max_backoff_seconds=10,
    )
    low = decide_retry(
        policy=policy,
        attempt=3,
        replay_safe=True,
        outcome_uncertain=False,
        error_category="transport",
        jitter_fraction=0.25,
    )
    high = decide_retry(
        policy=policy,
        attempt=3,
        replay_safe=True,
        outcome_uncertain=False,
        error_category="transport",
        jitter_fraction=1,
    )
    assert low.delay_seconds == 2
    assert high.delay_seconds == 8
    with pytest.raises(ValueError, match="between 0 and 1"):
        decide_retry(
            policy=policy,
            attempt=1,
            replay_safe=True,
            outcome_uncertain=False,
            status_code=500,
            jitter_fraction=1.1,
        )


def test_malformed_retry_after_falls_back_to_local_policy() -> None:
    decision = decide_retry(
        policy=RetryPolicy(base_delay_seconds=2),
        attempt=1,
        replay_safe=True,
        outcome_uncertain=False,
        status_code=503,
        response_headers={"Retry-After": "not-a-date"},
        now=NOW,
        jitter_fraction=0.5,
    )
    assert decision.retry and decision.delay_seconds == 1
    assert decision.retry_after_state == "malformed"
    assert decision.retry_after_source is None


def test_retry_inputs_reject_boolean_coercion_and_ambiguous_headers() -> None:
    with pytest.raises(ValueError, match="must be booleans"):
        decide_retry(
            policy=RetryPolicy(),
            attempt=1,
            replay_safe=1,  # type: ignore[arg-type]
            outcome_uncertain=False,
            status_code=500,
        )
    with pytest.raises(ValueError, match="duplicate case-insensitive"):
        decide_retry(
            policy=RetryPolicy(),
            attempt=1,
            replay_safe=True,
            outcome_uncertain=False,
            status_code=429,
            response_headers={"Retry-After": "1", "retry-after": "2"},
            now=NOW,
        )
