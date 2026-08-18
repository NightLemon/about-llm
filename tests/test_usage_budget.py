from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier

import pytest

from about_llm.integrations.cloud_api import (
    ChatMessage,
    RequestSpec,
    build_anthropic_request,
    build_gemini_request,
    build_openai_compatible_request,
)
from about_llm.integrations.usage_budget import (
    PostCallBudgetExceededError,
    TokenPricingSnapshot,
    UsageBudgetExceededError,
    UsageBudgetLedger,
    UsageBudgetLimits,
    UsageBudgetStateError,
    cloud_request_budget_fingerprint,
    request_maximum_output_tokens,
    request_model_identifier,
)

pytestmark = [pytest.mark.formula, pytest.mark.contract, pytest.mark.security]

_REQUEST_FINGERPRINT = "sha256:" + "a" * 64


def _pricing() -> TokenPricingSnapshot:
    return TokenPricingSnapshot(
        pricing_id="provider/model@price-2026-08-01",
        provider="provider",
        model="model",
        revision="price-2026-08-01",
        checked_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        input_microusd_per_million=1_000_000,
        output_microusd_per_million=2_000_000,
    )


def _ledger(**overrides: int | None) -> UsageBudgetLedger:
    values: dict[str, int | None] = {
        "max_input_tokens": 100,
        "max_output_tokens": 20,
        "max_estimated_microusd": 140,
    }
    values.update(overrides)
    return UsageBudgetLedger(limits=UsageBudgetLimits(**values), pricing=_pricing())


def test_pricing_quote_uses_integer_math_and_one_conservative_rounding() -> None:
    pricing = TokenPricingSnapshot(
        pricing_id="price-v1",
        provider="provider",
        model="model",
        revision="v1",
        checked_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        input_microusd_per_million=333_333,
        output_microusd_per_million=666_667,
    )

    assert pricing.quote_microusd(1, 0) == 1
    assert pricing.quote_microusd(1, 1) == 1
    assert pricing.quote_microusd(3, 0) == 1
    assert pricing.quote_microusd(0, 0) == 0


def test_reservation_holds_worst_case_capacity_and_settlement_refunds_it() -> None:
    ledger = _ledger()

    receipt = ledger.reserve(
        "call-1",
        request_fingerprint=_REQUEST_FINGERPRINT,
        estimated_input_tokens=60,
        maximum_output_tokens=10,
    )
    active = ledger.snapshot()
    assert receipt.maximum_estimated_microusd == 80
    assert active.reserved_input_tokens == 60
    assert active.reserved_output_tokens == 10
    assert active.reserved_estimated_microusd == 80

    settled = ledger.settle(
        "call-1",
        request_fingerprint=receipt.request_fingerprint,
        actual_input_tokens=58,
        actual_output_tokens=4,
    )
    assert settled.committed_input_tokens == 58
    assert settled.committed_output_tokens == 4
    assert settled.committed_estimated_microusd == 66
    assert settled.pricing_id == "provider/model@price-2026-08-01"
    assert settled.max_estimated_microusd == 140
    assert settled.reserved_estimated_microusd == 0
    ledger.reserve(
        "call-2",
        request_fingerprint=_REQUEST_FINGERPRINT,
        estimated_input_tokens=40,
        maximum_output_tokens=10,
    )


def test_active_reservations_fail_closed_before_transport() -> None:
    ledger = _ledger(max_estimated_microusd=100)
    ledger.reserve(
        "call-1",
        request_fingerprint=_REQUEST_FINGERPRINT,
        estimated_input_tokens=60,
        maximum_output_tokens=10,
    )

    with pytest.raises(UsageBudgetExceededError):
        ledger.reserve(
            "call-2",
            request_fingerprint=_REQUEST_FINGERPRINT,
            estimated_input_tokens=10,
            maximum_output_tokens=6,
        )

    snapshot = ledger.snapshot()
    assert snapshot.active_reservations == 1
    assert snapshot.over_limit is False


def test_unknown_usage_consumes_full_reservation_and_cannot_be_released() -> None:
    ledger = _ledger()
    ledger.reserve(
        "call-1",
        request_fingerprint=_REQUEST_FINGERPRINT,
        estimated_input_tokens=60,
        maximum_output_tokens=10,
    )

    snapshot = ledger.mark_usage_uncertain(
        "call-1", request_fingerprint=_REQUEST_FINGERPRINT
    )

    assert snapshot.committed_input_tokens == 60
    assert snapshot.committed_output_tokens == 10
    assert snapshot.committed_estimated_microusd == 80
    assert snapshot.uncertain_settlements == 1
    with pytest.raises(UsageBudgetStateError):
        ledger.cancel_before_send(
            "call-1", request_fingerprint=_REQUEST_FINGERPRINT
        )


def test_cancel_before_send_releases_capacity_but_id_is_not_reusable() -> None:
    ledger = _ledger()
    ledger.reserve(
        "call-1",
        request_fingerprint=_REQUEST_FINGERPRINT,
        estimated_input_tokens=100,
        maximum_output_tokens=20,
    )

    cancelled = ledger.cancel_before_send(
        "call-1", request_fingerprint=_REQUEST_FINGERPRINT
    )

    assert cancelled.active_reservations == 0
    assert cancelled.committed_input_tokens == 0
    with pytest.raises(UsageBudgetStateError, match="already been used"):
        ledger.reserve(
            "call-1",
            request_fingerprint=_REQUEST_FINGERPRINT,
            estimated_input_tokens=1,
            maximum_output_tokens=1,
        )


def test_reported_usage_overrun_is_committed_then_trips_post_call_breach() -> None:
    ledger = _ledger(max_input_tokens=70, max_output_tokens=20)
    ledger.reserve(
        "call-1",
        request_fingerprint=_REQUEST_FINGERPRINT,
        estimated_input_tokens=60,
        maximum_output_tokens=10,
    )

    with pytest.raises(PostCallBudgetExceededError) as captured:
        ledger.settle(
            "call-1",
            request_fingerprint=_REQUEST_FINGERPRINT,
            actual_input_tokens=71,
            actual_output_tokens=4,
        )

    assert captured.value.snapshot.committed_input_tokens == 71
    assert captured.value.snapshot.over_limit is True
    with pytest.raises(UsageBudgetExceededError):
        ledger.reserve(
            "call-2",
            request_fingerprint=_REQUEST_FINGERPRINT,
            estimated_input_tokens=0,
            maximum_output_tokens=0,
        )


def test_two_concurrent_reservations_cannot_both_spend_same_capacity() -> None:
    ledger = _ledger(
        max_input_tokens=10,
        max_output_tokens=None,
        max_estimated_microusd=None,
    )
    barrier = Barrier(2)

    def attempt(name: str) -> str:
        barrier.wait()
        try:
            ledger.reserve(
                name,
                request_fingerprint=_REQUEST_FINGERPRINT,
                estimated_input_tokens=10,
                maximum_output_tokens=0,
            )
        except UsageBudgetExceededError:
            return "rejected"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, ("call-1", "call-2")))

    assert sorted(outcomes) == ["rejected", "reserved"]
    assert ledger.snapshot().reserved_input_tokens == 10


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_microusd_per_million", True),
        ("output_microusd_per_million", -1),
    ],
)
def test_pricing_rejects_invalid_rates(field: str, value: int) -> None:
    values = {
        "pricing_id": "price-v1",
        "provider": "provider",
        "model": "model",
        "revision": "v1",
        "checked_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
        "input_microusd_per_million": 1,
        "output_microusd_per_million": 1,
    }
    values[field] = value

    with pytest.raises(ValueError):
        TokenPricingSnapshot(**values)  # type: ignore[arg-type]


def test_pricing_requires_timezone_aware_check_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        TokenPricingSnapshot(
            pricing_id="price-v1",
            provider="provider",
            model="model",
            revision="v1",
            checked_at=datetime(2026, 8, 1),
            input_microusd_per_million=1,
            output_microusd_per_million=1,
        )


def test_limits_require_at_least_one_strict_non_negative_cap() -> None:
    with pytest.raises(ValueError, match="at least one"):
        UsageBudgetLimits()
    with pytest.raises(ValueError, match="max_input_tokens"):
        UsageBudgetLimits(max_input_tokens=True)


def _requests(api_key: str = "secret-a") -> tuple[RequestSpec, ...]:
    messages = [ChatMessage("user", "hello")]
    return (
        build_openai_compatible_request(
            base_url="https://openai.example",
            api_key=api_key,
            model="model",
            messages=messages,
            max_tokens=12,
        ),
        build_anthropic_request(
            base_url="https://anthropic.example",
            api_key=api_key,
            api_version="2023-06-01",
            model="model",
            messages=messages,
            max_tokens=12,
        ),
        build_gemini_request(
            base_url="https://gemini.example",
            api_key=api_key,
            model="model",
            messages=messages,
            max_tokens=12,
        ),
    )


def test_supported_requests_bind_exact_output_cap_and_hide_credentials() -> None:
    first = _requests("secret-a")
    rotated = _requests("secret-b")

    for request, rotated_request in zip(first, rotated, strict=True):
        fingerprint = cloud_request_budget_fingerprint(
            request, billing_scope="account/project-a"
        )
        assert request_maximum_output_tokens(request) == 12
        assert request_model_identifier(request) == "model"
        assert fingerprint == cloud_request_budget_fingerprint(
            rotated_request, billing_scope="account/project-a"
        )
        assert "secret-a" not in fingerprint
        assert len(fingerprint) == 71
        assert fingerprint != cloud_request_budget_fingerprint(
            request, billing_scope="account/project-b"
        )


def test_request_semantic_or_cap_drift_changes_budget_fingerprint() -> None:
    original = build_openai_compatible_request(
        base_url="https://openai.example",
        api_key="secret",
        model="model",
        messages=[ChatMessage("user", "hello")],
        max_tokens=12,
    )
    changed_prompt = build_openai_compatible_request(
        base_url="https://openai.example",
        api_key="secret",
        model="model",
        messages=[ChatMessage("user", "changed")],
        max_tokens=12,
    )
    changed_cap = build_openai_compatible_request(
        base_url="https://openai.example",
        api_key="secret",
        model="model",
        messages=[ChatMessage("user", "hello")],
        max_tokens=13,
    )
    identity = cloud_request_budget_fingerprint(
        original, billing_scope="account/project-a"
    )

    assert identity != cloud_request_budget_fingerprint(
        changed_prompt, billing_scope="account/project-a"
    )
    assert identity != cloud_request_budget_fingerprint(
        changed_cap, billing_scope="account/project-a"
    )


def test_reserve_request_derives_cap_and_stores_request_identity() -> None:
    request = _requests()[0]
    ledger = _ledger()

    receipt = ledger.reserve_request(
        "call-1",
        request=request,
        billing_scope="account/project-a",
        estimated_input_tokens=60,
    )

    assert receipt.maximum_output_tokens == 12
    assert receipt.request_fingerprint == cloud_request_budget_fingerprint(
        request, billing_scope="account/project-a"
    )


def test_reserve_request_rejects_model_pricing_mismatch_before_mutation() -> None:
    request = build_openai_compatible_request(
        base_url="https://openai.example",
        api_key="secret",
        model="different-model",
        messages=[ChatMessage("user", "hello")],
        max_tokens=12,
    )
    ledger = _ledger()

    with pytest.raises(ValueError, match="pricing snapshot model"):
        ledger.reserve_request(
            "call-1",
            request=request,
            billing_scope="account/project-a",
            estimated_input_tokens=1,
        )

    assert ledger.snapshot().active_reservations == 0


def test_request_fingerprint_normalizes_header_order_and_name_case() -> None:
    first = RequestSpec(
        "https://provider.example/v1/request",
        {"model": "model", "max_tokens": 1},
        {"Authorization": "secret-a", "Content-Type": "application/json"},
    )
    second = RequestSpec(
        "https://provider.example/v1/request",
        {"max_tokens": 1, "model": "model"},
        {"content-type": "application/json", "authorization": "secret-b"},
    )

    assert cloud_request_budget_fingerprint(
        first, billing_scope="account/project-a"
    ) == cloud_request_budget_fingerprint(second, billing_scope="account/project-a")


def test_terminal_transition_rejects_wrong_request_without_releasing_capacity() -> None:
    ledger = _ledger()
    ledger.reserve(
        "call-1",
        request_fingerprint=_REQUEST_FINGERPRINT,
        estimated_input_tokens=60,
        maximum_output_tokens=10,
    )
    wrong = "sha256:" + "b" * 64

    with pytest.raises(UsageBudgetStateError, match="does not match"):
        ledger.settle(
            "call-1",
            request_fingerprint=wrong,
            actual_input_tokens=1,
            actual_output_tokens=1,
        )

    assert ledger.snapshot().active_reservations == 1
    assert ledger.snapshot().reserved_input_tokens == 60


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"max_tokens": 1, "generationConfig": {"maxOutputTokens": 1}},
        {"max_tokens": True},
        {"generationConfig": []},
    ],
)
def test_request_cap_extraction_rejects_missing_ambiguous_or_invalid_fields(
    body: dict[str, object],
) -> None:
    request = RequestSpec(
        "https://provider.example/v1/request",
        body,
        {"Content-Type": "application/json"},
    )

    with pytest.raises(ValueError):
        request_maximum_output_tokens(request)


@pytest.mark.parametrize(
    ("url", "body"),
    [
        ("https://provider.example/v1/request", {"max_tokens": 1}),
        (
            "https://provider.example/v1beta/models/url-model:generateContent",
            {"model": "body-model", "max_tokens": 1},
        ),
        (
            "https://provider.example/v1/request",
            {"model": True, "max_tokens": 1},
        ),
    ],
)
def test_request_model_extraction_rejects_missing_ambiguous_or_invalid_sources(
    url: str, body: dict[str, object]
) -> None:
    request = RequestSpec(url, body, {"Content-Type": "application/json"})

    with pytest.raises(ValueError):
        request_model_identifier(request)


def test_manual_reservation_requires_canonical_request_fingerprint() -> None:
    ledger = _ledger()

    with pytest.raises(ValueError, match="lowercase sha256"):
        ledger.reserve(
            "call-1",
            request_fingerprint="unbound",
            estimated_input_tokens=1,
            maximum_output_tokens=1,
        )
