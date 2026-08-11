from __future__ import annotations

from datetime import datetime, timezone

from about_llm.integrations.cloud_api import (
    ChatMessage,
    build_openai_compatible_request,
)
from about_llm.integrations.usage_budget import (
    TokenPricingSnapshot,
    UsageBudgetLedger,
    UsageBudgetLimits,
)


def main() -> None:
    pricing = TokenPricingSnapshot(
        pricing_id="authored-provider/model@price-v1",
        provider="authored-provider",
        model="model",
        revision="price-v1",
        checked_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        input_microusd_per_million=1_000_000,
        output_microusd_per_million=2_000_000,
    )
    ledger = UsageBudgetLedger(
        limits=UsageBudgetLimits(
            max_input_tokens=100,
            max_output_tokens=20,
            max_estimated_microusd=140,
        ),
        pricing=pricing,
    )
    messages = [ChatMessage("user", "authored request")]
    first_request = build_openai_compatible_request(
        base_url="https://provider.invalid",
        api_key="example-secret-not-real",
        model="model",
        messages=messages,
        max_tokens=10,
    )

    first = ledger.reserve_request(
        "call-1",
        request=first_request,
        billing_scope="authored-account/project",
        estimated_input_tokens=60,
    )
    print("reserved:", first)
    print(
        "settled:",
        ledger.settle(
            "call-1",
            request_fingerprint=first.request_fingerprint,
            actual_input_tokens=58,
            actual_output_tokens=4,
        ),
    )

    second_request = build_openai_compatible_request(
        base_url="https://provider.invalid",
        api_key="rotated-example-secret-not-real",
        model="model",
        messages=messages,
        max_tokens=5,
    )
    second = ledger.reserve_request(
        "call-2",
        request=second_request,
        billing_scope="authored-account/project",
        estimated_input_tokens=20,
    )
    print("reserved:", second)
    print(
        "uncertain:",
        ledger.mark_usage_uncertain(
            "call-2", request_fingerprint=second.request_fingerprint
        ),
    )


if __name__ == "__main__":
    main()
