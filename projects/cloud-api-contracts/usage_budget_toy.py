"""用内存账本演示云 API 请求的预算预留、结算与不确定状态。

第一次调用按供应商返回的实际 token 数结算；第二次模拟请求已发出但无法确认账单，
因此进入 uncertain 而不是退款。示例不发网络请求，只展示预算状态机。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from about_llm.integrations.cloud_api import (
    ChatMessage,
    build_openai_compatible_request,
)
from about_llm.integrations.usage_budget import (
    TokenPricingSnapshot,
    UsageBudgetLedger,
    UsageBudgetLimits,
)


def run_demo() -> dict[str, Any]:
    """按时间顺序执行两次逻辑调用并返回完整账本报告。"""

    # 价格以每百万 token 的微美元记录，整数运算避免浮点货币误差。
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

    # 调用前按预计输入与最大输出预留，成功后再替换成实际 usage。
    first = ledger.reserve_request(
        "call-1",
        request=first_request,
        billing_scope="authored-account/project",
        estimated_input_tokens=60,
    )
    settled = ledger.settle(
        "call-1",
        request_fingerprint=first.request_fingerprint,
        actual_input_tokens=58,
        actual_output_tokens=4,
    )

    # 更换 API key 不应改变请求的计费指纹；密钥本身也不应写入账本。
    second_request = build_openai_compatible_request(
        base_url="https://provider.invalid",
        api_key="rotated-example-secret-not-real",
        model="model",
        messages=messages,
        max_tokens=5,
    )
    # 状态不明时保留最坏情况额度，等待供应商账单或人工流程对账。
    second = ledger.reserve_request(
        "call-2",
        request=second_request,
        billing_scope="authored-account/project",
        estimated_input_tokens=20,
    )
    uncertain = ledger.mark_usage_uncertain(
        "call-2", request_fingerprint=second.request_fingerprint
    )
    return {
        "schema_version": 1,
        "scenario": "in-memory pre-call reservation and post-call usage accounting",
        "configuration": {
            "pricing": {
                "pricing_id": pricing.pricing_id,
                "provider": pricing.provider,
                "model": pricing.model,
                "revision": pricing.revision,
                "checked_at": pricing.checked_at.isoformat(),
                "input_microusd_per_million": pricing.input_microusd_per_million,
                "output_microusd_per_million": pricing.output_microusd_per_million,
            },
            "limits": {
                "max_input_tokens": 100,
                "max_output_tokens": 20,
                "max_estimated_microusd": 140,
            },
            "billing_scope": "authored-account/project",
        },
        "input": {
            "messages": [asdict(message) for message in messages],
            "model": "model",
            "base_url": "https://provider.invalid",
            "api_key_included_in_report": False,
        },
        "transitions": [
            {
                "reservation_id": "call-1",
                "path": ["reserved", "settled"],
                "reservation": asdict(first),
                "provider_reported_usage": {
                    "input_tokens": 58,
                    "output_tokens": 4,
                },
                "final_snapshot": asdict(settled),
            },
            {
                "reservation_id": "call-2",
                "path": ["reserved", "uncertain"],
                "reservation": asdict(second),
                "provider_reported_usage": None,
                "final_snapshot": asdict(uncertain),
            },
        ],
        "conclusion": {
            "settled_call_commits_reported_usage_and_releases_unused_reservation": True,
            "uncertain_call_commits_the_full_reserved_worst_case": True,
        },
        "scope": {
            "ledger_storage": "in-memory",
            "network_used": False,
            "provider_response_or_invoice_authenticated": False,
            "exactly_once_remote_billing_proved": False,
            "microusd_values_are_local_estimates": True,
        },
    }


def main() -> None:
    """执行两条预算状态路径并打印结构化报告。"""

    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
