"""用内存账本演示云 API 请求的预算预留、结算与不确定状态。

第一次调用按供应商返回的实际 token 数结算；第二次模拟请求已发出但无法确认账单，
因此进入 uncertain 而不是退款。示例不发网络请求，只展示预算状态机。
"""

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
    """按时间顺序执行两次逻辑调用并打印账本快照。"""

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
    print("reserved:", second)
    print(
        "uncertain:",
        ledger.mark_usage_uncertain(
            "call-2", request_fingerprint=second.request_fingerprint
        ),
    )


if __name__ == "__main__":
    main()
