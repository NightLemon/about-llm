"""演示 prefix cache 如何同时校验身份、完整 token 与最长前缀。

实验故意让所有条目的 fingerprint 都碰撞，证明命中不能只比较哈希；又用不同 tenant 查询
同一 token 前缀，确认权限身份不一致时不会复用缓存。
"""

from __future__ import annotations

import json
from dataclasses import replace

from about_llm.inference import PrefixCache, PrefixCacheIdentity


def _identity(tenant: str = "tenant-a") -> PrefixCacheIdentity:
    """构造会影响 KV 可复用性的完整运行身份。"""

    # 除 token 外，租户、ACL、模型、tokenizer、模板、RoPE 与 dtype 都会影响安全或数值。
    return PrefixCacheIdentity(
        trusted_tenant_id=tenant,
        visibility_domain="private",
        authorization_revision="acl-v3",
        policy_revision="policy-v5",
        model_revision="model-sha-111",
        tokenizer_revision="tokenizer-sha-222",
        chat_template_revision="template-sha-333",
        adapter_revision="adapter-none",
        position_config_revision="rope-base-10000-max-8192",
        kv_dtype="float16",
    )


def main() -> None:
    """注入哈希碰撞，检查最长前缀命中和跨租户 miss。"""

    # fingerprint 函数恒定返回 collision，模拟最坏情况的哈希碰撞。
    cache = PrefixCache(
        capacity_entries=3,
        fingerprint=lambda _identity, _tokens: "collision",
    )
    identity = _identity()
    cache.store(identity, (11, 12))
    cache.store(identity, (11, 12, 13))
    cache.store(replace(identity, trusted_tenant_id="tenant-b"), (11, 12, 13))

    # 当前身份下应选择三 token 条目，而不是较短的两 token 条目。
    longest = cache.acquire_longest_prefix(identity, (11, 12, 13, 14))
    if longest is None:
        raise RuntimeError("fixture expected a longest-prefix hit")
    # token 完全相同但 tenant 不同，必须当作 miss。
    cross_tenant = cache.acquire_longest_prefix(
        replace(identity, trusted_tenant_id="tenant-c"), (11, 12, 13, 14)
    )
    if cross_tenant is not None:
        raise RuntimeError("fixture unexpectedly reused a cross-tenant prefix")
    # acquire 创建 lease，使用结束后显式 release 才允许条目被安全淘汰。
    cache.release(longest)

    report = cache.report()
    artifact = {
        "fixture": {
            "injected_fingerprint": "collision",
            "longest_match_token_ids": longest.matched_token_ids,
            "longest_match_length": longest.matched_length,
            "cross_tenant_result": None,
        },
        "report": {
            "capacity_entries": report.capacity_entries,
            "resident_entries": report.resident_entries,
            "leased_entries": report.leased_entries,
            "active_leases": report.active_leases,
            "hits": report.hits,
            "misses": report.misses,
            "evictions": report.evictions,
        },
        "scope": {
            "full_identity_and_exact_token_comparison_executed": True,
            "fingerprint_collision_injected": True,
            "cross_tenant_reuse_observed": False,
            "real_kv_tensors_or_gpu_runtime_executed": False,
            "vram_latency_hit_rate_or_prefill_savings_proved": False,
            "timing_channel_mitigation_proved": False,
            "fingerprint_confidentiality_or_authorization_proved": False,
            "vllm_prefix_cache_equivalence_proved": False,
        },
    }
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
