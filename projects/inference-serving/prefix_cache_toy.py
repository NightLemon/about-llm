"""Emit a deterministic, collision-safe prefix-cache metadata fixture."""

from __future__ import annotations

import json
from dataclasses import replace

from about_llm.inference import PrefixCache, PrefixCacheIdentity


def _identity(tenant: str = "tenant-a") -> PrefixCacheIdentity:
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
    cache = PrefixCache(
        capacity_entries=3,
        fingerprint=lambda _identity, _tokens: "collision",
    )
    identity = _identity()
    cache.store(identity, (11, 12))
    cache.store(identity, (11, 12, 13))
    cache.store(replace(identity, trusted_tenant_id="tenant-b"), (11, 12, 13))

    longest = cache.acquire_longest_prefix(identity, (11, 12, 13, 14))
    if longest is None:
        raise RuntimeError("fixture expected a longest-prefix hit")
    cross_tenant = cache.acquire_longest_prefix(
        replace(identity, trusted_tenant_id="tenant-c"), (11, 12, 13, 14)
    )
    if cross_tenant is not None:
        raise RuntimeError("fixture unexpectedly reused a cross-tenant prefix")
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
