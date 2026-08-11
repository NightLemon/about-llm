from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from about_llm.inference import (
    PrefixCache,
    PrefixCacheCapacityError,
    PrefixCacheIdentity,
    PrefixCacheLeaseError,
    sha256_prefix_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]


def _identity() -> PrefixCacheIdentity:
    return PrefixCacheIdentity(
        trusted_tenant_id="tenant-a",
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


def test_longest_exact_token_prefix_is_leased_and_accounted() -> None:
    cache = PrefixCache(capacity_entries=3)
    identity = _identity()
    short = cache.store(identity, [11, 12])
    long = cache.store(identity, (11, 12, 13))

    lease = cache.acquire_longest_prefix(identity, [11, 12, 13, 14])

    assert lease is not None
    assert lease.entry_id == long.entry_id
    assert lease.entry_id != short.entry_id
    assert lease.matched_token_ids == (11, 12, 13)
    assert lease.matched_length == 3
    assert cache.report().hits == 1
    assert cache.report().active_leases == 1
    cache.release(lease)
    assert cache.report().active_leases == 0


@pytest.mark.parametrize(
    "changed_field",
    [
        "trusted_tenant_id",
        "visibility_domain",
        "authorization_revision",
        "policy_revision",
        "model_revision",
        "tokenizer_revision",
        "chat_template_revision",
        "adapter_revision",
        "position_config_revision",
        "kv_dtype",
    ],
)
def test_every_security_and_execution_identity_dimension_partitions_reuse(
    changed_field: str,
) -> None:
    cache = PrefixCache(capacity_entries=1)
    identity = _identity()
    cache.store(identity, (7, 8, 9))
    changed = replace(identity, **{changed_field: f"different-{changed_field}"})

    assert cache.acquire_longest_prefix(changed, (7, 8, 9, 10)) is None
    assert cache.report().misses == 1


def test_injected_hash_collision_never_bypasses_full_identity_or_token_comparison() -> None:
    cache = PrefixCache(
        capacity_entries=3,
        fingerprint=lambda _identity, _tokens: "same-fingerprint",
    )
    identity = _identity()
    other_tenant = replace(identity, trusted_tenant_id="tenant-b")
    first = cache.store(identity, (1, 2))
    second = cache.store(identity, (1, 3))
    third = cache.store(other_tenant, (1, 2))

    exact = cache.acquire_longest_prefix(identity, (1, 2, 4))
    wrong_tokens = cache.acquire_longest_prefix(identity, (1, 4))

    assert len({first.entry_id, second.entry_id, third.entry_id}) == 3
    assert exact is not None and exact.entry_id == first.entry_id
    assert wrong_tokens is None
    cache.release(exact)


def test_leases_pin_entries_lru_evicts_only_unleased_and_full_failure_is_atomic() -> None:
    cache = PrefixCache(capacity_entries=2)
    identity = _identity()
    first = cache.store(identity, (1,))
    second = cache.store(identity, (2,))
    first_lease = cache.acquire_longest_prefix(identity, (1, 9))
    assert first_lease is not None

    third = cache.store(identity, (3,))
    assert {entry.entry_id for entry in cache.entries()} == {first.entry_id, third.entry_id}
    assert second.entry_id not in {entry.entry_id for entry in cache.entries()}
    assert cache.report().evictions == 1

    third_lease = cache.acquire_longest_prefix(identity, (3, 9))
    assert third_lease is not None
    before_entries = cache.entries()
    before_report = cache.report()
    with pytest.raises(PrefixCacheCapacityError, match="every resident entry"):
        cache.store(identity, (4,))
    assert cache.entries() == before_entries
    assert cache.report() == before_report

    cache.release(first_lease)
    cache.release(third_lease)
    fourth = cache.store(identity, (4,))
    assert first.entry_id not in {entry.entry_id for entry in cache.entries()}
    assert fourth.entry_id > third.entry_id


def test_duplicate_store_is_idempotent_even_while_entry_is_leased() -> None:
    cache = PrefixCache(capacity_entries=1)
    identity = _identity()
    original = cache.store(identity, (5, 6))
    lease = cache.acquire_longest_prefix(identity, (5, 6, 7))
    assert lease is not None

    duplicate = cache.store(identity, [5, 6])

    assert duplicate.entry_id == original.entry_id
    assert duplicate.lease_count == 1
    assert cache.report().resident_entries == 1
    assert cache.report().evictions == 0
    cache.release(lease)


def test_double_stale_and_cross_cache_release_fail_closed() -> None:
    identity = _identity()
    first_cache = PrefixCache(capacity_entries=1)
    second_cache = PrefixCache(capacity_entries=1)
    first_cache.store(identity, (1,))
    lease = first_cache.acquire_longest_prefix(identity, (1, 2))
    assert lease is not None

    with pytest.raises(PrefixCacheLeaseError, match="different cache"):
        second_cache.release(lease)
    first_cache.release(lease)
    with pytest.raises(PrefixCacheLeaseError, match="already released"):
        first_cache.release(lease)


@pytest.mark.parametrize(
    ("operation", "error_type", "message"),
    [
        (lambda: PrefixCache(capacity_entries=0), ValueError, "positive integer"),
        (
            lambda: PrefixCacheIdentity(
                **{**_identity().to_dict(), "policy_revision": ""}
            ),
            ValueError,
            "policy_revision",
        ),
        (
            lambda: PrefixCache(capacity_entries=1).store(_identity(), ()),
            ValueError,
            "cannot be empty",
        ),
        (
            lambda: PrefixCache(capacity_entries=1).store(_identity(), (1, -1)),
            ValueError,
            "non-negative integer",
        ),
        (
            lambda: PrefixCache(
                capacity_entries=1, fingerprint=lambda _identity, _tokens: ""
            ).store(_identity(), (1,)),
            ValueError,
            "fingerprint result",
        ),
    ],
)
def test_invalid_contracts_fail_closed(
    operation: Callable[[], object], error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        operation()


def test_default_fingerprint_is_deterministic_but_not_a_security_boundary() -> None:
    identity = _identity()
    assert sha256_prefix_fingerprint(identity, (1, 2)) == sha256_prefix_fingerprint(
        identity, (1, 2)
    )
    assert sha256_prefix_fingerprint(identity, (1, 2)) != sha256_prefix_fingerprint(
        identity, (1, 3)
    )


def test_prefix_cache_toy_records_collision_isolation_and_scope() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "projects" / "inference-serving" / "prefix_cache_toy.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = json.loads(completed.stdout)

    assert artifact["fixture"]["injected_fingerprint"] == "collision"
    assert artifact["fixture"]["longest_match_token_ids"] == [11, 12, 13]
    assert artifact["fixture"]["longest_match_length"] == 3
    assert artifact["fixture"]["cross_tenant_result"] is None
    assert artifact["report"] == {
        "active_leases": 0,
        "capacity_entries": 3,
        "evictions": 0,
        "hits": 1,
        "leased_entries": 0,
        "misses": 1,
        "resident_entries": 3,
    }
    assert artifact["scope"] == {
        "cross_tenant_reuse_observed": False,
        "fingerprint_collision_injected": True,
        "fingerprint_confidentiality_or_authorization_proved": False,
        "full_identity_and_exact_token_comparison_executed": True,
        "real_kv_tensors_or_gpu_runtime_executed": False,
        "timing_channel_mitigation_proved": False,
        "vllm_prefix_cache_equivalence_proved": False,
        "vram_latency_hit_rate_or_prefill_savings_proved": False,
    }
