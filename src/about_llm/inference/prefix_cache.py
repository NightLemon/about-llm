"""Collision-safe metadata oracle for security-scoped prefix-cache reuse."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


class PrefixCacheCapacityError(RuntimeError):
    """Raised when every resident entry is leased and none can be evicted."""


class PrefixCacheLeaseError(RuntimeError):
    """Raised for a foreign, stale, or already released cache lease."""


def _non_empty_string(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")
    return value


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _token_tuple(token_ids: Sequence[int], label: str) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError(f"{label} must be a sequence of token ids")
    result = tuple(token_ids)
    if not result:
        raise ValueError(f"{label} cannot be empty")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in result
    ):
        raise ValueError(f"{label} must contain non-negative integer token ids")
    return result


@dataclass(frozen=True)
class PrefixCacheIdentity:
    """All trusted metadata that this oracle requires before KV reuse.

    The caller must derive tenant and authorization fields from trusted server
    state, never directly from an untrusted request body. Revisions should be
    immutable identifiers, not mutable display names.
    """

    trusted_tenant_id: str
    visibility_domain: str
    authorization_revision: str
    policy_revision: str
    model_revision: str
    tokenizer_revision: str
    chat_template_revision: str
    adapter_revision: str
    position_config_revision: str
    kv_dtype: str

    def __post_init__(self) -> None:
        for label, value in self.to_dict().items():
            _non_empty_string(value, label)

    def to_dict(self) -> dict[str, str]:
        return {
            "trusted_tenant_id": self.trusted_tenant_id,
            "visibility_domain": self.visibility_domain,
            "authorization_revision": self.authorization_revision,
            "policy_revision": self.policy_revision,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_revision": self.chat_template_revision,
            "adapter_revision": self.adapter_revision,
            "position_config_revision": self.position_config_revision,
            "kv_dtype": self.kv_dtype,
        }


PrefixFingerprint = Callable[[PrefixCacheIdentity, tuple[int, ...]], str]


def sha256_prefix_fingerprint(
    identity: PrefixCacheIdentity, token_ids: tuple[int, ...]
) -> str:
    """Return an unkeyed lookup fingerprint, not an authorization decision."""

    payload = {
        "identity": identity.to_dict(),
        "token_ids": token_ids,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PrefixCacheEntryState:
    entry_id: int
    identity: PrefixCacheIdentity
    token_ids: tuple[int, ...]
    fingerprint: str
    lease_count: int
    last_access_order: int

    @property
    def prefix_length(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class PrefixCacheLease:
    lease_id: int
    entry_id: int
    identity: PrefixCacheIdentity
    matched_token_ids: tuple[int, ...]
    fingerprint: str
    _owner_token: object = field(repr=False, compare=False)

    @property
    def matched_length(self) -> int:
        return len(self.matched_token_ids)


@dataclass(frozen=True)
class PrefixCacheReport:
    capacity_entries: int
    resident_entries: int
    leased_entries: int
    active_leases: int
    hits: int
    misses: int
    evictions: int


@dataclass
class _Entry:
    entry_id: int
    identity: PrefixCacheIdentity
    token_ids: tuple[int, ...]
    fingerprint: str
    lease_count: int
    last_access_order: int


class PrefixCache:
    """Bounded exact-prefix cache index that stores metadata, never K/V tensors.

    Fingerprints are only bucket hints. Every hit requires full identity and
    token-tuple equality, so even an injected fingerprint collision cannot
    cause cross-identity or cross-token reuse. Active leases pin entries against
    eviction; a full, entirely leased cache rejects insertion before mutation.
    """

    def __init__(
        self,
        *,
        capacity_entries: int,
        fingerprint: PrefixFingerprint = sha256_prefix_fingerprint,
    ) -> None:
        self.capacity_entries = _positive_integer(
            capacity_entries, "capacity_entries"
        )
        if not callable(fingerprint):
            raise TypeError("fingerprint must be callable")
        self._fingerprint = fingerprint
        self._entries: dict[int, _Entry] = {}
        self._fingerprint_index: dict[str, list[int]] = {}
        self._active_leases: dict[int, int] = {}
        self._next_entry_id = 0
        self._next_lease_id = 0
        self._access_order = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._owner_token = object()
        self._lock = threading.RLock()

    def store(
        self,
        identity: PrefixCacheIdentity,
        token_ids: Sequence[int],
    ) -> PrefixCacheEntryState:
        """Store one materialized prefix or return its existing exact entry."""

        identity = self._validate_identity(identity)
        tokens = _token_tuple(token_ids, "token_ids")
        with self._lock:
            fingerprint = self._compute_fingerprint(identity, tokens)
            for entry_id in self._fingerprint_index.get(fingerprint, []):
                entry = self._entries[entry_id]
                if entry.identity == identity and entry.token_ids == tokens:
                    entry.last_access_order = self._tick()
                    return self._entry_state(entry)

            victim: _Entry | None = None
            if len(self._entries) == self.capacity_entries:
                candidates = [
                    entry for entry in self._entries.values() if entry.lease_count == 0
                ]
                if not candidates:
                    raise PrefixCacheCapacityError(
                        "cache is full and every resident entry has an active lease"
                    )
                victim = min(
                    candidates,
                    key=lambda entry: (entry.last_access_order, entry.entry_id),
                )

            if victim is not None:
                self._remove_entry(victim)
                self._evictions += 1

            entry = _Entry(
                entry_id=self._next_entry_id,
                identity=identity,
                token_ids=tokens,
                fingerprint=fingerprint,
                lease_count=0,
                last_access_order=self._tick(),
            )
            self._next_entry_id += 1
            self._entries[entry.entry_id] = entry
            self._fingerprint_index.setdefault(fingerprint, []).append(entry.entry_id)
            self._check_invariants()
            return self._entry_state(entry)

    def acquire_longest_prefix(
        self,
        identity: PrefixCacheIdentity,
        query_token_ids: Sequence[int],
    ) -> PrefixCacheLease | None:
        """Lease the longest exact resident token prefix for one full identity."""

        identity = self._validate_identity(identity)
        query = _token_tuple(query_token_ids, "query_token_ids")
        with self._lock:
            match: _Entry | None = None
            for prefix_length in range(len(query), 0, -1):
                candidate_tokens = query[:prefix_length]
                fingerprint = self._compute_fingerprint(identity, candidate_tokens)
                for entry_id in self._fingerprint_index.get(fingerprint, []):
                    entry = self._entries[entry_id]
                    if (
                        entry.identity == identity
                        and entry.token_ids == candidate_tokens
                    ):
                        match = entry
                        break
                if match is not None:
                    break

            if match is None:
                self._misses += 1
                return None

            match.lease_count += 1
            match.last_access_order = self._tick()
            lease_id = self._next_lease_id
            self._next_lease_id += 1
            self._active_leases[lease_id] = match.entry_id
            self._hits += 1
            self._check_invariants()
            return PrefixCacheLease(
                lease_id=lease_id,
                entry_id=match.entry_id,
                identity=match.identity,
                matched_token_ids=match.token_ids,
                fingerprint=match.fingerprint,
                _owner_token=self._owner_token,
            )

    def release(self, lease: PrefixCacheLease) -> None:
        """Release exactly one live lease; double and cross-cache release fail."""

        if not isinstance(lease, PrefixCacheLease):
            raise TypeError("lease must be a PrefixCacheLease")
        with self._lock:
            if lease._owner_token is not self._owner_token:
                raise PrefixCacheLeaseError("lease belongs to a different cache")
            entry_id = self._active_leases.get(lease.lease_id)
            if entry_id is None:
                raise PrefixCacheLeaseError("lease is stale or already released")
            if entry_id != lease.entry_id:
                raise PrefixCacheLeaseError("lease entry identity is inconsistent")
            entry = self._entries.get(entry_id)
            if entry is None or entry.lease_count <= 0:
                raise RuntimeError("active lease references an invalid entry")
            entry.lease_count -= 1
            del self._active_leases[lease.lease_id]
            self._check_invariants()

    def entries(self) -> tuple[PrefixCacheEntryState, ...]:
        with self._lock:
            return tuple(
                self._entry_state(entry)
                for entry in sorted(
                    self._entries.values(), key=lambda item: item.entry_id
                )
            )

    def report(self) -> PrefixCacheReport:
        with self._lock:
            return PrefixCacheReport(
                capacity_entries=self.capacity_entries,
                resident_entries=len(self._entries),
                leased_entries=sum(
                    entry.lease_count > 0 for entry in self._entries.values()
                ),
                active_leases=len(self._active_leases),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )

    @staticmethod
    def _validate_identity(identity: PrefixCacheIdentity) -> PrefixCacheIdentity:
        if not isinstance(identity, PrefixCacheIdentity):
            raise TypeError("identity must be a PrefixCacheIdentity")
        return identity

    def _compute_fingerprint(
        self, identity: PrefixCacheIdentity, token_ids: tuple[int, ...]
    ) -> str:
        fingerprint = self._fingerprint(identity, token_ids)
        return _non_empty_string(fingerprint, "fingerprint result")

    def _tick(self) -> int:
        current = self._access_order
        self._access_order += 1
        return current

    def _remove_entry(self, entry: _Entry) -> None:
        if entry.lease_count != 0:
            raise RuntimeError("cannot evict a leased entry")
        bucket = self._fingerprint_index[entry.fingerprint]
        bucket.remove(entry.entry_id)
        if not bucket:
            del self._fingerprint_index[entry.fingerprint]
        del self._entries[entry.entry_id]

    @staticmethod
    def _entry_state(entry: _Entry) -> PrefixCacheEntryState:
        return PrefixCacheEntryState(
            entry_id=entry.entry_id,
            identity=entry.identity,
            token_ids=entry.token_ids,
            fingerprint=entry.fingerprint,
            lease_count=entry.lease_count,
            last_access_order=entry.last_access_order,
        )

    def _check_invariants(self) -> None:
        if len(self._entries) > self.capacity_entries:
            raise RuntimeError("resident entries exceed configured capacity")
        indexed_ids: list[int] = []
        for fingerprint, entry_ids in self._fingerprint_index.items():
            if len(entry_ids) != len(set(entry_ids)):
                raise RuntimeError("fingerprint bucket contains duplicate entry ids")
            for entry_id in entry_ids:
                entry = self._entries.get(entry_id)
                if entry is None or entry.fingerprint != fingerprint:
                    raise RuntimeError("fingerprint index is inconsistent")
                indexed_ids.append(entry_id)
        if sorted(indexed_ids) != sorted(self._entries):
            raise RuntimeError("not every resident entry is indexed exactly once")
        lease_counts = {entry_id: 0 for entry_id in self._entries}
        for entry_id in self._active_leases.values():
            if entry_id not in lease_counts:
                raise RuntimeError("active lease references a non-resident entry")
            lease_counts[entry_id] += 1
        if any(
            entry.lease_count != lease_counts[entry_id]
            for entry_id, entry in self._entries.items()
        ):
            raise RuntimeError("entry lease count is inconsistent")
