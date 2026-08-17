"""Offline controls for context-bound opaque reasoning artifacts.

This module models a provider-neutral envelope with authored bytes. It does not
parse, mint, or attack any real provider artifact.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

REASONING_ENVELOPE_VERSION = "about-llm.reasoning-envelope.v1"
BindingMode = Literal["content-only", "context-bound"]


class ReasoningArtifactError(ValueError):
    """Stable, value-redacted failure at the reasoning-artifact boundary."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"reasoning artifact rejected: {reason}")


@dataclass(frozen=True)
class ReasoningArtifactClaims:
    """Cleartext metadata authenticated by a context-bound envelope."""

    artifact_id: str
    provider: str
    key_id: str
    subject_id: str
    tenant_id: str
    session_id: str
    branch_id: str
    predecessor_digest: str
    model_audience: tuple[str, ...]
    issued_at_epoch_seconds: int
    expires_at_epoch_seconds: int

    def __post_init__(self) -> None:
        string_fields = (
            self.artifact_id,
            self.provider,
            self.key_id,
            self.subject_id,
            self.tenant_id,
            self.session_id,
            self.branch_id,
        )
        if any(not isinstance(value, str) or not value for value in string_fields):
            raise ValueError("reasoning artifact claims require non-empty strings")
        if (
            len(self.predecessor_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.predecessor_digest)
        ):
            raise ValueError("predecessor_digest must be 64 lowercase hex characters")
        if (
            not self.model_audience
            or tuple(sorted(set(self.model_audience))) != self.model_audience
            or any(not isinstance(model, str) or not model for model in self.model_audience)
        ):
            raise ValueError("model_audience must be a sorted tuple of unique model ids")
        timestamps = (self.issued_at_epoch_seconds, self.expires_at_epoch_seconds)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in timestamps):
            raise ValueError("reasoning artifact timestamps must be integers")
        if self.issued_at_epoch_seconds < 0:
            raise ValueError("issued_at_epoch_seconds must be non-negative")
        if self.expires_at_epoch_seconds <= self.issued_at_epoch_seconds:
            raise ValueError("reasoning artifact expiry must be after issuance")

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON fields used as associated data."""
        return {
            "artifact_id": self.artifact_id,
            "branch_id": self.branch_id,
            "expires_at_epoch_seconds": self.expires_at_epoch_seconds,
            "issued_at_epoch_seconds": self.issued_at_epoch_seconds,
            "key_id": self.key_id,
            "model_audience": list(self.model_audience),
            "predecessor_digest": self.predecessor_digest,
            "provider": self.provider,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "tenant_id": self.tenant_id,
        }


@dataclass(frozen=True)
class ReasoningReplayContext:
    """Trusted control-plane values supplied when an envelope is consumed."""

    provider: str
    subject_id: str
    tenant_id: str
    session_id: str
    branch_id: str
    predecessor_digest: str
    model_id: str
    now_epoch_seconds: int

    def __post_init__(self) -> None:
        strings = (
            self.provider,
            self.subject_id,
            self.tenant_id,
            self.session_id,
            self.branch_id,
            self.model_id,
        )
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("reasoning replay context requires non-empty strings")
        if (
            len(self.predecessor_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.predecessor_digest)
        ):
            raise ValueError("predecessor_digest must be 64 lowercase hex characters")
        if not isinstance(self.now_epoch_seconds, int) or isinstance(
            self.now_epoch_seconds, bool
        ):
            raise ValueError("now_epoch_seconds must be an integer")


@dataclass(frozen=True)
class ReasoningEnvelope:
    """Authored AEAD envelope used only by the local protocol control."""

    binding_mode: BindingMode
    claims: ReasoningArtifactClaims
    nonce: bytes
    ciphertext: bytes
    version: str = REASONING_ENVELOPE_VERSION

    def __post_init__(self) -> None:
        if self.binding_mode not in {"content-only", "context-bound"}:
            raise ValueError("unsupported reasoning envelope binding mode")
        if self.version != REASONING_ENVELOPE_VERSION:
            raise ValueError("unsupported reasoning envelope version")
        if not isinstance(self.nonce, bytes) or len(self.nonce) != 12:
            raise ValueError("reasoning envelope nonce must contain 12 bytes")
        if not isinstance(self.ciphertext, bytes) or len(self.ciphertext) <= 16:
            raise ValueError("reasoning envelope ciphertext is invalid")


class InMemoryNonceLedger:
    """Reject nonce reuse for one key within this teaching process."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, bytes]] = set()
        self._lock = threading.Lock()

    def claim(self, *, key_id: str, nonce: bytes) -> None:
        identity = (key_id, nonce)
        with self._lock:
            if identity in self._seen:
                raise ReasoningArtifactError("nonce_reused")
            self._seen.add(identity)


class InMemoryConsumptionLedger:
    """Enforce single consumption for an authenticated artifact identity."""

    def __init__(self) -> None:
        self._consumed: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def claim(self, *, key_id: str, artifact_id: str) -> None:
        identity = (key_id, artifact_id)
        with self._lock:
            if identity in self._consumed:
                raise ReasoningArtifactError("replay_detected")
            self._consumed.add(identity)


def issue_reasoning_envelope(
    *,
    key: bytes,
    claims: ReasoningArtifactClaims,
    plaintext: bytes,
    binding_mode: BindingMode,
    nonce: bytes,
    nonce_ledger: InMemoryNonceLedger,
) -> ReasoningEnvelope:
    """Issue a local envelope using a caller-supplied deterministic nonce."""
    _validate_key(key)
    if not isinstance(plaintext, bytes) or not plaintext:
        raise ValueError("reasoning plaintext must contain bytes")
    if not isinstance(nonce, bytes) or len(nonce) != 12:
        raise ValueError("reasoning envelope nonce must contain 12 bytes")
    nonce_ledger.claim(key_id=claims.key_id, nonce=nonce)
    associated_data = _associated_data(binding_mode=binding_mode, claims=claims)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data)
    return ReasoningEnvelope(
        binding_mode=binding_mode,
        claims=claims,
        nonce=nonce,
        ciphertext=ciphertext,
    )


def consume_reasoning_envelope(
    envelope: ReasoningEnvelope,
    *,
    keys: Mapping[str, bytes],
    retired_key_ids: frozenset[str],
    context: ReasoningReplayContext,
    consumption_ledger: InMemoryConsumptionLedger | None = None,
) -> bytes:
    """Authenticate, authorize, and optionally consume one local envelope."""
    key_id = envelope.claims.key_id
    if key_id in retired_key_ids:
        raise ReasoningArtifactError("retired_key")
    key = keys.get(key_id)
    if key is None:
        raise ReasoningArtifactError("unknown_key")
    _validate_key(key)
    try:
        plaintext = AESGCM(key).decrypt(
            envelope.nonce,
            envelope.ciphertext,
            _associated_data(
                binding_mode=envelope.binding_mode,
                claims=envelope.claims,
            ),
        )
    except InvalidTag as error:
        raise ReasoningArtifactError("authentication_failed") from error
    _authorize_replay(envelope, context=context)
    if consumption_ledger is not None:
        consumption_ledger.claim(
            key_id=envelope.claims.key_id,
            artifact_id=envelope.claims.artifact_id,
        )
    return plaintext


def _associated_data(
    *, binding_mode: BindingMode, claims: ReasoningArtifactClaims
) -> bytes:
    payload: dict[str, object] = {
        "binding_mode": binding_mode,
        "key_id": claims.key_id,
        "provider": claims.provider,
        "version": REASONING_ENVELOPE_VERSION,
    }
    if binding_mode == "context-bound":
        payload["context"] = claims.as_dict()
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _authorize_replay(
    envelope: ReasoningEnvelope, *, context: ReasoningReplayContext
) -> None:
    claims = envelope.claims
    if claims.provider != context.provider:
        raise ReasoningArtifactError("provider_mismatch")
    if envelope.binding_mode == "content-only":
        return
    checks = (
        (claims.subject_id == context.subject_id, "subject_mismatch"),
        (claims.tenant_id == context.tenant_id, "tenant_mismatch"),
        (claims.session_id == context.session_id, "session_mismatch"),
        (claims.branch_id == context.branch_id, "branch_mismatch"),
        (
            claims.predecessor_digest == context.predecessor_digest,
            "predecessor_mismatch",
        ),
        (context.model_id in claims.model_audience, "model_not_allowed"),
        (
            context.now_epoch_seconds >= claims.issued_at_epoch_seconds,
            "not_yet_valid",
        ),
        (context.now_epoch_seconds < claims.expires_at_epoch_seconds, "expired"),
    )
    for accepted, reason in checks:
        if not accepted:
            raise ReasoningArtifactError(reason)


def _validate_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) not in {16, 24, 32}:
        raise ValueError("AES-GCM key must contain 16, 24, or 32 bytes")
