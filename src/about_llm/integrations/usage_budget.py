"""Concurrency-safe token and estimated-cost reservations for cloud calls.

The ledger is deliberately provider neutral. It enforces a caller-supplied
pricing snapshot and token caps; it does not claim to reproduce a provider's
invoice, cached/reasoning-token accounting, taxes, credits, or billing tiers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any
from urllib.parse import unquote, urlsplit

from about_llm.integrations.cloud_api import CREDENTIAL_HEADER_NAMES, RequestSpec

_TOKENS_PER_MILLION = 1_000_000
_BUDGET_REQUEST_IDENTITY_VERSION = "about-llm.cloud-budget-request.v1"


def _non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _non_negative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _sha256_fingerprint(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
    ):
        raise ValueError(f"{name} must be a lowercase sha256: fingerprint")
    try:
        digest_value = int(value[7:], 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase sha256: fingerprint") from error
    if value[7:] != f"{digest_value:064x}":
        raise ValueError(f"{name} must be a lowercase sha256: fingerprint")


def request_maximum_output_tokens(request: RequestSpec) -> int:
    """Extract the output cap from one supported text request contract."""

    if not isinstance(request, RequestSpec):
        raise TypeError("request must be RequestSpec")
    body = request.body
    candidates: list[tuple[str, Any]] = []
    if "max_tokens" in body:
        candidates.append(("max_tokens", body["max_tokens"]))
    generation_config = body.get("generationConfig")
    if generation_config is not None:
        if not isinstance(generation_config, dict):
            raise ValueError("request generationConfig must be an object")
        if "maxOutputTokens" in generation_config:
            candidates.append(
                ("generationConfig.maxOutputTokens", generation_config["maxOutputTokens"])
            )
    if len(candidates) != 1:
        raise ValueError(
            "request must contain exactly one supported maximum-output field"
        )
    field, value = candidates[0]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"request {field} must be a positive integer")
    return value


def request_model_identifier(request: RequestSpec) -> str:
    """Extract the exact model identifier from one supported request shape."""

    if not isinstance(request, RequestSpec):
        raise TypeError("request must be RequestSpec")
    candidates: list[tuple[str, Any]] = []
    if "model" in request.body:
        candidates.append(("body.model", request.body["model"]))
    path = urlsplit(request.url).path
    marker = "/models/"
    suffix = ":generateContent"
    if marker in path and path.endswith(suffix):
        candidates.append(
            (
                "url.models",
                unquote(path.rsplit(marker, 1)[1][: -len(suffix)]),
            )
        )
    if len(candidates) != 1:
        raise ValueError("request must contain exactly one supported model identifier")
    field, model = candidates[0]
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"request {field} must be a non-empty string")
    return model


def cloud_request_budget_fingerprint(
    request: RequestSpec, *, billing_scope: str
) -> str:
    """Bind request semantics without exposing credential header values.

    ``billing_scope`` is a caller-selected stable account/project identifier,
    not a credential. Credential values are replaced before canonical hashing,
    so rotating a key in the same billing scope preserves request identity.
    """

    if not isinstance(request, RequestSpec):
        raise TypeError("request must be RequestSpec")
    _non_empty(billing_scope, "billing_scope")
    headers = [
        [
            name.lower(),
            (
                "<credential-bound-by-billing-scope>"
                if name.lower() in CREDENTIAL_HEADER_NAMES
                else value
            ),
        ]
        for name, value in request.headers.items()
    ]
    headers.sort(key=lambda item: (item[0], item[1]))
    canonical = json.dumps(
        {
            "identity_version": _BUDGET_REQUEST_IDENTITY_VERSION,
            "billing_scope": billing_scope,
            "url": request.url,
            "headers": headers,
            "body": dict(request.body),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class TokenPricingSnapshot:
    """Caller-reviewed token rates, expressed as micro-USD per million tokens."""

    pricing_id: str
    provider: str
    model: str
    revision: str
    checked_at: datetime
    input_microusd_per_million: int
    output_microusd_per_million: int

    def __post_init__(self) -> None:
        for name in ("pricing_id", "provider", "model", "revision"):
            _non_empty(getattr(self, name), name)
        if not isinstance(self.checked_at, datetime) or self.checked_at.tzinfo is None:
            raise ValueError("checked_at must be a timezone-aware datetime")
        if self.checked_at.utcoffset() is None:
            raise ValueError("checked_at must have a valid UTC offset")
        _non_negative_integer(
            self.input_microusd_per_million, "input_microusd_per_million"
        )
        _non_negative_integer(
            self.output_microusd_per_million, "output_microusd_per_million"
        )

    def quote_microusd(self, input_tokens: int, output_tokens: int) -> int:
        """Return a conservative per-call estimate, rounded up once."""

        _non_negative_integer(input_tokens, "input_tokens")
        _non_negative_integer(output_tokens, "output_tokens")
        numerator = (
            input_tokens * self.input_microusd_per_million
            + output_tokens * self.output_microusd_per_million
        )
        if numerator == 0:
            return 0
        return (numerator + _TOKENS_PER_MILLION - 1) // _TOKENS_PER_MILLION


@dataclass(frozen=True)
class UsageBudgetLimits:
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_estimated_microusd: int | None = None

    def __post_init__(self) -> None:
        if all(
            value is None
            for value in (
                self.max_input_tokens,
                self.max_output_tokens,
                self.max_estimated_microusd,
            )
        ):
            raise ValueError("at least one usage budget limit is required")
        for name in (
            "max_input_tokens",
            "max_output_tokens",
            "max_estimated_microusd",
        ):
            value = getattr(self, name)
            if value is not None:
                _non_negative_integer(value, name)


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    request_fingerprint: str
    input_tokens: int
    maximum_output_tokens: int
    maximum_estimated_microusd: int
    pricing_id: str


@dataclass(frozen=True)
class UsageBudgetSnapshot:
    pricing_id: str
    max_input_tokens: int | None
    max_output_tokens: int | None
    max_estimated_microusd: int | None
    committed_input_tokens: int
    committed_output_tokens: int
    committed_estimated_microusd: int
    reserved_input_tokens: int
    reserved_output_tokens: int
    reserved_estimated_microusd: int
    active_reservations: int
    uncertain_settlements: int
    remaining_input_tokens: int | None
    remaining_output_tokens: int | None
    remaining_estimated_microusd: int | None
    over_limit: bool


class UsageBudgetExceededError(RuntimeError):
    """A new call cannot fit within the remaining reserved capacity."""


class PostCallBudgetExceededError(RuntimeError):
    """Reported usage exceeded the pre-call reservation and a hard limit."""

    def __init__(self, snapshot: UsageBudgetSnapshot) -> None:
        self.snapshot = snapshot
        super().__init__("reported usage exceeded a hard usage budget after the call")


class UsageBudgetStateError(RuntimeError):
    """The requested reservation transition is invalid or already terminal."""


class UsageBudgetLedger:
    """In-memory atomic reservation ledger for one explicit pricing snapshot.

    Active calls reserve estimated input tokens, their maximum output tokens,
    and the resulting cost estimate. Successful calls settle to reported
    usage. An outcome with unknown usage consumes the whole reservation. Only
    a call known not to have been sent may be cancelled and fully released.
    """

    def __init__(
        self, *, limits: UsageBudgetLimits, pricing: TokenPricingSnapshot
    ) -> None:
        if not isinstance(limits, UsageBudgetLimits):
            raise TypeError("limits must be UsageBudgetLimits")
        if not isinstance(pricing, TokenPricingSnapshot):
            raise TypeError("pricing must be TokenPricingSnapshot")
        self._limits = limits
        self._pricing = pricing
        self._lock = Lock()
        self._active: dict[str, BudgetReservation] = {}
        self._seen_ids: set[str] = set()
        self._committed_input_tokens = 0
        self._committed_output_tokens = 0
        self._committed_estimated_microusd = 0
        self._uncertain_settlements = 0

    def reserve(
        self,
        reservation_id: str,
        *,
        request_fingerprint: str,
        estimated_input_tokens: int,
        maximum_output_tokens: int,
    ) -> BudgetReservation:
        """Atomically reserve worst-case capacity before sending a request."""

        _non_empty(reservation_id, "reservation_id")
        _sha256_fingerprint(request_fingerprint, "request_fingerprint")
        _non_negative_integer(estimated_input_tokens, "estimated_input_tokens")
        _non_negative_integer(maximum_output_tokens, "maximum_output_tokens")
        reservation = BudgetReservation(
            reservation_id=reservation_id,
            request_fingerprint=request_fingerprint,
            input_tokens=estimated_input_tokens,
            maximum_output_tokens=maximum_output_tokens,
            maximum_estimated_microusd=self._pricing.quote_microusd(
                estimated_input_tokens, maximum_output_tokens
            ),
            pricing_id=self._pricing.pricing_id,
        )
        with self._lock:
            if reservation_id in self._seen_ids:
                raise UsageBudgetStateError("reservation_id has already been used")
            projected = self._projected_totals_unlocked(reservation)
            if self._exceeds_limits(*projected):
                raise UsageBudgetExceededError(
                    "reservation would exceed a hard token or estimated-cost limit"
                )
            self._active[reservation_id] = reservation
            self._seen_ids.add(reservation_id)
        return reservation

    def reserve_request(
        self,
        reservation_id: str,
        *,
        request: RequestSpec,
        billing_scope: str,
        estimated_input_tokens: int,
    ) -> BudgetReservation:
        """Bind reservation identity and output cap to a supported RequestSpec."""

        request_model = request_model_identifier(request)
        if request_model != self._pricing.model:
            raise ValueError(
                "request model must exactly match the pricing snapshot model"
            )
        return self.reserve(
            reservation_id,
            request_fingerprint=cloud_request_budget_fingerprint(
                request, billing_scope=billing_scope
            ),
            estimated_input_tokens=estimated_input_tokens,
            maximum_output_tokens=request_maximum_output_tokens(request),
        )

    def settle(
        self,
        reservation_id: str,
        *,
        request_fingerprint: str,
        actual_input_tokens: int,
        actual_output_tokens: int,
    ) -> UsageBudgetSnapshot:
        """Commit provider-reported usage and release unused reserved capacity.

        If reported usage is larger than reserved capacity, it is still
        committed because the call already happened. A post-call breach is
        then raised with the resulting snapshot and future reservations remain
        fail-closed until an operator replaces or resets the ledger.
        """

        _non_empty(reservation_id, "reservation_id")
        _sha256_fingerprint(request_fingerprint, "request_fingerprint")
        _non_negative_integer(actual_input_tokens, "actual_input_tokens")
        _non_negative_integer(actual_output_tokens, "actual_output_tokens")
        actual_cost = self._pricing.quote_microusd(
            actual_input_tokens, actual_output_tokens
        )
        with self._lock:
            self._take_active_unlocked(reservation_id, request_fingerprint)
            self._committed_input_tokens += actual_input_tokens
            self._committed_output_tokens += actual_output_tokens
            self._committed_estimated_microusd += actual_cost
            snapshot = self._snapshot_unlocked()
        if snapshot.over_limit:
            raise PostCallBudgetExceededError(snapshot)
        return snapshot

    def mark_usage_uncertain(
        self, reservation_id: str, *, request_fingerprint: str
    ) -> UsageBudgetSnapshot:
        """Conservatively commit the full reservation when usage is unknown."""

        _non_empty(reservation_id, "reservation_id")
        _sha256_fingerprint(request_fingerprint, "request_fingerprint")
        with self._lock:
            reservation = self._take_active_unlocked(
                reservation_id, request_fingerprint
            )
            self._committed_input_tokens += reservation.input_tokens
            self._committed_output_tokens += reservation.maximum_output_tokens
            self._committed_estimated_microusd += (
                reservation.maximum_estimated_microusd
            )
            self._uncertain_settlements += 1
            return self._snapshot_unlocked()

    def cancel_before_send(
        self, reservation_id: str, *, request_fingerprint: str
    ) -> UsageBudgetSnapshot:
        """Release a reservation only when transport is known not to have run."""

        _non_empty(reservation_id, "reservation_id")
        _sha256_fingerprint(request_fingerprint, "request_fingerprint")
        with self._lock:
            self._take_active_unlocked(reservation_id, request_fingerprint)
            return self._snapshot_unlocked()

    def snapshot(self) -> UsageBudgetSnapshot:
        with self._lock:
            return self._snapshot_unlocked()

    def _take_active_unlocked(
        self, reservation_id: str, request_fingerprint: str
    ) -> BudgetReservation:
        try:
            reservation = self._active[reservation_id]
        except KeyError as error:
            raise UsageBudgetStateError(
                "reservation is unknown or already terminal"
            ) from error
        if reservation.request_fingerprint != request_fingerprint:
            raise UsageBudgetStateError(
                "request_fingerprint does not match the active reservation"
            )
        return self._active.pop(reservation_id)

    def _reserved_totals_unlocked(self) -> tuple[int, int, int]:
        return (
            sum(item.input_tokens for item in self._active.values()),
            sum(item.maximum_output_tokens for item in self._active.values()),
            sum(item.maximum_estimated_microusd for item in self._active.values()),
        )

    def _projected_totals_unlocked(
        self, additional: BudgetReservation
    ) -> tuple[int, int, int]:
        reserved_input, reserved_output, reserved_cost = (
            self._reserved_totals_unlocked()
        )
        return (
            self._committed_input_tokens + reserved_input + additional.input_tokens,
            self._committed_output_tokens
            + reserved_output
            + additional.maximum_output_tokens,
            self._committed_estimated_microusd
            + reserved_cost
            + additional.maximum_estimated_microusd,
        )

    def _exceeds_limits(self, input_tokens: int, output_tokens: int, cost: int) -> bool:
        return any(
            limit is not None and value > limit
            for value, limit in (
                (input_tokens, self._limits.max_input_tokens),
                (output_tokens, self._limits.max_output_tokens),
                (cost, self._limits.max_estimated_microusd),
            )
        )

    def _snapshot_unlocked(self) -> UsageBudgetSnapshot:
        reserved_input, reserved_output, reserved_cost = (
            self._reserved_totals_unlocked()
        )
        projected = (
            self._committed_input_tokens + reserved_input,
            self._committed_output_tokens + reserved_output,
            self._committed_estimated_microusd + reserved_cost,
        )
        return UsageBudgetSnapshot(
            pricing_id=self._pricing.pricing_id,
            max_input_tokens=self._limits.max_input_tokens,
            max_output_tokens=self._limits.max_output_tokens,
            max_estimated_microusd=self._limits.max_estimated_microusd,
            committed_input_tokens=self._committed_input_tokens,
            committed_output_tokens=self._committed_output_tokens,
            committed_estimated_microusd=self._committed_estimated_microusd,
            reserved_input_tokens=reserved_input,
            reserved_output_tokens=reserved_output,
            reserved_estimated_microusd=reserved_cost,
            active_reservations=len(self._active),
            uncertain_settlements=self._uncertain_settlements,
            remaining_input_tokens=self._remaining(
                self._limits.max_input_tokens, projected[0]
            ),
            remaining_output_tokens=self._remaining(
                self._limits.max_output_tokens, projected[1]
            ),
            remaining_estimated_microusd=self._remaining(
                self._limits.max_estimated_microusd, projected[2]
            ),
            over_limit=self._exceeds_limits(*projected),
        )

    @staticmethod
    def _remaining(limit: int | None, used: int) -> int | None:
        return None if limit is None else max(0, limit - used)
