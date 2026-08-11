"""Deterministic MinHash/LSH candidates with exact lexical rechecking."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from about_llm.llmops import artifact_fingerprint

MINHASH_LSH_VERSION = "about-llm.minhash-lsh.v1"
_MERSENNE_PRIME_61 = (1 << 61) - 1
_MAX_HASHES = 4096
_MAX_SEED = (1 << 64) - 1


@dataclass(frozen=True)
class MinHashLSHConfig:
    num_hashes: int
    bands: int
    seed: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_hashes, bool)
            or not isinstance(self.num_hashes, int)
            or not 1 <= self.num_hashes <= _MAX_HASHES
        ):
            raise ValueError(f"num_hashes must be an integer in [1, {_MAX_HASHES}]")
        if (
            isinstance(self.bands, bool)
            or not isinstance(self.bands, int)
            or self.bands <= 0
            or self.bands > self.num_hashes
            or self.num_hashes % self.bands != 0
        ):
            raise ValueError("bands must be positive and divide num_hashes exactly")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= _MAX_SEED
        ):
            raise ValueError("seed must be an unsigned 64-bit integer")

    @property
    def rows_per_band(self) -> int:
        return self.num_hashes // self.bands

    def to_dict(self) -> dict[str, int]:
        return {
            "num_hashes": self.num_hashes,
            "bands": self.bands,
            "rows_per_band": self.rows_per_band,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class MinHashLSHCandidatePair:
    left_id: str
    right_id: str
    colliding_bands: tuple[int, ...]
    signature_matches: int
    signature_size: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.left_id, str)
            or not isinstance(self.right_id, str)
            or not self.left_id
            or self.left_id >= self.right_id
        ):
            raise ValueError("candidate ids must be non-empty and canonical left < right")
        bands = tuple(self.colliding_bands)
        if (
            not bands
            or any(
                isinstance(band, bool) or not isinstance(band, int) or band < 0
                for band in bands
            )
            or bands != tuple(sorted(set(bands)))
        ):
            raise ValueError("colliding_bands must be unique sorted non-negative integers")
        if (
            isinstance(self.signature_size, bool)
            or not isinstance(self.signature_size, int)
            or self.signature_size <= 0
            or isinstance(self.signature_matches, bool)
            or not isinstance(self.signature_matches, int)
            or not 0 <= self.signature_matches <= self.signature_size
        ):
            raise ValueError("signature matches/size are inconsistent")
        object.__setattr__(self, "colliding_bands", bands)

    @property
    def minhash_similarity_estimate(self) -> float:
        return self.signature_matches / self.signature_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "colliding_bands": list(self.colliding_bands),
            "signature_matches": self.signature_matches,
            "signature_size": self.signature_size,
            "minhash_similarity_estimate": self.minhash_similarity_estimate,
        }


@dataclass(frozen=True)
class MinHashLSHCandidateReport:
    config: MinHashLSHConfig
    item_count: int
    possible_pair_count: int
    unique_band_bucket_count: int
    collision_band_bucket_count: int
    candidates: tuple[MinHashLSHCandidatePair, ...]
    item_shingle_fingerprint: str

    def __post_init__(self) -> None:
        _validate_config(self.config)
        for value, name in (
            (self.item_count, "item_count"),
            (self.possible_pair_count, "possible_pair_count"),
            (self.unique_band_bucket_count, "unique_band_bucket_count"),
            (self.collision_band_bucket_count, "collision_band_bucket_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.item_count < 2:
            raise ValueError("item_count must be at least two")
        if self.possible_pair_count != self.item_count * (self.item_count - 1) // 2:
            raise ValueError("possible_pair_count does not match item_count")
        if self.collision_band_bucket_count > self.unique_band_bucket_count:
            raise ValueError("collision bucket count exceeds unique bucket count")
        candidates = tuple(self.candidates)
        if (
            any(not isinstance(item, MinHashLSHCandidatePair) for item in candidates)
            or candidates
            != tuple(sorted(candidates, key=lambda item: (item.left_id, item.right_id)))
            or len({(item.left_id, item.right_id) for item in candidates})
            != len(candidates)
            or len(candidates) > self.possible_pair_count
        ):
            raise ValueError("candidates must be unique and canonically sorted")
        if any(
            item.signature_size != self.config.num_hashes
            or item.colliding_bands[-1] >= self.config.bands
            for item in candidates
        ):
            raise ValueError("candidate signature or band metadata mismatches config")
        _sha256_fingerprint(self.item_shingle_fingerprint)
        object.__setattr__(self, "candidates", candidates)

    @property
    def candidate_pair_count(self) -> int:
        return len(self.candidates)

    @property
    def candidate_fraction(self) -> float:
        if self.possible_pair_count == 0:
            return 0.0
        return self.candidate_pair_count / self.possible_pair_count

    @property
    def manifest_fingerprint(self) -> str:
        return "sha256:" + artifact_fingerprint(
            {
                "version": MINHASH_LSH_VERSION,
                "config": self.config.to_dict(),
                "item_shingle_fingerprint": self.item_shingle_fingerprint,
                "candidates": [candidate.to_dict() for candidate in self.candidates],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MINHASH_LSH_VERSION,
            "config": self.config.to_dict(),
            "item_count": self.item_count,
            "possible_pair_count": self.possible_pair_count,
            "unique_band_bucket_count": self.unique_band_bucket_count,
            "collision_band_bucket_count": self.collision_band_bucket_count,
            "candidate_pair_count": self.candidate_pair_count,
            "candidate_fraction": self.candidate_fraction,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "item_shingle_fingerprint": self.item_shingle_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "scope": {
                "lexical_shingle_candidate_generation": True,
                "exact_jaccard_recheck_included": False,
                "candidate_recall_guaranteed": False,
                "semantic_equivalence_verified": False,
                "cryptographic_authenticity": False,
            },
        }


@dataclass(frozen=True)
class ExactLSHCandidateComparison:
    left_id: str
    right_id: str
    similarity: float
    intersection_size: int
    union_size: int
    passes_threshold: bool
    colliding_bands: tuple[int, ...]
    minhash_similarity_estimate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "similarity": self.similarity,
            "intersection_size": self.intersection_size,
            "union_size": self.union_size,
            "passes_threshold": self.passes_threshold,
            "colliding_bands": list(self.colliding_bands),
            "minhash_similarity_estimate": self.minhash_similarity_estimate,
        }


@dataclass(frozen=True)
class MinHashLSHRecallAudit:
    threshold: float
    possible_pair_count: int
    candidate_pair_count: int
    exact_positive_pair_count: int
    recovered_positive_pair_count: int
    false_positive_candidate_count: int
    missed_exact_positive_pairs: tuple[tuple[str, str], ...]
    item_shingle_fingerprint: str
    candidate_manifest_fingerprint: str

    def __post_init__(self) -> None:
        _probability(self.threshold, "threshold", include_zero=False)
        for value, name in (
            (self.possible_pair_count, "possible_pair_count"),
            (self.candidate_pair_count, "candidate_pair_count"),
            (self.exact_positive_pair_count, "exact_positive_pair_count"),
            (self.recovered_positive_pair_count, "recovered_positive_pair_count"),
            (self.false_positive_candidate_count, "false_positive_candidate_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not (
            self.recovered_positive_pair_count <= self.exact_positive_pair_count
            <= self.possible_pair_count
            and self.candidate_pair_count <= self.possible_pair_count
            and self.false_positive_candidate_count
            == self.candidate_pair_count - self.recovered_positive_pair_count
        ):
            raise ValueError("recall-audit counts are inconsistent")
        missed = tuple(self.missed_exact_positive_pairs)
        if (
            len(missed)
            != self.exact_positive_pair_count - self.recovered_positive_pair_count
            or missed != tuple(sorted(set(missed)))
            or any(
                not isinstance(left, str)
                or not isinstance(right, str)
                or not left
                or left >= right
                for left, right in missed
            )
        ):
            raise ValueError("missed exact-positive pairs are inconsistent")
        _sha256_fingerprint(self.item_shingle_fingerprint)
        _sha256_fingerprint(self.candidate_manifest_fingerprint)
        object.__setattr__(self, "missed_exact_positive_pairs", missed)

    @property
    def candidate_recall(self) -> float | None:
        if self.exact_positive_pair_count == 0:
            return None
        return self.recovered_positive_pair_count / self.exact_positive_pair_count

    @property
    def candidate_precision(self) -> float | None:
        if self.candidate_pair_count == 0:
            return None
        return self.recovered_positive_pair_count / self.candidate_pair_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MINHASH_LSH_VERSION,
            "threshold": self.threshold,
            "possible_pair_count": self.possible_pair_count,
            "candidate_pair_count": self.candidate_pair_count,
            "exact_positive_pair_count": self.exact_positive_pair_count,
            "recovered_positive_pair_count": self.recovered_positive_pair_count,
            "false_positive_candidate_count": self.false_positive_candidate_count,
            "missed_exact_positive_pairs": [
                list(pair) for pair in self.missed_exact_positive_pairs
            ],
            "candidate_recall": self.candidate_recall,
            "candidate_precision": self.candidate_precision,
            "item_shingle_fingerprint": self.item_shingle_fingerprint,
            "candidate_manifest_fingerprint": self.candidate_manifest_fingerprint,
            "scope": {
                "ground_truth_uses_exhaustive_exact_pairs": True,
                "ground_truth_comparison_count": self.possible_pair_count,
                "scalable_validation_method": False,
                "recall_generalizes_beyond_this_snapshot": False,
                "semantic_equivalence_verified": False,
            },
        }


def minhash_signature(
    shingles: frozenset[str], *, config: MinHashLSHConfig
) -> tuple[int, ...]:
    """Build a stable universal-hash MinHash approximation for one non-empty set."""

    _validate_config(config)
    normalized = _validate_shingles(shingles, "shingles")
    return _signature_with_coefficients(
        normalized, coefficients=_hash_coefficients(config)
    )


def generate_minhash_lsh_candidates(
    items: Mapping[str, frozenset[str]], *, config: MinHashLSHConfig
) -> MinHashLSHCandidateReport:
    """Generate deterministic candidate pairs sharing at least one exact band."""

    _validate_config(config)
    snapshot, fingerprint = _item_snapshot(items)
    coefficients = _hash_coefficients(config)
    signatures = {
        item_id: _signature_with_coefficients(shingles, coefficients=coefficients)
        for item_id, shingles in snapshot
    }
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    rows = config.rows_per_band
    for item_id in sorted(signatures):
        signature = signatures[item_id]
        for band in range(config.bands):
            start = band * rows
            buckets[(band, signature[start : start + rows])].append(item_id)

    pair_bands: dict[tuple[str, str], set[int]] = defaultdict(set)
    collision_bucket_count = 0
    for (band, _), member_ids in buckets.items():
        if len(member_ids) < 2:
            continue
        collision_bucket_count += 1
        for left_id, right_id in combinations(sorted(member_ids), 2):
            pair_bands[(left_id, right_id)].add(band)

    candidates = tuple(
        MinHashLSHCandidatePair(
            left_id=left_id,
            right_id=right_id,
            colliding_bands=tuple(sorted(bands)),
            signature_matches=sum(
                left == right
                for left, right in zip(
                    signatures[left_id], signatures[right_id], strict=True
                )
            ),
            signature_size=config.num_hashes,
        )
        for (left_id, right_id), bands in sorted(pair_bands.items())
    )
    item_count = len(snapshot)
    return MinHashLSHCandidateReport(
        config=config,
        item_count=item_count,
        possible_pair_count=item_count * (item_count - 1) // 2,
        unique_band_bucket_count=len(buckets),
        collision_band_bucket_count=collision_bucket_count,
        candidates=candidates,
        item_shingle_fingerprint=fingerprint,
    )


def exact_recheck_lsh_candidates(
    items: Mapping[str, frozenset[str]],
    report: MinHashLSHCandidateReport,
    *,
    threshold: float,
) -> tuple[ExactLSHCandidateComparison, ...]:
    """Compute exact set Jaccard only for LSH candidates."""

    snapshot, fingerprint = _item_snapshot(items)
    _validate_candidate_report(report, fingerprint)
    exact_threshold = _probability(threshold, "threshold", include_zero=False)
    shingle_by_id = dict(snapshot)
    comparisons = []
    for candidate in report.candidates:
        similarity, intersection, union = _jaccard(
            shingle_by_id[candidate.left_id], shingle_by_id[candidate.right_id]
        )
        comparisons.append(
            ExactLSHCandidateComparison(
                left_id=candidate.left_id,
                right_id=candidate.right_id,
                similarity=similarity,
                intersection_size=intersection,
                union_size=union,
                passes_threshold=similarity >= exact_threshold,
                colliding_bands=candidate.colliding_bands,
                minhash_similarity_estimate=candidate.minhash_similarity_estimate,
            )
        )
    return tuple(comparisons)


def audit_minhash_lsh_recall(
    items: Mapping[str, frozenset[str]],
    report: MinHashLSHCandidateReport,
    *,
    threshold: float,
) -> MinHashLSHRecallAudit:
    """Exhaustively measure candidate recall on one snapshot; this is O(N^2)."""

    snapshot, fingerprint = _item_snapshot(items)
    _validate_candidate_report(report, fingerprint)
    exact_threshold = _probability(threshold, "threshold", include_zero=False)
    candidate_pairs = {
        (candidate.left_id, candidate.right_id) for candidate in report.candidates
    }
    exact_positive_pairs = {
        (left_id, right_id)
        for (left_id, left), (right_id, right) in combinations(snapshot, 2)
        if _jaccard(left, right)[0] >= exact_threshold
    }
    recovered = candidate_pairs & exact_positive_pairs
    missed = tuple(sorted(exact_positive_pairs - candidate_pairs))
    return MinHashLSHRecallAudit(
        threshold=exact_threshold,
        possible_pair_count=report.possible_pair_count,
        candidate_pair_count=report.candidate_pair_count,
        exact_positive_pair_count=len(exact_positive_pairs),
        recovered_positive_pair_count=len(recovered),
        false_positive_candidate_count=len(candidate_pairs - exact_positive_pairs),
        missed_exact_positive_pairs=missed,
        item_shingle_fingerprint=fingerprint,
        candidate_manifest_fingerprint=report.manifest_fingerprint,
    )


def lsh_candidate_probability(
    similarity: float, *, bands: int, rows_per_band: int
) -> float:
    """Ideal independent-Minhash probability of at least one band collision."""

    exact_similarity = _probability(similarity, "similarity", include_zero=True)
    for value, name in ((bands, "bands"), (rows_per_band, "rows_per_band")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return 1.0 - (1.0 - exact_similarity**rows_per_band) ** bands


def _validate_config(config: MinHashLSHConfig) -> None:
    if not isinstance(config, MinHashLSHConfig):
        raise TypeError("config must be MinHashLSHConfig")


def _item_snapshot(
    items: Mapping[str, frozenset[str]],
) -> tuple[tuple[tuple[str, frozenset[str]], ...], str]:
    if not isinstance(items, Mapping) or len(items) < 2:
        raise ValueError("items must be a mapping with at least two entries")
    snapshot = []
    for item_id, shingles in items.items():
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("item ids must be non-empty strings")
        snapshot.append((item_id, _validate_shingles(shingles, f"items[{item_id!r}]")))
    snapshot.sort(key=lambda item: item[0])
    fingerprint = "sha256:" + artifact_fingerprint(
        {
            "items": [
                {"item_id": item_id, "shingles": sorted(shingles)}
                for item_id, shingles in snapshot
            ]
        }
    )
    return tuple(snapshot), fingerprint


def _validate_shingles(value: frozenset[str], name: str) -> frozenset[str]:
    if not isinstance(value, frozenset) or not value:
        raise ValueError(f"{name} must be a non-empty frozenset")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    return value


def _stable_shingle_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest(), "big") % (
        _MERSENNE_PRIME_61
    )


def _coefficient(seed: int, index: int, label: str) -> int:
    material = f"{MINHASH_LSH_VERSION}:{seed}:{index}:{label}".encode()
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def _hash_coefficients(config: MinHashLSHConfig) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            1
            + _coefficient(config.seed, index, "a")
            % (_MERSENNE_PRIME_61 - 1),
            _coefficient(config.seed, index, "b") % _MERSENNE_PRIME_61,
        )
        for index in range(config.num_hashes)
    )


def _signature_with_coefficients(
    shingles: frozenset[str], *, coefficients: tuple[tuple[int, int], ...]
) -> tuple[int, ...]:
    hashed_shingles = tuple(_stable_shingle_hash(value) for value in shingles)
    return tuple(
        min(
            (coefficient_a * value + coefficient_b) % _MERSENNE_PRIME_61
            for value in hashed_shingles
        )
        for coefficient_a, coefficient_b in coefficients
    )


def _jaccard(
    left: frozenset[str], right: frozenset[str]
) -> tuple[float, int, int]:
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union, intersection, union


def _probability(value: float, name: str, *, include_zero: bool) -> float:
    lower_bound = 0 if include_zero else 0.0
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < lower_bound
        or (not include_zero and float(value) == 0)
        or float(value) > 1
    ):
        interval = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{name} must be finite and in {interval}")
    return float(value)


def _validate_candidate_report(
    report: MinHashLSHCandidateReport, fingerprint: str
) -> None:
    if not isinstance(report, MinHashLSHCandidateReport):
        raise TypeError("report must be MinHashLSHCandidateReport")
    if report.item_shingle_fingerprint != fingerprint:
        raise ValueError("candidate report is bound to different item shingles")


def _sha256_fingerprint(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError("fingerprint must be canonical sha256:<lowercase-hex>")
