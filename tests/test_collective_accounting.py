from __future__ import annotations

from fractions import Fraction

import pytest

pytestmark = pytest.mark.formula


def _ring_rounds(participants: int) -> int:
    if participants < 1:
        raise ValueError("participants must be at least 1")
    return 2 * (participants - 1)


def _ring_sent_per_rank(participants: int, input_bytes: int) -> Fraction:
    if participants < 1:
        raise ValueError("participants must be at least 1")
    if input_bytes < 0:
        raise ValueError("input_bytes must be non-negative")
    return Fraction(2 * (participants - 1) * input_bytes, participants)


def test_ring_single_rank_has_no_rounds_or_traffic() -> None:
    assert _ring_rounds(1) == 0
    assert _ring_sent_per_rank(1, 4 * 1024**2) == 0


def test_ring_four_rank_per_rank_payload_uses_partitioned_chunks() -> None:
    input_mib = 96

    assert _ring_rounds(4) == 6
    assert _ring_sent_per_rank(4, input_mib) == 144
    assert 2 * (4 - 1) * input_mib == 576


def test_two_node_eight_rail_cross_domain_ledger() -> None:
    shard_mib = Fraction(192, 8)
    per_pair_bidirectional_mib = 2 * shard_mib

    assert shard_mib == 24
    assert per_pair_bidirectional_mib == 48
    assert 8 * per_pair_bidirectional_mib == 384


@pytest.mark.parametrize(
    ("oversubscription", "expected_limit"),
    [(1, Fraction(1, 1)), (3, Fraction(1, 3))],
)
def test_uniform_cut_model_limits_cross_leaf_fraction(
    oversubscription: int,
    expected_limit: Fraction,
) -> None:
    assert Fraction(1, oversubscription) == expected_limit


@pytest.mark.parametrize("participants", [0, -1])
def test_ring_formula_rejects_invalid_participant_counts(participants: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _ring_rounds(participants)
