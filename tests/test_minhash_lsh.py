from __future__ import annotations

import math
from pathlib import Path

import pytest

from about_llm.finetuning.minhash_lsh import (
    MinHashLSHConfig,
    audit_minhash_lsh_recall,
    exact_recheck_lsh_candidates,
    generate_minhash_lsh_candidates,
    lsh_candidate_probability,
    minhash_signature,
)

ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides: int) -> MinHashLSHConfig:
    values = {"num_hashes": 32, "bands": 8, "seed": 17}
    values.update(overrides)
    return MinHashLSHConfig(**values)


def test_identical_sets_collide_in_every_band_and_recheck_exactly() -> None:
    items = {
        "a": frozenset({"one", "two", "three"}),
        "b": frozenset({"one", "two", "three"}),
        "c": frozenset({"other"}),
    }
    report = generate_minhash_lsh_candidates(items, config=_config())

    assert report.possible_pair_count == 3
    assert report.candidate_pair_count == 1
    assert report.candidates[0].left_id == "a"
    assert report.candidates[0].right_id == "b"
    assert report.candidates[0].colliding_bands == tuple(range(8))
    assert report.candidates[0].signature_matches == 32
    assert report.candidates[0].minhash_similarity_estimate == 1
    exact = exact_recheck_lsh_candidates(items, report, threshold=0.9)
    assert exact[0].similarity == 1
    assert exact[0].intersection_size == exact[0].union_size == 3
    assert exact[0].passes_threshold is True


def test_candidate_generation_is_stable_under_mapping_order() -> None:
    first = {
        "z": frozenset({"a", "b", "c"}),
        "a": frozenset({"a", "b", "c"}),
        "m": frozenset({"x", "y"}),
    }
    second = {key: first[key] for key in reversed(tuple(first))}

    left = generate_minhash_lsh_candidates(first, config=_config())
    right = generate_minhash_lsh_candidates(second, config=_config())

    assert left.to_dict() == right.to_dict()
    assert left.manifest_fingerprint == right.manifest_fingerprint


def test_signature_is_seeded_and_does_not_use_python_hash_randomization() -> None:
    shingles = frozenset({"中文", "alpha", "beta"})

    first = minhash_signature(shingles, config=_config(seed=17))
    repeated = minhash_signature(shingles, config=_config(seed=17))
    different_seed = minhash_signature(shingles, config=_config(seed=18))

    assert first == repeated
    assert first != different_seed
    assert len(first) == 32
    assert all(0 <= value < (1 << 61) - 1 for value in first)


def test_exhaustive_recall_audit_reports_snapshot_specific_false_negatives() -> None:
    items = {
        "a": frozenset({"a", "b", "c", "d", "e"}),
        "b": frozenset({"a", "b", "c", "d", "x"}),
        "c": frozenset({"other"}),
    }
    report = generate_minhash_lsh_candidates(
        items, config=MinHashLSHConfig(num_hashes=1, bands=1, seed=0)
    )
    recall = audit_minhash_lsh_recall(items, report, threshold=2 / 3)

    assert recall.exact_positive_pair_count == 1
    assert recall.recovered_positive_pair_count == 0
    assert recall.candidate_recall == 0
    assert recall.missed_exact_positive_pairs == (("a", "b"),)
    assert recall.to_dict()["scope"]["scalable_validation_method"] is False
    assert recall.candidate_manifest_fingerprint == report.manifest_fingerprint


def test_recheck_rejects_candidate_report_for_different_items() -> None:
    items = {
        "a": frozenset({"same"}),
        "b": frozenset({"same"}),
    }
    report = generate_minhash_lsh_candidates(items, config=_config())
    changed = {**items, "b": frozenset({"changed"})}

    with pytest.raises(ValueError, match="different item shingles"):
        exact_recheck_lsh_candidates(changed, report, threshold=0.9)


def test_ideal_band_collision_probability_matches_formula() -> None:
    probability = lsh_candidate_probability(0.8, bands=20, rows_per_band=5)

    assert probability == pytest.approx(1 - (1 - 0.8**5) ** 20)
    assert lsh_candidate_probability(0, bands=20, rows_per_band=5) == 0
    assert lsh_candidate_probability(1, bands=20, rows_per_band=5) == 1


@pytest.mark.parametrize(
    "config",
    [
        lambda: MinHashLSHConfig(num_hashes=0, bands=1, seed=0),
        lambda: MinHashLSHConfig(num_hashes=8, bands=3, seed=0),
        lambda: MinHashLSHConfig(num_hashes=8, bands=9, seed=0),
        lambda: MinHashLSHConfig(num_hashes=8, bands=2, seed=-1),
    ],
)
def test_invalid_lsh_config_is_rejected(config) -> None:
    with pytest.raises(ValueError):
        config()


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, math.nan, True])
def test_exact_recheck_requires_finite_positive_threshold(threshold: float) -> None:
    items = {"a": frozenset({"x"}), "b": frozenset({"x"})}
    report = generate_minhash_lsh_candidates(items, config=_config())

    with pytest.raises(ValueError, match="threshold"):
        exact_recheck_lsh_candidates(items, report, threshold=threshold)


@pytest.mark.parametrize(
    "items",
    [
        {},
        {"only": frozenset({"x"})},
        {"": frozenset({"x"}), "b": frozenset({"y"})},
        {"a": frozenset(), "b": frozenset({"y"})},
        {"a": {"x"}, "b": frozenset({"y"})},
    ],
)
def test_invalid_item_snapshot_is_rejected(items) -> None:
    with pytest.raises(ValueError):
        generate_minhash_lsh_candidates(items, config=_config())


