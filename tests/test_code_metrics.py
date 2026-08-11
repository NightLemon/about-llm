from __future__ import annotations

import pytest

from about_llm.evaluation import pass_at_k


def test_pass_at_k_matches_combinatorial_definition() -> None:
    assert pass_at_k(num_samples=10, num_correct=2, k=1) == pytest.approx(0.2)
    assert pass_at_k(num_samples=10, num_correct=2, k=2) == pytest.approx(17 / 45)
    assert pass_at_k(num_samples=10, num_correct=2, k=9) == 1.0


def test_pass_at_k_boundary_cases() -> None:
    assert pass_at_k(num_samples=4, num_correct=0, k=4) == 0.0
    assert pass_at_k(num_samples=4, num_correct=4, k=1) == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_samples": 0, "num_correct": 0, "k": 1}, "num_samples"),
        ({"num_samples": 4, "num_correct": -1, "k": 1}, "num_correct"),
        ({"num_samples": 4, "num_correct": 5, "k": 1}, "num_correct"),
        ({"num_samples": 4, "num_correct": 1, "k": 0}, "k"),
        ({"num_samples": 4, "num_correct": 1, "k": 5}, "k"),
    ],
)
def test_pass_at_k_rejects_invalid_ranges(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pass_at_k(**kwargs)


@pytest.mark.parametrize("bad", [True, 1.5, "4"])
def test_pass_at_k_rejects_non_integer_inputs(bad: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        pass_at_k(num_samples=bad, num_correct=1, k=1)  # type: ignore[arg-type]
