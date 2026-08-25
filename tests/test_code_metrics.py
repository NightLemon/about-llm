from __future__ import annotations

import pytest

from about_llm.evaluation import pass_at_k, summarize_candidate_selection

pytestmark = pytest.mark.formula


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


def test_candidate_selection_separates_oracle_coverage_from_actual_choice() -> None:
    summary = summarize_candidate_selection(
        candidate_correctness=(
            (False, True, False),
            (True, False, False),
            (False, False, False),
            (False, True, True),
        ),
        selected_indices=(0, 0, 1, 1),
    )

    assert summary.task_count == 4
    assert summary.k == 3
    assert summary.oracle_success_count == 3
    assert summary.selected_success_count == 2
    assert summary.oracle_at_k == 3 / 4
    assert summary.selected_at_k == 1 / 2
    assert summary.selector_recall_on_oracle_positive == 2 / 3


def test_candidate_selection_reports_no_selector_recall_without_oracle_positive() -> None:
    summary = summarize_candidate_selection(
        candidate_correctness=((False, False),),
        selected_indices=(0,),
    )

    assert summary.oracle_at_k == 0.0
    assert summary.selected_at_k == 0.0
    assert summary.selector_recall_on_oracle_positive is None


@pytest.mark.parametrize(
    ("candidate_correctness", "selected_indices", "error", "message"),
    [
        ((), (), ValueError, "at least one task"),
        (((),), (0,), ValueError, "at least one candidate"),
        (((True,), (True, False)), (0, 0), ValueError, "same number"),
        (((1, False),), (0,), TypeError, "booleans"),
        (((True, False),), (), ValueError, "one index per task"),
        (((True, False),), (True,), TypeError, "integers"),
        (((True, False),), (2,), ValueError, "outside"),
    ],
)
def test_candidate_selection_rejects_ambiguous_shapes(
    candidate_correctness: tuple[tuple[object, ...], ...],
    selected_indices: tuple[object, ...],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        summarize_candidate_selection(
            candidate_correctness=candidate_correctness,  # type: ignore[arg-type]
            selected_indices=selected_indices,  # type: ignore[arg-type]
        )
