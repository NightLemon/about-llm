from __future__ import annotations

import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from about_llm.finetuning import (
    CategoricalMicrobatch,
    CategoricalTokenRecord,
    analyze_masked_token_gradient_accumulation,
)

ROOT = Path(__file__).resolve().parents[1]
TOY = (
    ROOT
    / "projects"
    / "single-gpu-finetuning"
    / "gradient_accumulation_toy.py"
)


def fixture_microbatches() -> tuple[CategoricalMicrobatch, ...]:
    return (
        CategoricalMicrobatch(
            "short",
            (
                CategoricalTokenRecord("short.valid", (9, 1), 0),
                CategoricalTokenRecord("short.padding", (1, 1), None),
            ),
        ),
        CategoricalMicrobatch(
            "long",
            (
                CategoricalTokenRecord("long.1", (4, 1), 1),
                CategoricalTokenRecord("long.2", (4, 1), 1),
                CategoricalTokenRecord("long.3", (4, 1), 1),
                CategoricalTokenRecord("long.padding", (1, 1), None),
            ),
        ),
    )


def test_exact_oracle_separates_token_mean_from_equal_microbatch_mean() -> None:
    result = analyze_masked_token_gradient_accumulation(fixture_microbatches())

    assert result.microbatch_count == 2
    assert result.valid_token_count == 4
    assert result.ignored_token_count == 2
    assert [item.valid_token_count for item in result.microbatches] == [1, 3]
    assert [item.correct_global_weight for item in result.microbatches] == [
        Fraction(1, 4),
        Fraction(3, 4),
    ]
    assert [item.naive_equal_microbatch_weight for item in result.microbatches] == [
        Fraction(1, 2),
        Fraction(1, 2),
    ]
    assert result.full_batch_class_aggregate_logit_gradient == (
        Fraction(23, 40),
        Fraction(-23, 40),
    )
    assert result.count_scaled_accumulated_class_aggregate_logit_gradient == (
        Fraction(23, 40),
        Fraction(-23, 40),
    )
    assert result.naive_equal_microbatch_class_aggregate_logit_gradient == (
        Fraction(7, 20),
        Fraction(-7, 20),
    )
    assert result.naive_minus_full_class_aggregate_logit_gradient == (
        Fraction(-9, 40),
        Fraction(9, 40),
    )


def test_exact_loss_uses_valid_tokens_not_microbatch_count() -> None:
    result = analyze_masked_token_gradient_accumulation(fixture_microbatches())
    expected_full = (-math.log(0.9) - 3 * math.log(0.2)) / 4
    expected_naive = (-math.log(0.9) - math.log(0.2)) / 2

    assert result.full_batch_token_mean_negative_log_likelihood == pytest.approx(
        expected_full
    )
    assert result.count_scaled_accumulated_negative_log_likelihood == pytest.approx(
        expected_full
    )
    assert result.naive_equal_microbatch_negative_log_likelihood == pytest.approx(
        expected_naive
    )
    assert result.naive_negative_log_likelihood_bias == pytest.approx(
        expected_naive - expected_full
    )


def test_token_gradient_is_exact_softmax_minus_one_hot_and_ignore_is_none() -> None:
    supervised = CategoricalTokenRecord("supervised", (2, 3, 5), 1)
    ignored = CategoricalTokenRecord("ignored", (2, 3, 5), None)

    assert supervised.probabilities == (
        Fraction(1, 5),
        Fraction(3, 10),
        Fraction(1, 2),
    )
    assert supervised.logit_gradient == (
        Fraction(1, 5),
        Fraction(-7, 10),
        Fraction(1, 2),
    )
    assert ignored.logit_gradient is None
    assert ignored.negative_log_likelihood is None


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("bad id", (1, 1), 0), "token_id"),
        (("x", [1, 1], 0), "tuple"),
        (("x", (1,), 0), "vocabulary"),
        (("x", (True, 1), 0), "integers"),
        (("x", (-1, 1), 0), "integers"),
        (("x", (1_000_001, 1), 0), "integers"),
        (("x", (0, 0), 0), "all be zero"),
        (("x", (1, 1), True), "target_index"),
        (("x", (1, 1), 2), "target_index"),
        (("x", (1, 0), 1), "positive probability"),
    ],
)
def test_token_record_rejects_invalid_fields(
    args: tuple[object, object, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CategoricalTokenRecord(*args)  # type: ignore[arg-type]


def test_microbatch_rejects_empty_all_ignored_duplicate_or_wrong_tokens() -> None:
    valid = CategoricalTokenRecord("valid", (1, 1), 0)
    ignored = CategoricalTokenRecord("ignored", (1, 1), None)

    with pytest.raises(ValueError, match="tokens must be a tuple"):
        CategoricalMicrobatch("batch", ())
    with pytest.raises(ValueError, match="supervised"):
        CategoricalMicrobatch("batch", (ignored,))
    with pytest.raises(ValueError, match="unique"):
        CategoricalMicrobatch("batch", (valid, valid))
    with pytest.raises(ValueError, match="CategoricalTokenRecord"):
        CategoricalMicrobatch("batch", ("wrong",))  # type: ignore[arg-type]


def test_analysis_rejects_empty_duplicate_mixed_vocab_or_wrong_types() -> None:
    valid = CategoricalTokenRecord("valid", (1, 1), 0)
    first = CategoricalMicrobatch("first", (valid,))

    with pytest.raises(ValueError, match="at least one"):
        analyze_masked_token_gradient_accumulation(())
    with pytest.raises(ValueError, match="unique"):
        analyze_masked_token_gradient_accumulation((first, first))
    with pytest.raises(ValueError, match="CategoricalMicrobatch"):
        analyze_masked_token_gradient_accumulation(("wrong",))  # type: ignore[arg-type]
    second = CategoricalMicrobatch(
        "second",
        (CategoricalTokenRecord("other", (1, 1, 1), 0),),
    )
    with pytest.raises(ValueError, match="vocabulary"):
        analyze_masked_token_gradient_accumulation((first, second))


def test_analysis_rejects_global_duplicate_token_ids_and_resource_caps() -> None:
    first = CategoricalMicrobatch(
        "first",
        (CategoricalTokenRecord("same", (1, 1), 0),),
    )
    second = CategoricalMicrobatch(
        "second",
        (CategoricalTokenRecord("same", (1, 1), 1),),
    )
    with pytest.raises(ValueError, match="globally unique"):
        analyze_masked_token_gradient_accumulation((first, second))

    too_many = tuple(
        CategoricalMicrobatch(
            f"batch-{index}",
            (CategoricalTokenRecord(f"token-{index}", (1, 1), 0),),
        )
        for index in range(257)
    )
    with pytest.raises(ValueError, match="cannot exceed 256"):
        analyze_masked_token_gradient_accumulation(too_many)


def test_report_dict_preserves_exact_fraction_payloads() -> None:
    payload = analyze_masked_token_gradient_accumulation(
        fixture_microbatches()
    ).to_dict()

    assert payload["microbatches"][0]["correct_global_weight"] == {
        "numerator": 1,
        "denominator": 4,
        "decimal": 0.25,
    }
    assert payload["full_batch_class_aggregate_logit_gradient"] == [
        {"numerator": 23, "denominator": 40, "decimal": 0.575},
        {"numerator": -23, "denominator": 40, "decimal": -0.575},
    ]


def test_toy_executes_pytorch_backward_and_reports_scope() -> None:
    completed = subprocess.run(
        [sys.executable, str(TOY)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(completed.stdout)

    assert completed.stderr == ""
    assert report["implementation"] == "about-llm.gradient-accumulation-toy.v1"
    assert report["observations"] == {
        "count_scaled_matches_full_exactly_in_fraction_oracle": True,
        "equal_microbatch_mean_changes_the_objective": True,
        "ignored_positions_have_zero_gradient": True,
        "pytorch_count_scaled_gradient_matches_full": True,
        "pytorch_naive_gradient_differs_from_full": True,
        "valid_token_counts_are_one_and_three": True,
    }
    exact = report["exact_oracle"]
    assert exact["valid_token_count"] == 4
    assert exact["ignored_token_count"] == 3
    assert exact["full_batch_class_aggregate_logit_gradient"] == [
        {"numerator": 23, "denominator": 40, "decimal": 0.575},
        {"numerator": -23, "denominator": 40, "decimal": -0.575},
    ]
    autograd = report["pytorch_autograd"]
    assert autograd["full_vs_count_scaled_max_abs_gradient_error"] == 0.0
    assert autograd["full_vs_naive_max_abs_gradient_difference"] > 0
    assert autograd["ignored_row_max_abs_gradient"] == 0.0
    assert autograd["full_class_aggregate_gradient"] == pytest.approx(
        [0.575, -0.575]
    )
    assert autograd["naive_equal_microbatch_class_aggregate_gradient"] == (
        pytest.approx([0.35, -0.35])
    )
    assert report["scope"] == {
        "amp_cuda_gpu_memory_throughput_or_quality_measured": False,
        "authored_probabilities_targets_and_padding_executed": True,
        "ddp_fsdp_zero_collective_or_no_sync_executed": False,
        "dropout_batchnorm_or_stochastic_model_equivalence_proved": False,
        "exact_fraction_logit_gradient_oracle_executed": True,
        "optimizer_step_or_parameter_update_executed": False,
        "pytorch_float64_cross_entropy_backward_executed": True,
        "target_llm_tokenizer_dataset_or_training_run_executed": False,
    }
