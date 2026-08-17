"""Exact masked-token weighting oracles for local and DDP accumulation."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

MAX_GRADIENT_MICROBATCHES = 256
MAX_GRADIENT_TOKENS_PER_MICROBATCH = 4_096
MAX_GRADIENT_VOCABULARY = 4_096
MAX_GRADIENT_WEIGHT = 1_000_000
MAX_GRADIENT_ELEMENTS = 1_000_000
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _fraction_payload(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _fraction_vector_payload(
    values: tuple[Fraction, ...],
) -> list[dict[str, int | float]]:
    return [_fraction_payload(value) for value in values]


@dataclass(frozen=True, slots=True)
class CategoricalTokenRecord:
    """One authored token distribution and optional supervised target."""

    token_id: str
    probability_weights: tuple[int, ...]
    target_index: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.token_id, str) or _IDENTIFIER.fullmatch(
            self.token_id
        ) is None:
            raise ValueError(
                "token_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
            )
        if type(self.probability_weights) is not tuple or not (
            2 <= len(self.probability_weights) <= MAX_GRADIENT_VOCABULARY
        ):
            raise ValueError(
                "probability_weights must be a tuple with vocabulary size in "
                f"[2, {MAX_GRADIENT_VOCABULARY}]"
            )
        for weight in self.probability_weights:
            if (
                type(weight) is not int
                or weight < 0
                or weight > MAX_GRADIENT_WEIGHT
            ):
                raise ValueError(
                    "probability weights must be integers in "
                    f"[0, {MAX_GRADIENT_WEIGHT}]"
                )
        if sum(self.probability_weights) <= 0:
            raise ValueError("probability weights cannot all be zero")
        if self.target_index is not None and (
            type(self.target_index) is not int
            or self.target_index < 0
            or self.target_index >= len(self.probability_weights)
        ):
            raise ValueError(
                "target_index must be None or an integer in the vocabulary"
            )
        if (
            self.target_index is not None
            and self.probability_weights[self.target_index] == 0
        ):
            raise ValueError("a supervised target must have positive probability weight")

    @property
    def probabilities(self) -> tuple[Fraction, ...]:
        total = sum(self.probability_weights)
        return tuple(Fraction(weight, total) for weight in self.probability_weights)

    @property
    def logit_gradient(self) -> tuple[Fraction, ...] | None:
        """Return exact ``softmax_probability - one_hot_target``."""

        if self.target_index is None:
            return None
        return tuple(
            probability - (1 if index == self.target_index else 0)
            for index, probability in enumerate(self.probabilities)
        )

    @property
    def negative_log_likelihood(self) -> float | None:
        if self.target_index is None:
            return None
        return -math.log(float(self.probabilities[self.target_index]))


@dataclass(frozen=True, slots=True)
class CategoricalMicrobatch:
    """One micro-batch containing supervised and ignored token positions."""

    microbatch_id: str
    tokens: tuple[CategoricalTokenRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.microbatch_id, str) or _IDENTIFIER.fullmatch(
            self.microbatch_id
        ) is None:
            raise ValueError(
                "microbatch_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
            )
        if type(self.tokens) is not tuple or not (
            1 <= len(self.tokens) <= MAX_GRADIENT_TOKENS_PER_MICROBATCH
        ):
            raise ValueError(
                "tokens must be a tuple with length in "
                f"[1, {MAX_GRADIENT_TOKENS_PER_MICROBATCH}]"
            )
        if any(not isinstance(token, CategoricalTokenRecord) for token in self.tokens):
            raise ValueError("all tokens must be CategoricalTokenRecord instances")
        token_ids = [token.token_id for token in self.tokens]
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("token_id values must be unique within a micro-batch")
        if not any(token.target_index is not None for token in self.tokens):
            raise ValueError("each micro-batch must contain at least one supervised token")


@dataclass(frozen=True, slots=True)
class MicrobatchGradientContribution:
    """Loss and gradient summary for one micro-batch."""

    microbatch_id: str
    valid_token_count: int
    ignored_token_count: int
    correct_global_weight: Fraction
    naive_equal_microbatch_weight: Fraction
    token_mean_negative_log_likelihood: float
    token_mean_class_aggregate_logit_gradient: tuple[Fraction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "microbatch_id": self.microbatch_id,
            "valid_token_count": self.valid_token_count,
            "ignored_token_count": self.ignored_token_count,
            "correct_global_weight": _fraction_payload(self.correct_global_weight),
            "naive_equal_microbatch_weight": _fraction_payload(
                self.naive_equal_microbatch_weight
            ),
            "token_mean_negative_log_likelihood": (
                self.token_mean_negative_log_likelihood
            ),
            "token_mean_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.token_mean_class_aggregate_logit_gradient
                )
            ),
        }


@dataclass(frozen=True, slots=True)
class GradientAccumulationAnalysis:
    """Exact comparison of token weighting across micro-batch reductions."""

    vocabulary_size: int
    microbatch_count: int
    valid_token_count: int
    ignored_token_count: int
    microbatches: tuple[MicrobatchGradientContribution, ...]
    full_batch_token_mean_negative_log_likelihood: float
    count_scaled_accumulated_negative_log_likelihood: float
    naive_equal_microbatch_negative_log_likelihood: float
    naive_negative_log_likelihood_bias: float
    full_batch_class_aggregate_logit_gradient: tuple[Fraction, ...]
    count_scaled_accumulated_class_aggregate_logit_gradient: tuple[Fraction, ...]
    naive_equal_microbatch_class_aggregate_logit_gradient: tuple[Fraction, ...]
    naive_minus_full_class_aggregate_logit_gradient: tuple[Fraction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "vocabulary_size": self.vocabulary_size,
            "microbatch_count": self.microbatch_count,
            "valid_token_count": self.valid_token_count,
            "ignored_token_count": self.ignored_token_count,
            "microbatches": [microbatch.to_dict() for microbatch in self.microbatches],
            "full_batch_token_mean_negative_log_likelihood": (
                self.full_batch_token_mean_negative_log_likelihood
            ),
            "count_scaled_accumulated_negative_log_likelihood": (
                self.count_scaled_accumulated_negative_log_likelihood
            ),
            "naive_equal_microbatch_negative_log_likelihood": (
                self.naive_equal_microbatch_negative_log_likelihood
            ),
            "naive_negative_log_likelihood_bias": (
                self.naive_negative_log_likelihood_bias
            ),
            "full_batch_class_aggregate_logit_gradient": _fraction_vector_payload(
                self.full_batch_class_aggregate_logit_gradient
            ),
            "count_scaled_accumulated_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.count_scaled_accumulated_class_aggregate_logit_gradient
                )
            ),
            "naive_equal_microbatch_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.naive_equal_microbatch_class_aggregate_logit_gradient
                )
            ),
            "naive_minus_full_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.naive_minus_full_class_aggregate_logit_gradient
                )
            ),
        }


@dataclass(frozen=True, slots=True)
class DDPTokenMeanAnalysis:
    """Exact default-DDP gradient averaging for one shard per rank."""

    data_parallel_world_size: int
    valid_token_count: int
    ignored_token_count: int
    valid_token_counts_by_rank: tuple[int, ...]
    correct_local_loss_sum_scale: Fraction
    missing_world_size_local_loss_sum_scale: Fraction
    rank_local_sum_class_aggregate_logit_gradients: tuple[
        tuple[Fraction, ...], ...
    ]
    full_batch_class_aggregate_logit_gradient: tuple[Fraction, ...]
    correctly_scaled_default_ddp_class_aggregate_logit_gradient: tuple[
        Fraction, ...
    ]
    missing_world_size_default_ddp_class_aggregate_logit_gradient: tuple[
        Fraction, ...
    ]
    equal_rank_local_mean_class_aggregate_logit_gradient: tuple[Fraction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "data_parallel_world_size": self.data_parallel_world_size,
            "valid_token_count": self.valid_token_count,
            "ignored_token_count": self.ignored_token_count,
            "valid_token_counts_by_rank": list(self.valid_token_counts_by_rank),
            "correct_local_loss_sum_scale": _fraction_payload(
                self.correct_local_loss_sum_scale
            ),
            "missing_world_size_local_loss_sum_scale": _fraction_payload(
                self.missing_world_size_local_loss_sum_scale
            ),
            "rank_local_sum_class_aggregate_logit_gradients": [
                _fraction_vector_payload(gradient)
                for gradient in self.rank_local_sum_class_aggregate_logit_gradients
            ],
            "full_batch_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.full_batch_class_aggregate_logit_gradient
                )
            ),
            "correctly_scaled_default_ddp_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.correctly_scaled_default_ddp_class_aggregate_logit_gradient
                )
            ),
            "missing_world_size_default_ddp_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.missing_world_size_default_ddp_class_aggregate_logit_gradient
                )
            ),
            "equal_rank_local_mean_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.equal_rank_local_mean_class_aggregate_logit_gradient
                )
            ),
        }


@dataclass(frozen=True, slots=True)
class DDPGradientAccumulationAnalysis:
    """Exact DDP accumulation and plain-SGD algebra for one update window."""

    data_parallel_world_size: int
    accumulation_steps: int
    valid_token_count: int
    ignored_token_count: int
    valid_token_counts_by_rank_and_microbatch: tuple[tuple[int, ...], ...]
    valid_token_counts_by_rank: tuple[int, ...]
    correct_local_loss_sum_scale: Fraction
    rank_microbatch_loss_sum_class_aggregate_logit_gradients: tuple[
        tuple[tuple[Fraction, ...], ...], ...
    ]
    rank_accumulated_loss_sum_class_aggregate_logit_gradients: tuple[
        tuple[Fraction, ...], ...
    ]
    full_batch_class_aggregate_logit_gradient: tuple[Fraction, ...]
    one_sync_after_accumulation_class_aggregate_logit_gradient: tuple[
        Fraction, ...
    ]
    sync_every_microbatch_class_aggregate_logit_gradient: tuple[Fraction, ...]
    unclipped_sgd_learning_rate: Fraction
    unclipped_sgd_parameter_delta: tuple[Fraction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "data_parallel_world_size": self.data_parallel_world_size,
            "accumulation_steps": self.accumulation_steps,
            "valid_token_count": self.valid_token_count,
            "ignored_token_count": self.ignored_token_count,
            "valid_token_counts_by_rank_and_microbatch": [
                list(counts)
                for counts in self.valid_token_counts_by_rank_and_microbatch
            ],
            "valid_token_counts_by_rank": list(self.valid_token_counts_by_rank),
            "correct_local_loss_sum_scale": _fraction_payload(
                self.correct_local_loss_sum_scale
            ),
            "rank_microbatch_loss_sum_class_aggregate_logit_gradients": [
                [_fraction_vector_payload(gradient) for gradient in rank]
                for rank in (
                    self.rank_microbatch_loss_sum_class_aggregate_logit_gradients
                )
            ],
            "rank_accumulated_loss_sum_class_aggregate_logit_gradients": [
                _fraction_vector_payload(gradient)
                for gradient in (
                    self.rank_accumulated_loss_sum_class_aggregate_logit_gradients
                )
            ],
            "full_batch_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.full_batch_class_aggregate_logit_gradient
                )
            ),
            "one_sync_after_accumulation_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.one_sync_after_accumulation_class_aggregate_logit_gradient
                )
            ),
            "sync_every_microbatch_class_aggregate_logit_gradient": (
                _fraction_vector_payload(
                    self.sync_every_microbatch_class_aggregate_logit_gradient
                )
            ),
            "unclipped_sgd_learning_rate": _fraction_payload(
                self.unclipped_sgd_learning_rate
            ),
            "unclipped_sgd_parameter_delta": _fraction_vector_payload(
                self.unclipped_sgd_parameter_delta
            ),
        }


def _sum_fraction_vectors(
    vectors: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, ...]:
    return tuple(
        sum((vector[index] for vector in vectors), start=Fraction(0, 1))
        for index in range(len(vectors[0]))
    )


def _mean_fraction_vectors(
    vectors: tuple[tuple[Fraction, ...], ...],
) -> tuple[Fraction, ...]:
    denominator = len(vectors)
    return tuple(
        sum((vector[index] for vector in vectors), start=Fraction(0, 1))
        / denominator
        for index in range(len(vectors[0]))
    )


def analyze_masked_token_gradient_accumulation(
    microbatches: Iterable[CategoricalMicrobatch],
) -> GradientAccumulationAnalysis:
    """Compare correct token-mean accumulation with equal micro-batch means.

    The correct reduction gives every supervised token coefficient ``1/N``.
    Equivalently, a micro-batch mean must be weighted by ``n_i/N``. Averaging
    ``M`` micro-batch means equally instead gives each token in micro-batch ``i``
    coefficient ``1/(M*n_i)`` and changes the objective when valid-token counts
    differ. Ignored positions never enter either denominator.

    Exact gradients here are class-aggregated gradients with respect to authored
    token logits. The coefficient identity applies through the chain rule to any
    shared model parameters, but this function does not run a model, backward,
    optimizer, stochastic layer, distributed collective, AMP, or CUDA kernel.
    """

    microbatch_tuple = tuple(microbatches)
    if not microbatch_tuple:
        raise ValueError("at least one micro-batch is required")
    if len(microbatch_tuple) > MAX_GRADIENT_MICROBATCHES:
        raise ValueError(
            f"micro-batch count cannot exceed {MAX_GRADIENT_MICROBATCHES}"
        )
    if any(
        not isinstance(microbatch, CategoricalMicrobatch)
        for microbatch in microbatch_tuple
    ):
        raise ValueError("all microbatches must be CategoricalMicrobatch instances")
    microbatch_ids = [microbatch.microbatch_id for microbatch in microbatch_tuple]
    if len(set(microbatch_ids)) != len(microbatch_ids):
        raise ValueError("microbatch_id values must be unique")
    all_tokens = tuple(
        token for microbatch in microbatch_tuple for token in microbatch.tokens
    )
    token_ids = [token.token_id for token in all_tokens]
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("token_id values must be globally unique")
    vocabulary_size = len(all_tokens[0].probability_weights)
    if any(
        len(token.probability_weights) != vocabulary_size for token in all_tokens
    ):
        raise ValueError("all probability vectors must share one vocabulary size")
    if len(all_tokens) * vocabulary_size > MAX_GRADIENT_ELEMENTS:
        raise ValueError(
            f"token-by-vocabulary elements cannot exceed {MAX_GRADIENT_ELEMENTS}"
        )

    valid_by_microbatch = tuple(
        tuple(token for token in microbatch.tokens if token.target_index is not None)
        for microbatch in microbatch_tuple
    )
    valid_token_count = sum(len(tokens) for tokens in valid_by_microbatch)
    ignored_token_count = len(all_tokens) - valid_token_count
    microbatch_count = len(microbatch_tuple)
    naive_microbatch_weight = Fraction(1, microbatch_count)
    contributions: list[MicrobatchGradientContribution] = []
    mean_gradients: list[tuple[Fraction, ...]] = []
    mean_losses: list[float] = []

    for microbatch, valid_tokens in zip(
        microbatch_tuple, valid_by_microbatch, strict=True
    ):
        gradients = tuple(
            token.logit_gradient for token in valid_tokens
        )
        if any(gradient is None for gradient in gradients):
            raise AssertionError("valid tokens must have logit gradients")
        typed_gradients = tuple(
            gradient for gradient in gradients if gradient is not None
        )
        mean_gradient = _mean_fraction_vectors(typed_gradients)
        losses = tuple(
            token.negative_log_likelihood for token in valid_tokens
        )
        if any(loss is None for loss in losses):
            raise AssertionError("valid tokens must have negative log likelihoods")
        typed_losses = tuple(loss for loss in losses if loss is not None)
        mean_loss = math.fsum(typed_losses) / len(typed_losses)
        contributions.append(
            MicrobatchGradientContribution(
                microbatch_id=microbatch.microbatch_id,
                valid_token_count=len(valid_tokens),
                ignored_token_count=len(microbatch.tokens) - len(valid_tokens),
                correct_global_weight=Fraction(
                    len(valid_tokens), valid_token_count
                ),
                naive_equal_microbatch_weight=naive_microbatch_weight,
                token_mean_negative_log_likelihood=mean_loss,
                token_mean_class_aggregate_logit_gradient=mean_gradient,
            )
        )
        mean_gradients.append(mean_gradient)
        mean_losses.append(mean_loss)

    full_gradient = tuple(
        sum(
            (
                contribution.correct_global_weight
                * contribution.token_mean_class_aggregate_logit_gradient[index]
                for contribution in contributions
            ),
            start=Fraction(0, 1),
        )
        for index in range(vocabulary_size)
    )
    correctly_accumulated_gradient = tuple(
        sum(
            (
                contribution.correct_global_weight
                * mean_gradient[index]
                for contribution, mean_gradient in zip(
                    contributions, mean_gradients, strict=True
                )
            ),
            start=Fraction(0, 1),
        )
        for index in range(vocabulary_size)
    )
    naive_gradient = tuple(
        sum(
            (
                naive_microbatch_weight * mean_gradient[index]
                for mean_gradient in mean_gradients
            ),
            start=Fraction(0, 1),
        )
        for index in range(vocabulary_size)
    )
    if correctly_accumulated_gradient != full_gradient:
        raise AssertionError("count-scaled accumulation must equal the full token mean")

    full_loss = math.fsum(
        contribution.correct_global_weight
        * contribution.token_mean_negative_log_likelihood
        for contribution in contributions
    )
    correctly_accumulated_loss = math.fsum(
        contribution.correct_global_weight * mean_loss
        for contribution, mean_loss in zip(
            contributions, mean_losses, strict=True
        )
    )
    naive_loss = math.fsum(mean_losses) / microbatch_count
    if not math.isclose(
        full_loss,
        correctly_accumulated_loss,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise AssertionError("count-scaled loss must equal the full token mean")
    return GradientAccumulationAnalysis(
        vocabulary_size=vocabulary_size,
        microbatch_count=microbatch_count,
        valid_token_count=valid_token_count,
        ignored_token_count=ignored_token_count,
        microbatches=tuple(contributions),
        full_batch_token_mean_negative_log_likelihood=full_loss,
        count_scaled_accumulated_negative_log_likelihood=(
            correctly_accumulated_loss
        ),
        naive_equal_microbatch_negative_log_likelihood=naive_loss,
        naive_negative_log_likelihood_bias=naive_loss - full_loss,
        full_batch_class_aggregate_logit_gradient=full_gradient,
        count_scaled_accumulated_class_aggregate_logit_gradient=(
            correctly_accumulated_gradient
        ),
        naive_equal_microbatch_class_aggregate_logit_gradient=naive_gradient,
        naive_minus_full_class_aggregate_logit_gradient=tuple(
            naive - full for naive, full in zip(naive_gradient, full_gradient, strict=True)
        ),
    )


def analyze_default_ddp_token_mean(
    rank_shards: Iterable[CategoricalMicrobatch],
    *,
    data_parallel_world_size: int,
) -> DDPTokenMeanAnalysis:
    """Compare token-mean gradients under default DDP gradient averaging.

    This oracle assumes exactly one authored shard per data-parallel rank and a
    reducer that averages synchronized gradients across ``D`` ranks. For a
    global valid-token count ``N``, each rank must therefore backpropagate its
    local loss sum scaled by ``D/N``. Scaling by ``1/N`` before a default DDP
    mean introduces an extra ``1/D``. Taking one local token mean per rank and
    averaging ranks instead gives each token on rank ``r`` weight
    ``1/(D*n_r)``.

    The function performs exact coefficient and token-logit gradient algebra.
    It does not initialize a process group, run a collective, execute backward,
    or establish the reducer semantics of a particular framework version.
    """

    if (
        type(data_parallel_world_size) is not int
        or data_parallel_world_size < 2
        or data_parallel_world_size > MAX_GRADIENT_MICROBATCHES
    ):
        raise ValueError(
            "data_parallel_world_size must be an integer in "
            f"[2, {MAX_GRADIENT_MICROBATCHES}]"
        )
    shard_tuple = tuple(rank_shards)
    if len(shard_tuple) != data_parallel_world_size:
        raise ValueError(
            "rank_shards must contain exactly one shard per data-parallel rank"
        )
    accumulation = analyze_masked_token_gradient_accumulation(shard_tuple)
    world_size = data_parallel_world_size
    global_count = accumulation.valid_token_count
    rank_sum_gradients = tuple(
        tuple(
            contribution.valid_token_count * value
            for value in contribution.token_mean_class_aggregate_logit_gradient
        )
        for contribution in accumulation.microbatches
    )
    correct_scale = Fraction(world_size, global_count)
    missing_world_size_scale = Fraction(1, global_count)

    def default_ddp_mean(scale: Fraction) -> tuple[Fraction, ...]:
        return tuple(
            sum(
                (
                    scale * rank_gradient[class_index]
                    for rank_gradient in rank_sum_gradients
                ),
                start=Fraction(0, 1),
            )
            / world_size
            for class_index in range(accumulation.vocabulary_size)
        )

    correctly_scaled = default_ddp_mean(correct_scale)
    missing_world_size = default_ddp_mean(missing_world_size_scale)
    expected_missing_world_size = tuple(
        value / world_size
        for value in accumulation.full_batch_class_aggregate_logit_gradient
    )
    if correctly_scaled != accumulation.full_batch_class_aggregate_logit_gradient:
        raise AssertionError("D/N scaling under a DDP mean must equal the token mean")
    if missing_world_size != expected_missing_world_size:
        raise AssertionError("1/N scaling under a DDP mean must add one 1/D factor")
    return DDPTokenMeanAnalysis(
        data_parallel_world_size=world_size,
        valid_token_count=global_count,
        ignored_token_count=accumulation.ignored_token_count,
        valid_token_counts_by_rank=tuple(
            contribution.valid_token_count
            for contribution in accumulation.microbatches
        ),
        correct_local_loss_sum_scale=correct_scale,
        missing_world_size_local_loss_sum_scale=missing_world_size_scale,
        rank_local_sum_class_aggregate_logit_gradients=rank_sum_gradients,
        full_batch_class_aggregate_logit_gradient=(
            accumulation.full_batch_class_aggregate_logit_gradient
        ),
        correctly_scaled_default_ddp_class_aggregate_logit_gradient=(
            correctly_scaled
        ),
        missing_world_size_default_ddp_class_aggregate_logit_gradient=(
            missing_world_size
        ),
        equal_rank_local_mean_class_aggregate_logit_gradient=(
            accumulation.naive_equal_microbatch_class_aggregate_logit_gradient
        ),
    )


def analyze_default_ddp_gradient_accumulation(
    rank_windows: Iterable[Iterable[CategoricalMicrobatch]],
    *,
    data_parallel_world_size: int,
    unclipped_sgd_learning_rate: Fraction,
) -> DDPGradientAccumulationAnalysis:
    """Analyze one equal-step DDP accumulation window exactly.

    The oracle assumes ``D`` ranks, the same positive number of micro-batches on
    every rank, and default DDP gradient averaging. Every local loss sum is
    therefore scaled by ``D/N``. Synchronizing only the final backward and
    synchronizing every backward have identical exact gradients; ``no_sync``
    changes communication timing, not the estimand. The plain-SGD parameter
    delta assumes zero-initialized parameters, no momentum, no weight decay,
    and no clipping.

    This function does not initialize a process group, execute ``no_sync``, run
    backward, clip gradients, or step an optimizer. A runtime control must prove
    those framework behaviors separately.
    """

    if (
        type(data_parallel_world_size) is not int
        or data_parallel_world_size < 2
        or data_parallel_world_size > MAX_GRADIENT_MICROBATCHES
    ):
        raise ValueError(
            "data_parallel_world_size must be an integer in "
            f"[2, {MAX_GRADIENT_MICROBATCHES}]"
        )
    if (
        not isinstance(unclipped_sgd_learning_rate, Fraction)
        or unclipped_sgd_learning_rate <= 0
        or unclipped_sgd_learning_rate > MAX_GRADIENT_WEIGHT
    ):
        raise ValueError(
            "unclipped_sgd_learning_rate must be a positive Fraction no greater "
            f"than {MAX_GRADIENT_WEIGHT}"
        )

    windows = tuple(tuple(window) for window in rank_windows)
    if len(windows) != data_parallel_world_size:
        raise ValueError(
            "rank_windows must contain exactly one accumulation window per rank"
        )
    step_counts = tuple(len(window) for window in windows)
    if not step_counts or step_counts[0] < 1:
        raise ValueError("every rank window must contain at least one micro-batch")
    if len(set(step_counts)) != 1:
        raise ValueError("every rank must have the same accumulation step count")
    if sum(step_counts) > MAX_GRADIENT_MICROBATCHES:
        raise ValueError(
            "total rank micro-batch count cannot exceed "
            f"{MAX_GRADIENT_MICROBATCHES}"
        )

    flattened = tuple(microbatch for window in windows for microbatch in window)
    accumulation = analyze_masked_token_gradient_accumulation(flattened)
    contributions = accumulation.microbatches
    steps = step_counts[0]
    rank_microbatch_sums: list[tuple[tuple[Fraction, ...], ...]] = []
    rank_counts: list[tuple[int, ...]] = []
    offset = 0
    for _window in windows:
        rank_contributions = contributions[offset : offset + steps]
        offset += steps
        rank_counts.append(
            tuple(item.valid_token_count for item in rank_contributions)
        )
        rank_microbatch_sums.append(
            tuple(
                tuple(
                    item.valid_token_count * value
                    for value in item.token_mean_class_aggregate_logit_gradient
                )
                for item in rank_contributions
            )
        )

    typed_rank_microbatch_sums = tuple(rank_microbatch_sums)
    rank_accumulated_sums = tuple(
        _sum_fraction_vectors(rank) for rank in typed_rank_microbatch_sums
    )
    world_size = data_parallel_world_size
    correct_scale = Fraction(world_size, accumulation.valid_token_count)
    scaled_rank_accumulated = tuple(
        tuple(correct_scale * value for value in rank_gradient)
        for rank_gradient in rank_accumulated_sums
    )
    one_sync_after_accumulation = _mean_fraction_vectors(
        scaled_rank_accumulated
    )
    synchronized_step_gradients = tuple(
        _mean_fraction_vectors(
            tuple(
                tuple(correct_scale * value for value in rank[step])
                for rank in typed_rank_microbatch_sums
            )
        )
        for step in range(steps)
    )
    sync_every_microbatch = _sum_fraction_vectors(synchronized_step_gradients)
    full_gradient = accumulation.full_batch_class_aggregate_logit_gradient
    if one_sync_after_accumulation != full_gradient:
        raise AssertionError("one final DDP mean must equal the full token mean")
    if sync_every_microbatch != full_gradient:
        raise AssertionError("per-microbatch DDP means must equal the full token mean")
    learning_rate = unclipped_sgd_learning_rate
    parameter_delta = tuple(-learning_rate * value for value in full_gradient)
    counts_by_rank = tuple(rank_counts)
    return DDPGradientAccumulationAnalysis(
        data_parallel_world_size=world_size,
        accumulation_steps=steps,
        valid_token_count=accumulation.valid_token_count,
        ignored_token_count=accumulation.ignored_token_count,
        valid_token_counts_by_rank_and_microbatch=counts_by_rank,
        valid_token_counts_by_rank=tuple(sum(counts) for counts in counts_by_rank),
        correct_local_loss_sum_scale=correct_scale,
        rank_microbatch_loss_sum_class_aggregate_logit_gradients=(
            typed_rank_microbatch_sums
        ),
        rank_accumulated_loss_sum_class_aggregate_logit_gradients=(
            rank_accumulated_sums
        ),
        full_batch_class_aggregate_logit_gradient=full_gradient,
        one_sync_after_accumulation_class_aggregate_logit_gradient=(
            one_sync_after_accumulation
        ),
        sync_every_microbatch_class_aggregate_logit_gradient=(
            sync_every_microbatch
        ),
        unclipped_sgd_learning_rate=learning_rate,
        unclipped_sgd_parameter_delta=parameter_delta,
    )
