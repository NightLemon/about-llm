"""Inference measurement and sampling utilities."""

from typing import TYPE_CHECKING, cast

from about_llm.inference.beam_search import (
    BeamPrefix,
    BeamSearchResult,
    BeamSearchStep,
    BeamSequence,
    beam_search_from_probabilities,
)
from about_llm.inference.constrained import (
    ConstrainedDecodingResult,
    ConstrainedDecodingStep,
    ConstraintDeadEndError,
    LiteralSetConstraint,
    constrained_greedy_from_probabilities,
)
from about_llm.inference.continuous_batching import (
    BatchingRequest,
    ContinuousBatchingReport,
    ContinuousBatchStep,
    PrefillSlice,
    RequestSchedule,
    simulate_continuous_batching,
)
from about_llm.inference.kv_allocator import (
    KVAllocatorReport,
    KVAppendResult,
    KVBlockState,
    KVCapacityError,
    KVSequenceState,
    PagedKVAllocator,
)
from about_llm.inference.kv_preemption import (
    KVPreemptionBatchingReport,
    KVPreemptionBatchStep,
    KVPreemptionEvent,
    KVPreemptionRequestSchedule,
    KVWorkSlice,
    simulate_kv_preemption_batching,
)
from about_llm.inference.kv_quantization import (
    QuantizedKVCache,
    quantize_kv_cache_int8,
    quantized_kv_grouped_query_attention,
)
from about_llm.inference.memory import (
    estimate_causal_generation_forward_positions,
    estimate_kv_cache_bytes,
)
from about_llm.inference.metrics import (
    InferenceAttempt,
    InferenceMeasurement,
    InferenceSummary,
    RequestOutcome,
    WorkloadSLO,
    WorkloadSummary,
    classify_http_failure,
    summarize_attempts,
    summarize_measurements,
)
from about_llm.inference.prefix_cache import (
    PrefixCache,
    PrefixCacheCapacityError,
    PrefixCacheEntryState,
    PrefixCacheIdentity,
    PrefixCacheLease,
    PrefixCacheLeaseError,
    PrefixCacheReport,
    sha256_prefix_fingerprint,
)
from about_llm.inference.quantization import (
    QUANTIZED_MATRIX_FORMAT_VERSION,
    GroupwiseQuantizedMatrix,
    PackedGroupwiseQuantizedMatrix,
    QuantizationError,
    quantization_error,
    quantize_symmetric_groupwise,
    quantized_linear,
)
from about_llm.inference.quantized_bundle import (
    QUANTIZED_BUNDLE_FORMAT_VERSION,
    QUANTIZED_BUNDLE_SCHEMA_VERSION,
    NamedQuantizedMatrix,
    QuantizedBundleIdentity,
    QuantizedBundleLimits,
    QuantizedMatrixBundle,
)
from about_llm.inference.roofline import RooflineBound, roofline_lower_bound
from about_llm.inference.sampling import (
    NextTokenSamplingStep,
    SamplingConfig,
    greedy_next_token,
    sample_next_token,
)
from about_llm.inference.self_consistency import (
    BinaryMajorityAnalysis,
    BinaryVoteRegime,
    RegimeMajorityContribution,
    analyze_latent_regime_binary_majority,
)
from about_llm.inference.speculative import (
    SpeculativeBlockResult,
    SpeculativeDistributionAudit,
    SpeculativeStepResult,
    audit_speculative_distribution,
    speculative_sample_step,
    verify_speculative_block,
)
from about_llm.inference.stop_matching import (
    IncrementalStopMatcher,
    StopMatcherReport,
    StopMatcherStateError,
    StopMatchUpdate,
)
from about_llm.inference.verifier_selection import (
    BestOfNAnalysis,
    CandidateSelectionProbability,
    VerifierCandidate,
    analyze_verifier_guided_best_of_n,
)
from about_llm.inference.workload import (
    ArrivalProcess,
    ArrivalSchedule,
    build_arrival_schedule,
)

if TYPE_CHECKING:
    from about_llm.inference.minigpt_checkpoint import (
        MINIGPT_ARCHITECTURE_ID,
        MINIGPT_ARCHITECTURE_REVISION,
        MINIGPT_CHECKPOINT_FORMAT_VERSION,
        MINIGPT_CHECKPOINT_SCHEMA_VERSION,
        LoadedMiniGPTCheckpoint,
        MiniGPTCheckpointIdentity,
        MiniGPTCheckpointLimits,
        load_quantized_minigpt_checkpoint,
        read_quantized_minigpt_checkpoint,
        serialize_quantized_minigpt_checkpoint,
        write_quantized_minigpt_checkpoint_new,
    )
    from about_llm.inference.paged_kv_torch import (
        KVTensorStorePoisonedError,
        PagedKVTensorStore,
    )

_MINIGPT_CHECKPOINT_EXPORTS = frozenset(
    {
        "MINIGPT_ARCHITECTURE_ID",
        "MINIGPT_ARCHITECTURE_REVISION",
        "MINIGPT_CHECKPOINT_FORMAT_VERSION",
        "MINIGPT_CHECKPOINT_SCHEMA_VERSION",
        "LoadedMiniGPTCheckpoint",
        "MiniGPTCheckpointIdentity",
        "MiniGPTCheckpointLimits",
        "load_quantized_minigpt_checkpoint",
        "read_quantized_minigpt_checkpoint",
        "serialize_quantized_minigpt_checkpoint",
        "write_quantized_minigpt_checkpoint_new",
    }
)

_PYTORCH_EXPORTS = frozenset(
    {"KVTensorStorePoisonedError", "PagedKVTensorStore"}
)


def __getattr__(name: str) -> object:
    if name in _MINIGPT_CHECKPOINT_EXPORTS:
        from about_llm.inference import minigpt_checkpoint

        return cast(object, getattr(minigpt_checkpoint, name))
    if name in _PYTORCH_EXPORTS:
        from about_llm.inference import paged_kv_torch

        return cast(object, getattr(paged_kv_torch, name))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MINIGPT_ARCHITECTURE_ID",
    "MINIGPT_ARCHITECTURE_REVISION",
    "MINIGPT_CHECKPOINT_FORMAT_VERSION",
    "MINIGPT_CHECKPOINT_SCHEMA_VERSION",
    "QUANTIZED_BUNDLE_FORMAT_VERSION",
    "QUANTIZED_BUNDLE_SCHEMA_VERSION",
    "QUANTIZED_MATRIX_FORMAT_VERSION",
    "ArrivalProcess",
    "ArrivalSchedule",
    "BatchingRequest",
    "BeamPrefix",
    "BeamSearchResult",
    "BeamSearchStep",
    "BeamSequence",
    "BestOfNAnalysis",
    "BinaryMajorityAnalysis",
    "BinaryVoteRegime",
    "CandidateSelectionProbability",
    "ConstrainedDecodingResult",
    "ConstrainedDecodingStep",
    "ConstraintDeadEndError",
    "ContinuousBatchStep",
    "ContinuousBatchingReport",
    "GroupwiseQuantizedMatrix",
    "IncrementalStopMatcher",
    "InferenceAttempt",
    "InferenceMeasurement",
    "InferenceSummary",
    "KVAllocatorReport",
    "KVAppendResult",
    "KVBlockState",
    "KVCapacityError",
    "KVPreemptionBatchStep",
    "KVPreemptionBatchingReport",
    "KVPreemptionEvent",
    "KVPreemptionRequestSchedule",
    "KVSequenceState",
    "KVTensorStorePoisonedError",
    "KVWorkSlice",
    "LiteralSetConstraint",
    "LoadedMiniGPTCheckpoint",
    "MiniGPTCheckpointIdentity",
    "MiniGPTCheckpointLimits",
    "NamedQuantizedMatrix",
    "NextTokenSamplingStep",
    "PackedGroupwiseQuantizedMatrix",
    "PagedKVAllocator",
    "PagedKVTensorStore",
    "PrefillSlice",
    "PrefixCache",
    "PrefixCacheCapacityError",
    "PrefixCacheEntryState",
    "PrefixCacheIdentity",
    "PrefixCacheLease",
    "PrefixCacheLeaseError",
    "PrefixCacheReport",
    "QuantizationError",
    "QuantizedBundleIdentity",
    "QuantizedBundleLimits",
    "QuantizedKVCache",
    "QuantizedMatrixBundle",
    "RegimeMajorityContribution",
    "RequestOutcome",
    "RequestSchedule",
    "RooflineBound",
    "SamplingConfig",
    "SpeculativeBlockResult",
    "SpeculativeDistributionAudit",
    "SpeculativeStepResult",
    "StopMatchUpdate",
    "StopMatcherReport",
    "StopMatcherStateError",
    "VerifierCandidate",
    "WorkloadSLO",
    "WorkloadSummary",
    "analyze_latent_regime_binary_majority",
    "analyze_verifier_guided_best_of_n",
    "audit_speculative_distribution",
    "beam_search_from_probabilities",
    "build_arrival_schedule",
    "classify_http_failure",
    "constrained_greedy_from_probabilities",
    "estimate_causal_generation_forward_positions",
    "estimate_kv_cache_bytes",
    "greedy_next_token",
    "load_quantized_minigpt_checkpoint",
    "quantization_error",
    "quantize_kv_cache_int8",
    "quantize_symmetric_groupwise",
    "quantized_kv_grouped_query_attention",
    "quantized_linear",
    "read_quantized_minigpt_checkpoint",
    "roofline_lower_bound",
    "sample_next_token",
    "serialize_quantized_minigpt_checkpoint",
    "sha256_prefix_fingerprint",
    "simulate_continuous_batching",
    "simulate_kv_preemption_batching",
    "speculative_sample_step",
    "summarize_attempts",
    "summarize_measurements",
    "verify_speculative_block",
    "write_quantized_minigpt_checkpoint_new",
]
