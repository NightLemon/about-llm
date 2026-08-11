"""Strict, pickle-free MiniGPT training checkpoint and exact-resume control."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from about_llm.from_scratch.gpt_torch import GPTConfig, MiniGPT, TransformerBlock
from about_llm.from_scratch.tokenizer import ByteBPETokenizer
from about_llm.llmops import canonical_json_bytes

MINIGPT_TRAINING_CHECKPOINT_FORMAT_VERSION = 1
MINIGPT_TRAINING_CHECKPOINT_SCHEMA_VERSION = (
    "about-llm.minigpt-training-checkpoint.v1"
)
MINIGPT_TRAINING_ARCHITECTURE_REVISION = "about-llm.minigpt.training-forward.v1"

_MAGIC = b"ALLMTRN1"
_HEADER = struct.Struct("<8sB3xIII")
_SHA256_BYTES = 32
_UINT32_MAX = (1 << 32) - 1
_MANIFEST_FIELDS = {
    "architecture",
    "config",
    "dataset",
    "identity",
    "progress",
    "schema_version",
    "tensors",
    "tokenizer",
    "training",
}
_TENSOR_FIELDS = {"dtype", "length", "name", "offset", "sha256", "shape"}
_DTYPES: dict[str, np.dtype[Any]] = {
    "float32-le": np.dtype("<f4"),
    "int64-le": np.dtype("<i8"),
    "uint8": np.dtype("u1"),
}


@dataclass(frozen=True)
class AdamWTrainingConfig:
    learning_rate: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        for name in ("learning_rate", "epsilon"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a finite positive number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 <= float(value) < 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if (
            not isinstance(self.weight_decay, (int, float))
            or isinstance(self.weight_decay, bool)
            or not math.isfinite(float(self.weight_decay))
            or float(self.weight_decay) < 0
        ):
            raise ValueError("weight_decay must be finite and non-negative")


@dataclass(frozen=True)
class LinearLearningRateSchedule:
    initial_learning_rate: float
    final_learning_rate: float
    total_updates: int

    def __post_init__(self) -> None:
        for name in ("initial_learning_rate", "final_learning_rate"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        _positive_integer(self.total_updates, "total_updates")
        if self.total_updates > 1 << 24:
            raise ValueError("total_updates exceeds exact float32 AdamW step range")

    def learning_rate_for_update(self, update_index: int) -> float:
        if (
            isinstance(update_index, bool)
            or not isinstance(update_index, int)
            or not 0 <= update_index < self.total_updates
        ):
            raise ValueError("update_index must be within the schedule")
        if self.total_updates == 1:
            return float(self.initial_learning_rate)
        fraction = update_index / (self.total_updates - 1)
        return float(
            self.initial_learning_rate
            + fraction * (self.final_learning_rate - self.initial_learning_rate)
        )


@dataclass(frozen=True)
class MiniGPTTrainingConfig:
    optimizer: AdamWTrainingConfig
    schedule: LinearLearningRateSchedule
    batch_size: int
    max_grad_norm: float

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, AdamWTrainingConfig):
            raise TypeError("optimizer must be AdamWTrainingConfig")
        if not isinstance(self.schedule, LinearLearningRateSchedule):
            raise TypeError("schedule must be LinearLearningRateSchedule")
        _positive_integer(self.batch_size, "batch_size")
        if (
            not isinstance(self.max_grad_norm, (int, float))
            or isinstance(self.max_grad_norm, bool)
            or not math.isfinite(float(self.max_grad_norm))
            or float(self.max_grad_norm) <= 0
        ):
            raise ValueError("max_grad_norm must be a finite positive number")
        if not math.isclose(
            float(self.optimizer.learning_rate),
            float(self.schedule.initial_learning_rate),
        ):
            raise ValueError("optimizer and schedule initial learning rates must match")


@dataclass(frozen=True)
class TrainingDatasetBinding:
    examples: int
    sequence_tokens: int
    sha256: str


@dataclass
class ShuffledBatchStream:
    permutation: Tensor
    cursor: int
    epoch: int
    generator_state: Tensor

    @classmethod
    def create(cls, example_count: int, *, seed: int) -> ShuffledBatchStream:
        _positive_integer(example_count, "example_count")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutation = torch.randperm(example_count, generator=generator)
        return cls(
            permutation=permutation,
            cursor=0,
            epoch=0,
            generator_state=generator.get_state().clone(),
        )

    def next_indices(self, *, example_count: int, batch_size: int) -> Tensor:
        _validate_stream(self, example_count=example_count, batch_size=batch_size)
        generator = torch.Generator(device="cpu")
        generator.set_state(self.generator_state)
        if self.cursor + batch_size > example_count:
            self.permutation = torch.randperm(example_count, generator=generator)
            self.cursor = 0
            self.epoch += 1
        indices = self.permutation[self.cursor : self.cursor + batch_size].clone()
        self.cursor += batch_size
        self.generator_state = generator.get_state().clone()
        return indices


@dataclass
class MiniGPTTrainingState:
    model: MiniGPT
    tokenizer: ByteBPETokenizer
    optimizer: torch.optim.AdamW
    training_config: MiniGPTTrainingConfig
    dataset_binding: TrainingDatasetBinding
    batch_stream: ShuffledBatchStream
    torch_cpu_rng_state: Tensor
    global_step: int = 0


@dataclass(frozen=True)
class MiniGPTTrainingCheckpointIdentity:
    run_id: str
    model_revision: str
    tokenizer_revision: str
    data_revision: str

    def __post_init__(self) -> None:
        for name in ("run_id", "model_revision", "tokenizer_revision", "data_revision"):
            _identity_string(getattr(self, name), name)


@dataclass(frozen=True)
class MiniGPTTrainingCheckpointLimits:
    max_artifact_bytes: int = 512 * 1024 * 1024
    max_manifest_bytes: int = 16 * 1024 * 1024
    max_tensors: int = 300_000
    max_tensor_bytes: int = 256 * 1024 * 1024
    max_model_parameter_count: int = 25_000_000
    max_tokenizer_merges: int = 1_000_000
    max_examples: int = 10_000_000
    max_rng_state_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= _UINT32_MAX
            ):
                raise ValueError(f"{name} must be an integer in [1, 2^32-1]")


@dataclass(frozen=True)
class MiniGPTTrainingStep:
    global_step: int
    epoch: int
    batch_indices: tuple[int, ...]
    learning_rate: float
    loss: float
    gradient_norm_before_clip: float


def create_minigpt_training_state(
    model: MiniGPT,
    tokenizer: ByteBPETokenizer,
    dataset: Tensor,
    *,
    training_config: MiniGPTTrainingConfig,
    training_seed: int,
    data_seed: int,
) -> MiniGPTTrainingState:
    if type(model) is not MiniGPT:
        raise TypeError("model must be MiniGPT")
    if type(tokenizer) is not ByteBPETokenizer:
        raise TypeError("tokenizer must be ByteBPETokenizer")
    binding = training_dataset_binding(dataset, vocab_size=model.config.vocab_size)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary must match MiniGPT config")
    if binding.sequence_tokens != model.config.context_length + 1:
        raise ValueError("dataset sequences must equal context_length + 1")
    if binding.examples < training_config.batch_size:
        raise ValueError("dataset must contain at least one full batch")
    _validate_model_structure(model)
    optimizer = _build_optimizer(model, training_config.optimizer)
    training_generator = torch.Generator(device="cpu")
    training_generator.manual_seed(_nonnegative_integer(training_seed, "training_seed"))
    state = MiniGPTTrainingState(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        training_config=training_config,
        dataset_binding=binding,
        batch_stream=ShuffledBatchStream.create(binding.examples, seed=data_seed),
        torch_cpu_rng_state=training_generator.get_state().clone(),
    )
    _validate_training_state(state, dataset=dataset, require_optimizer_state=False)
    return state


def run_minigpt_training_updates(
    state: MiniGPTTrainingState,
    dataset: Tensor,
    *,
    updates: int,
) -> tuple[MiniGPTTrainingStep, ...]:
    _positive_integer(updates, "updates")
    _validate_training_state(
        state,
        dataset=dataset,
        require_optimizer_state=state.global_step > 0,
    )
    if state.global_step + updates > state.training_config.schedule.total_updates:
        raise ValueError("requested updates exceed learning-rate schedule")
    external_rng_state = torch.get_rng_state().clone()
    reports: list[MiniGPTTrainingStep] = []
    torch.set_rng_state(state.torch_cpu_rng_state)
    try:
        state.model.train()
        for _ in range(updates):
            update_index = state.global_step
            learning_rate = state.training_config.schedule.learning_rate_for_update(
                update_index
            )
            state.optimizer.param_groups[0]["lr"] = learning_rate
            indices = state.batch_stream.next_indices(
                example_count=state.dataset_binding.examples,
                batch_size=state.training_config.batch_size,
            )
            batch = dataset.index_select(0, indices)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            state.optimizer.zero_grad(set_to_none=True)
            _, loss = state.model(inputs, targets)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("MiniGPT training loss must be finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                state.model.parameters(), state.training_config.max_grad_norm
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("MiniGPT gradient norm must be finite")
            state.optimizer.step()
            state.optimizer.zero_grad(set_to_none=True)
            state.global_step += 1
            reports.append(
                MiniGPTTrainingStep(
                    global_step=state.global_step,
                    epoch=state.batch_stream.epoch,
                    batch_indices=tuple(int(index) for index in indices.tolist()),
                    learning_rate=learning_rate,
                    loss=float(loss.detach()),
                    gradient_norm_before_clip=float(gradient_norm),
                )
            )
    finally:
        state.torch_cpu_rng_state = torch.get_rng_state().clone()
        torch.set_rng_state(external_rng_state)
    _validate_training_state(state, dataset=dataset, require_optimizer_state=True)
    return tuple(reports)


def serialize_minigpt_training_checkpoint(
    state: MiniGPTTrainingState,
    dataset: Tensor,
    *,
    identity: MiniGPTTrainingCheckpointIdentity,
    limits: MiniGPTTrainingCheckpointLimits | None = None,
) -> bytes:
    exact_limits = _limits(limits)
    if not isinstance(identity, MiniGPTTrainingCheckpointIdentity):
        raise TypeError("identity must be MiniGPTTrainingCheckpointIdentity")
    if state.global_step <= 0:
        raise ValueError("training checkpoint requires at least one completed update")
    _validate_training_state(state, dataset=dataset, require_optimizer_state=True)
    if len(state.tokenizer.merges) > exact_limits.max_tokenizer_merges:
        raise ValueError("tokenizer merge count exceeds configured limit")
    if state.dataset_binding.examples > exact_limits.max_examples:
        raise ValueError("dataset example count exceeds configured limit")
    if (
        state.torch_cpu_rng_state.numel() > exact_limits.max_rng_state_bytes
        or state.batch_stream.generator_state.numel()
        > exact_limits.max_rng_state_bytes
    ):
        raise ValueError("training RNG state exceeds configured limit")
    parameter_shapes = _expected_parameter_shapes(state.model.config)
    _validate_model_size(parameter_shapes, exact_limits)

    tensors: dict[str, Tensor] = {}
    named_parameters = dict(state.model.named_parameters())
    for name, parameter in named_parameters.items():
        tensors[f"model/{name}"] = parameter.detach().cpu()
        optimizer_state = state.optimizer.state[parameter]
        _validate_optimizer_parameter_state(
            optimizer_state, parameter=parameter, global_step=state.global_step, name=name
        )
        tensors[f"optimizer/{name}/exp_avg"] = cast(
            Tensor, optimizer_state["exp_avg"]
        ).detach().cpu()
        tensors[f"optimizer/{name}/exp_avg_sq"] = cast(
            Tensor, optimizer_state["exp_avg_sq"]
        ).detach().cpu()
    tensors["stream/permutation"] = state.batch_stream.permutation.detach().cpu()
    tensors["rng/data_generator"] = state.batch_stream.generator_state.detach().cpu()
    tensors["rng/torch_cpu"] = state.torch_cpu_rng_state.detach().cpu()
    if len(tensors) > exact_limits.max_tensors:
        raise ValueError("tensor count exceeds configured checkpoint limit")

    payload_parts: list[bytes] = []
    descriptors: list[dict[str, Any]] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        dtype_name, tensor_bytes = _encode_tensor(tensor, name=name)
        if len(tensor_bytes) > exact_limits.max_tensor_bytes:
            raise ValueError(f"tensor {name} exceeds configured byte limit")
        if offset + len(tensor_bytes) > _UINT32_MAX:
            raise ValueError("training checkpoint payload exceeds v1 uint32 range")
        descriptors.append(
            {
                "name": name,
                "dtype": dtype_name,
                "shape": list(tensor.shape),
                "offset": offset,
                "length": len(tensor_bytes),
                "sha256": "sha256:" + hashlib.sha256(tensor_bytes).hexdigest(),
            }
        )
        payload_parts.append(tensor_bytes)
        offset += len(tensor_bytes)

    current_lr = _current_learning_rate(state)
    manifest = canonical_json_bytes(
        {
            "schema_version": MINIGPT_TRAINING_CHECKPOINT_SCHEMA_VERSION,
            "architecture": {
                "id": "about-llm.minigpt",
                "revision": MINIGPT_TRAINING_ARCHITECTURE_REVISION,
                "tied_parameters": [
                    {
                        "alias": "lm_head.weight",
                        "target": "token_embedding.weight",
                    }
                ],
            },
            "identity": asdict(identity),
            "config": _config_payload(state.model.config),
            "tokenizer": {
                "kind": "about-llm.byte-bpe",
                "format_version": 1,
                "vocab_size": state.tokenizer.vocab_size,
                "merges": [list(pair) for pair in state.tokenizer.merges],
            },
            "dataset": asdict(state.dataset_binding),
            "training": _training_config_payload(state.training_config),
            "progress": {
                "global_step": state.global_step,
                "current_learning_rate": current_lr,
                "epoch": state.batch_stream.epoch,
                "cursor": state.batch_stream.cursor,
                "gradient_accumulation_position": 0,
                "checkpoint_at_optimizer_boundary": True,
            },
            "tensors": descriptors,
        }
    )
    payload = b"".join(payload_parts)
    if len(manifest) > exact_limits.max_manifest_bytes:
        raise ValueError("training checkpoint manifest exceeds configured limit")
    if len(manifest) > _UINT32_MAX or len(payload) > _UINT32_MAX:
        raise ValueError("training checkpoint manifest or payload exceeds v1 range")
    header = _HEADER.pack(
        _MAGIC,
        MINIGPT_TRAINING_CHECKPOINT_FORMAT_VERSION,
        len(manifest),
        len(descriptors),
        len(payload),
    )
    body = header + manifest + payload
    artifact = body + hashlib.sha256(body).digest()
    if len(artifact) > exact_limits.max_artifact_bytes:
        raise ValueError("training checkpoint artifact exceeds configured limit")
    return artifact


def load_minigpt_training_checkpoint(
    artifact: bytes,
    dataset: Tensor,
    *,
    limits: MiniGPTTrainingCheckpointLimits | None = None,
) -> tuple[MiniGPTTrainingState, MiniGPTTrainingCheckpointIdentity]:
    if not isinstance(artifact, bytes):
        raise TypeError("training checkpoint artifact must be immutable bytes")
    exact_limits = _limits(limits)
    if len(artifact) > exact_limits.max_artifact_bytes:
        raise ValueError("training checkpoint artifact exceeds configured limit")
    if len(artifact) < _HEADER.size + _SHA256_BYTES:
        raise ValueError("training checkpoint artifact is truncated")
    magic, version, manifest_length, tensor_count, payload_length = _HEADER.unpack_from(
        artifact
    )
    if magic != _MAGIC:
        raise ValueError("invalid MiniGPT training checkpoint magic")
    if version != MINIGPT_TRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported MiniGPT training checkpoint format")
    if manifest_length > exact_limits.max_manifest_bytes:
        raise ValueError("training checkpoint manifest exceeds configured limit")
    if tensor_count == 0 or tensor_count > exact_limits.max_tensors:
        raise ValueError("training checkpoint tensor count is invalid or exceeds limit")
    expected_length = _HEADER.size + manifest_length + payload_length + _SHA256_BYTES
    if len(artifact) != expected_length:
        if len(artifact) > expected_length:
            raise ValueError("training checkpoint contains trailing data")
        raise ValueError("training checkpoint length is inconsistent or truncated")
    body = artifact[:-_SHA256_BYTES]
    if not hmac.compare_digest(hashlib.sha256(body).digest(), artifact[-32:]):
        raise ValueError("training checkpoint SHA-256 mismatch")

    manifest_start = _HEADER.size
    payload_start = manifest_start + manifest_length
    manifest = _strict_manifest(artifact[manifest_start:payload_start])
    payload = artifact[payload_start:-_SHA256_BYTES]
    _parse_architecture(manifest["architecture"])
    identity = _parse_identity(manifest["identity"])
    config = _parse_config(manifest["config"])
    parameter_shapes = _expected_parameter_shapes(config)
    _validate_model_size(parameter_shapes, exact_limits)
    tokenizer = _parse_tokenizer(
        manifest["tokenizer"], config=config, limits=exact_limits
    )
    expected_binding = _parse_dataset_binding(manifest["dataset"])
    actual_binding = training_dataset_binding(dataset, vocab_size=config.vocab_size)
    if expected_binding != actual_binding:
        raise ValueError("training dataset identity does not match checkpoint")
    if actual_binding.examples > exact_limits.max_examples:
        raise ValueError("dataset example count exceeds configured limit")
    if actual_binding.sequence_tokens != config.context_length + 1:
        raise ValueError("training dataset sequence length does not match config")
    training_config = _parse_training_config(manifest["training"])
    progress = _parse_progress(
        manifest["progress"],
        training_config=training_config,
        example_count=actual_binding.examples,
    )
    descriptors_value = manifest["tensors"]
    if not isinstance(descriptors_value, list) or len(descriptors_value) != tensor_count:
        raise ValueError("training checkpoint tensor descriptors are inconsistent")
    expected_tensors = _expected_tensor_contract(
        parameter_shapes, example_count=actual_binding.examples
    )
    if len(descriptors_value) != len(expected_tensors):
        raise ValueError("training checkpoint tensor count does not match config")

    arrays: dict[str, NDArray[Any]] = {}
    expected_offset = 0
    previous_name: str | None = None
    for value in descriptors_value:
        descriptor = _parse_tensor_descriptor(value)
        name = cast(str, descriptor["name"])
        dtype_name = cast(str, descriptor["dtype"])
        shape = cast(tuple[int, ...], descriptor["shape"])
        offset = cast(int, descriptor["offset"])
        length = cast(int, descriptor["length"])
        if previous_name is not None and name <= previous_name:
            raise ValueError("training tensor descriptors must be name-sorted and unique")
        if name not in expected_tensors:
            raise ValueError(f"unexpected training checkpoint tensor: {name}")
        expected_dtype, expected_shape = expected_tensors[name]
        if dtype_name != expected_dtype:
            raise ValueError(f"training tensor {name} dtype does not match contract")
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(f"training tensor {name} shape does not match contract")
        if expected_shape is None and (
            len(shape) != 1 or shape[0] > exact_limits.max_rng_state_bytes
        ):
            raise ValueError(f"training RNG tensor {name} shape exceeds limit")
        if offset != expected_offset:
            raise ValueError("training checkpoint tensor offsets must be contiguous")
        if length > exact_limits.max_tensor_bytes:
            raise ValueError("training checkpoint tensor exceeds byte limit")
        stop = offset + length
        if stop > len(payload):
            raise ValueError("training checkpoint tensor range exceeds payload")
        tensor_bytes = payload[offset:stop]
        digest = "sha256:" + hashlib.sha256(tensor_bytes).hexdigest()
        if not hmac.compare_digest(digest, cast(str, descriptor["sha256"])):
            raise ValueError(f"training tensor {name} SHA-256 mismatch")
        arrays[name] = _decode_tensor(
            tensor_bytes, dtype_name=dtype_name, shape=shape, name=name
        )
        expected_offset = stop
        previous_name = name
    if expected_offset != len(payload) or set(arrays) != set(expected_tensors):
        raise ValueError("training checkpoint tensor payload/set is incomplete")

    external_rng_state = torch.get_rng_state().clone()
    try:
        model = MiniGPT(config).cpu().float()
    finally:
        torch.set_rng_state(external_rng_state)
    _validate_model_structure(model)
    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, parameter in named_parameters.items():
            parameter.copy_(torch.from_numpy(arrays[f"model/{name}"]))
    optimizer = _build_optimizer(model, training_config.optimizer)
    optimizer.param_groups[0]["lr"] = progress["current_learning_rate"]
    global_step = cast(int, progress["global_step"])
    for name, parameter in named_parameters.items():
        optimizer.state[parameter] = {
            "step": torch.tensor(float(global_step), dtype=torch.float32),
            "exp_avg": torch.from_numpy(arrays[f"optimizer/{name}/exp_avg"]).clone(),
            "exp_avg_sq": torch.from_numpy(
                arrays[f"optimizer/{name}/exp_avg_sq"]
            ).clone(),
        }
    stream = ShuffledBatchStream(
        permutation=torch.from_numpy(arrays["stream/permutation"]).clone(),
        cursor=cast(int, progress["cursor"]),
        epoch=cast(int, progress["epoch"]),
        generator_state=torch.from_numpy(arrays["rng/data_generator"]).clone(),
    )
    state = MiniGPTTrainingState(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        training_config=training_config,
        dataset_binding=actual_binding,
        batch_stream=stream,
        torch_cpu_rng_state=torch.from_numpy(arrays["rng/torch_cpu"]).clone(),
        global_step=global_step,
    )
    _validate_training_state(state, dataset=dataset, require_optimizer_state=True)
    return state, identity


def write_minigpt_training_checkpoint_new(
    path: Path,
    artifact: bytes,
    dataset: Tensor,
    *,
    limits: MiniGPTTrainingCheckpointLimits | None = None,
) -> None:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    load_minigpt_training_checkpoint(artifact, dataset, limits=limits)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(artifact)
        stream.flush()
        os.fsync(stream.fileno())


def read_minigpt_training_checkpoint(
    path: Path,
    dataset: Tensor,
    *,
    limits: MiniGPTTrainingCheckpointLimits | None = None,
) -> tuple[MiniGPTTrainingState, MiniGPTTrainingCheckpointIdentity]:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    exact_limits = _limits(limits)
    with path.open("rb") as stream:
        file_size = os.fstat(stream.fileno()).st_size
        if file_size > exact_limits.max_artifact_bytes:
            raise ValueError("training checkpoint file exceeds artifact limit")
        artifact = stream.read(exact_limits.max_artifact_bytes + 1)
    if len(artifact) != file_size:
        raise ValueError("training checkpoint file changed or was not read completely")
    return load_minigpt_training_checkpoint(artifact, dataset, limits=exact_limits)


def training_dataset_binding(dataset: Tensor, *, vocab_size: int) -> TrainingDatasetBinding:
    if not isinstance(dataset, Tensor):
        raise TypeError("dataset must be a torch Tensor")
    if dataset.device.type != "cpu" or dataset.dtype != torch.int64 or dataset.ndim != 2:
        raise ValueError("dataset must be a CPU int64 [examples, sequence_tokens] tensor")
    examples, sequence_tokens = dataset.shape
    if examples <= 0 or sequence_tokens <= 1:
        raise ValueError("dataset must contain examples with input and target tokens")
    if torch.any(dataset < 0) or torch.any(dataset >= vocab_size):
        raise ValueError("dataset token ids must be within model vocabulary")
    contiguous = dataset.detach().contiguous().numpy().astype("<i8", copy=False)
    identity = canonical_json_bytes(
        {
            "dtype": "int64-le",
            "shape": [examples, sequence_tokens],
            "values_sha256": "sha256:"
            + hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        }
    )
    return TrainingDatasetBinding(
        examples=examples,
        sequence_tokens=sequence_tokens,
        sha256="sha256:" + hashlib.sha256(identity).hexdigest(),
    )


def _build_optimizer(
    model: MiniGPT, config: AdamWTrainingConfig
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        list(model.parameters()),
        lr=float(config.learning_rate),
        betas=(float(config.beta1), float(config.beta2)),
        eps=float(config.epsilon),
        weight_decay=float(config.weight_decay),
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _validate_training_state(
    state: MiniGPTTrainingState,
    *,
    dataset: Tensor,
    require_optimizer_state: bool,
) -> None:
    if not isinstance(state, MiniGPTTrainingState):
        raise TypeError("state must be MiniGPTTrainingState")
    _validate_model_structure(state.model)
    if type(state.tokenizer) is not ByteBPETokenizer:
        raise TypeError("training tokenizer must be ByteBPETokenizer")
    if state.tokenizer.vocab_size != state.model.config.vocab_size:
        raise ValueError("training tokenizer/model vocabulary mismatch")
    if training_dataset_binding(dataset, vocab_size=state.model.config.vocab_size) != (
        state.dataset_binding
    ):
        raise ValueError("training dataset identity does not match state")
    if state.dataset_binding.sequence_tokens != state.model.config.context_length + 1:
        raise ValueError("training dataset sequence length does not match model")
    if (
        isinstance(state.global_step, bool)
        or not isinstance(state.global_step, int)
        or not 0 <= state.global_step <= state.training_config.schedule.total_updates
    ):
        raise ValueError("training global_step is invalid")
    _validate_stream(
        state.batch_stream,
        example_count=state.dataset_binding.examples,
        batch_size=state.training_config.batch_size,
    )
    _validate_rng_state(state.torch_cpu_rng_state, "torch CPU RNG")
    if any(parameter.grad is not None for parameter in state.model.parameters()):
        raise ValueError("checkpoint boundary requires all parameter gradients cleared")
    _validate_optimizer(
        state,
        require_parameter_state=require_optimizer_state,
    )


def _validate_optimizer(
    state: MiniGPTTrainingState, *, require_parameter_state: bool
) -> None:
    optimizer = state.optimizer
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("training optimizer must be one-group torch AdamW")
    group = optimizer.param_groups[0]
    expected_parameters = list(dict(state.model.named_parameters()).values())
    if list(group["params"]) != expected_parameters:
        raise ValueError("optimizer parameter order/identity does not match model")
    config = state.training_config.optimizer
    if (
        tuple(group["betas"]) != (float(config.beta1), float(config.beta2))
        or not math.isclose(float(group["eps"]), float(config.epsilon))
        or not math.isclose(float(group["weight_decay"]), float(config.weight_decay))
        or group["amsgrad"] is not False
        or group["maximize"] is not False
        or group["foreach"] is not False
        or group["capturable"] is not False
        or group["differentiable"] is not False
        or group["fused"] is not False
    ):
        raise ValueError("optimizer hyperparameters/backend flags drifted")
    if state.global_step == 0:
        expected_lr = float(config.learning_rate)
    else:
        expected_lr = state.training_config.schedule.learning_rate_for_update(
            state.global_step - 1
        )
    if not math.isclose(float(group["lr"]), expected_lr, rel_tol=0, abs_tol=1e-15):
        raise ValueError("optimizer current learning rate does not match progress")
    named_parameters = dict(state.model.named_parameters())
    if require_parameter_state:
        if len(optimizer.state) != len(named_parameters):
            raise ValueError("optimizer state does not cover every unique parameter")
        for name, parameter in named_parameters.items():
            _validate_optimizer_parameter_state(
                optimizer.state.get(parameter, {}),
                parameter=parameter,
                global_step=state.global_step,
                name=name,
            )
    elif optimizer.state:
        raise ValueError("fresh optimizer must not contain parameter state")


def _validate_optimizer_parameter_state(
    value: dict[str, Any], *, parameter: Tensor, global_step: int, name: str
) -> None:
    if set(value) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError(f"optimizer state fields are invalid for {name}")
    step = value["step"]
    if (
        not isinstance(step, Tensor)
        or step.device.type != "cpu"
        or step.dtype != torch.float32
        or step.ndim != 0
        or float(step) != global_step
    ):
        raise ValueError(f"optimizer step is invalid for {name}")
    for field in ("exp_avg", "exp_avg_sq"):
        tensor = value[field]
        if (
            not isinstance(tensor, Tensor)
            or tensor.device != parameter.device
            or tensor.dtype != torch.float32
            or tensor.shape != parameter.shape
            or not torch.all(torch.isfinite(tensor))
        ):
            raise ValueError(f"optimizer {field} is invalid for {name}")


def _validate_stream(
    stream: ShuffledBatchStream, *, example_count: int, batch_size: int
) -> None:
    if not isinstance(stream, ShuffledBatchStream):
        raise TypeError("batch_stream must be ShuffledBatchStream")
    if (
        stream.permutation.device.type != "cpu"
        or stream.permutation.dtype != torch.int64
        or stream.permutation.shape != (example_count,)
        or not torch.equal(torch.sort(stream.permutation).values, torch.arange(example_count))
    ):
        raise ValueError("batch stream permutation must contain every example once")
    if (
        isinstance(stream.cursor, bool)
        or not isinstance(stream.cursor, int)
        or stream.cursor < 0
        or stream.cursor > example_count
        or stream.cursor % batch_size != 0
    ):
        raise ValueError("batch stream cursor is invalid")
    if isinstance(stream.epoch, bool) or not isinstance(stream.epoch, int) or stream.epoch < 0:
        raise ValueError("batch stream epoch is invalid")
    _validate_rng_state(stream.generator_state, "data generator RNG")


def _validate_rng_state(value: Tensor, name: str) -> None:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() == 0
    ):
        raise ValueError(f"{name} state must be a non-empty CPU uint8 vector")
    generator = torch.Generator(device="cpu")
    try:
        generator.set_state(value)
    except RuntimeError as error:
        raise ValueError(f"{name} state is invalid") from error


def _current_learning_rate(state: MiniGPTTrainingState) -> float:
    if state.global_step <= 0:
        return float(state.training_config.optimizer.learning_rate)
    return state.training_config.schedule.learning_rate_for_update(state.global_step - 1)


def _encode_tensor(value: Tensor, *, name: str) -> tuple[str, bytes]:
    if value.device.type != "cpu" or not value.is_contiguous():
        value = value.detach().cpu().contiguous()
    if value.dtype == torch.float32:
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"tensor {name} contains non-finite float32 values")
        dtype_name = "float32-le"
        array = value.numpy().astype("<f4", copy=False)
    elif value.dtype == torch.int64:
        dtype_name = "int64-le"
        array = value.numpy().astype("<i8", copy=False)
    elif value.dtype == torch.uint8:
        dtype_name = "uint8"
        array = value.numpy().astype("u1", copy=False)
    else:
        raise ValueError(f"tensor {name} dtype is unsupported")
    return dtype_name, array.tobytes(order="C")


def _decode_tensor(
    value: bytes, *, dtype_name: str, shape: tuple[int, ...], name: str
) -> NDArray[Any]:
    dtype = _DTYPES[dtype_name]
    expected_length = math.prod(shape) * dtype.itemsize
    if len(value) != expected_length:
        raise ValueError(f"training tensor {name} byte length does not match shape")
    array = cast(NDArray[Any], np.frombuffer(value, dtype=dtype).copy().reshape(shape))
    if dtype_name == "float32-le" and not np.all(np.isfinite(array)):
        raise ValueError(f"training tensor {name} contains non-finite values")
    return array


def _expected_tensor_contract(
    parameter_shapes: dict[str, tuple[int, ...]], *, example_count: int
) -> dict[str, tuple[str, tuple[int, ...] | None]]:
    result: dict[str, tuple[str, tuple[int, ...] | None]] = {}
    for name, shape in parameter_shapes.items():
        result[f"model/{name}"] = ("float32-le", shape)
        result[f"optimizer/{name}/exp_avg"] = ("float32-le", shape)
        result[f"optimizer/{name}/exp_avg_sq"] = ("float32-le", shape)
    result["stream/permutation"] = ("int64-le", (example_count,))
    result["rng/data_generator"] = ("uint8", None)
    result["rng/torch_cpu"] = ("uint8", None)
    return result


def _strict_manifest(value: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: _invalid_constant(constant),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("training checkpoint manifest must be strict UTF-8 JSON") from error
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("training checkpoint manifest fields are invalid")
    if payload["schema_version"] != MINIGPT_TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported MiniGPT training checkpoint schema")
    try:
        canonical = canonical_json_bytes(payload)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("training checkpoint manifest contains unsupported JSON") from error
    if canonical != value:
        raise ValueError("training checkpoint manifest must use canonical JSON")
    return payload


def _parse_architecture(value: Any) -> None:
    record = _record(value, {"id", "revision", "tied_parameters"}, "architecture")
    if (
        record["id"] != "about-llm.minigpt"
        or record["revision"] != MINIGPT_TRAINING_ARCHITECTURE_REVISION
        or record["tied_parameters"]
        != [{"alias": "lm_head.weight", "target": "token_embedding.weight"}]
    ):
        raise ValueError("training checkpoint architecture contract is unsupported")


def _parse_identity(value: Any) -> MiniGPTTrainingCheckpointIdentity:
    record = _record(
        value,
        {"run_id", "model_revision", "tokenizer_revision", "data_revision"},
        "identity",
    )
    return MiniGPTTrainingCheckpointIdentity(
        run_id=_identity_string(record["run_id"], "run_id"),
        model_revision=_identity_string(record["model_revision"], "model_revision"),
        tokenizer_revision=_identity_string(
            record["tokenizer_revision"], "tokenizer_revision"
        ),
        data_revision=_identity_string(record["data_revision"], "data_revision"),
    )


def _config_payload(config: GPTConfig) -> dict[str, Any]:
    if isinstance(config.dropout, bool) or not isinstance(config.dropout, (int, float)):
        raise ValueError("config.dropout must be numeric")
    payload: dict[str, Any] = {
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "model_dim": config.model_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "mlp_ratio": config.mlp_ratio,
        "dropout": float(config.dropout),
        "bias": config.bias,
    }
    _parse_config(payload)
    return payload


def _parse_config(value: Any) -> GPTConfig:
    record = _record(
        value,
        {
            "vocab_size",
            "context_length",
            "model_dim",
            "num_heads",
            "num_layers",
            "mlp_ratio",
            "dropout",
            "bias",
        },
        "config",
    )
    dropout = _finite_number(record["dropout"], "config.dropout")
    if not isinstance(record["bias"], bool):
        raise ValueError("config.bias must be boolean")
    return GPTConfig(
        vocab_size=_positive_integer(record["vocab_size"], "config.vocab_size"),
        context_length=_positive_integer(
            record["context_length"], "config.context_length"
        ),
        model_dim=_positive_integer(record["model_dim"], "config.model_dim"),
        num_heads=_positive_integer(record["num_heads"], "config.num_heads"),
        num_layers=_positive_integer(record["num_layers"], "config.num_layers"),
        mlp_ratio=_positive_integer(record["mlp_ratio"], "config.mlp_ratio"),
        dropout=dropout,
        bias=record["bias"],
    )


def _parse_tokenizer(
    value: Any,
    *,
    config: GPTConfig,
    limits: MiniGPTTrainingCheckpointLimits,
) -> ByteBPETokenizer:
    record = _record(
        value, {"kind", "format_version", "vocab_size", "merges"}, "tokenizer"
    )
    if record["kind"] != "about-llm.byte-bpe" or record["format_version"] != 1:
        raise ValueError("training checkpoint tokenizer format is unsupported")
    merges_value = record["merges"]
    if not isinstance(merges_value, list) or len(merges_value) > limits.max_tokenizer_merges:
        raise ValueError("training checkpoint tokenizer merges are invalid or exceed limit")
    merges: list[tuple[int, int]] = []
    for index, pair in enumerate(merges_value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"tokenizer merge {index} must contain two ids")
        merges.append(
            (
                _nonnegative_integer(pair[0], f"tokenizer merge {index} left"),
                _nonnegative_integer(pair[1], f"tokenizer merge {index} right"),
            )
        )
    tokenizer = ByteBPETokenizer(merges)
    if (
        _positive_integer(record["vocab_size"], "tokenizer.vocab_size")
        != tokenizer.vocab_size
        or tokenizer.vocab_size != config.vocab_size
    ):
        raise ValueError("training checkpoint tokenizer vocabulary does not match config")
    return tokenizer


def _training_config_payload(config: MiniGPTTrainingConfig) -> dict[str, Any]:
    return {
        "optimizer": {
            "type": "torch.optim.AdamW",
            "learning_rate": float(config.optimizer.learning_rate),
            "beta1": float(config.optimizer.beta1),
            "beta2": float(config.optimizer.beta2),
            "epsilon": float(config.optimizer.epsilon),
            "weight_decay": float(config.optimizer.weight_decay),
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        },
        "schedule": {
            "type": "linear-per-update-v1",
            "initial_learning_rate": float(
                config.schedule.initial_learning_rate
            ),
            "final_learning_rate": float(config.schedule.final_learning_rate),
            "total_updates": config.schedule.total_updates,
        },
        "batch_size": config.batch_size,
        "max_grad_norm": float(config.max_grad_norm),
        "precision": "cpu-float32",
        "gradient_accumulation_steps": 1,
        "drop_last_incomplete_batch": True,
    }


def _parse_training_config(value: Any) -> MiniGPTTrainingConfig:
    record = _record(
        value,
        {
            "optimizer",
            "schedule",
            "batch_size",
            "max_grad_norm",
            "precision",
            "gradient_accumulation_steps",
            "drop_last_incomplete_batch",
        },
        "training",
    )
    if (
        record["precision"] != "cpu-float32"
        or record["gradient_accumulation_steps"] != 1
        or record["drop_last_incomplete_batch"] is not True
    ):
        raise ValueError("training precision/accumulation/batch-boundary contract drifted")
    optimizer = _record(
        record["optimizer"],
        {
            "type",
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
        },
        "training.optimizer",
    )
    if (
        optimizer["type"] != "torch.optim.AdamW"
        or any(
            optimizer[field] is not False
            for field in (
                "amsgrad",
                "maximize",
                "foreach",
                "capturable",
                "differentiable",
                "fused",
            )
        )
    ):
        raise ValueError("training optimizer type/backend flags are unsupported")
    optimizer_config = AdamWTrainingConfig(
        learning_rate=_finite_number(
            optimizer["learning_rate"], "optimizer.learning_rate"
        ),
        beta1=_finite_number(optimizer["beta1"], "optimizer.beta1"),
        beta2=_finite_number(optimizer["beta2"], "optimizer.beta2"),
        epsilon=_finite_number(optimizer["epsilon"], "optimizer.epsilon"),
        weight_decay=_finite_number(
            optimizer["weight_decay"], "optimizer.weight_decay"
        ),
    )
    schedule = _record(
        record["schedule"],
        {
            "type",
            "initial_learning_rate",
            "final_learning_rate",
            "total_updates",
        },
        "training.schedule",
    )
    if schedule["type"] != "linear-per-update-v1":
        raise ValueError("training schedule type is unsupported")
    return MiniGPTTrainingConfig(
        optimizer=optimizer_config,
        schedule=LinearLearningRateSchedule(
            initial_learning_rate=_finite_number(
                schedule["initial_learning_rate"], "schedule.initial_learning_rate"
            ),
            final_learning_rate=_finite_number(
                schedule["final_learning_rate"], "schedule.final_learning_rate"
            ),
            total_updates=_positive_integer(
                schedule["total_updates"], "schedule.total_updates"
            ),
        ),
        batch_size=_positive_integer(record["batch_size"], "training.batch_size"),
        max_grad_norm=_finite_number(
            record["max_grad_norm"], "training.max_grad_norm"
        ),
    )


def _parse_dataset_binding(value: Any) -> TrainingDatasetBinding:
    record = _record(value, {"examples", "sequence_tokens", "sha256"}, "dataset")
    return TrainingDatasetBinding(
        examples=_positive_integer(record["examples"], "dataset.examples"),
        sequence_tokens=_positive_integer(
            record["sequence_tokens"], "dataset.sequence_tokens"
        ),
        sha256=_sha256(record["sha256"], "dataset.sha256"),
    )


def _parse_progress(
    value: Any,
    *,
    training_config: MiniGPTTrainingConfig,
    example_count: int,
) -> dict[str, int | float]:
    record = _record(
        value,
        {
            "global_step",
            "current_learning_rate",
            "epoch",
            "cursor",
            "gradient_accumulation_position",
            "checkpoint_at_optimizer_boundary",
        },
        "progress",
    )
    global_step = _positive_integer(record["global_step"], "progress.global_step")
    if global_step > training_config.schedule.total_updates:
        raise ValueError("training checkpoint global_step exceeds schedule")
    current_lr = _finite_number(
        record["current_learning_rate"], "progress.current_learning_rate"
    )
    expected_lr = training_config.schedule.learning_rate_for_update(global_step - 1)
    if not math.isclose(current_lr, expected_lr, rel_tol=0, abs_tol=1e-15):
        raise ValueError("training checkpoint current learning rate drifted")
    epoch = _nonnegative_integer(record["epoch"], "progress.epoch")
    cursor = _nonnegative_integer(record["cursor"], "progress.cursor")
    if cursor > example_count or cursor % training_config.batch_size != 0:
        raise ValueError("training checkpoint batch cursor is invalid")
    if (
        record["gradient_accumulation_position"] != 0
        or record["checkpoint_at_optimizer_boundary"] is not True
    ):
        raise ValueError("training checkpoint is not at an optimizer boundary")
    return {
        "global_step": global_step,
        "current_learning_rate": current_lr,
        "epoch": epoch,
        "cursor": cursor,
    }


def _parse_tensor_descriptor(value: Any) -> dict[str, Any]:
    record = _record(value, _TENSOR_FIELDS, "tensor descriptor")
    name = _tensor_name(record["name"])
    dtype_name = record["dtype"]
    if dtype_name not in _DTYPES:
        raise ValueError("training tensor dtype is unsupported")
    shape_value = record["shape"]
    if not isinstance(shape_value, list) or not 1 <= len(shape_value) <= 2:
        raise ValueError("training tensor shape must have one or two dimensions")
    return {
        "name": name,
        "dtype": dtype_name,
        "shape": tuple(
            _positive_integer(dimension, "tensor shape dimension")
            for dimension in shape_value
        ),
        "offset": _nonnegative_integer(record["offset"], "tensor offset"),
        "length": _positive_integer(record["length"], "tensor length"),
        "sha256": _sha256(record["sha256"], "tensor sha256"),
    }


def _expected_parameter_shapes(config: GPTConfig) -> dict[str, tuple[int, ...]]:
    dimension = config.model_dim
    hidden = config.mlp_ratio * dimension
    shapes: dict[str, tuple[int, ...]] = {
        "token_embedding.weight": (config.vocab_size, dimension),
        "position_embedding.weight": (config.context_length, dimension),
        "final_norm.bias": (dimension,),
        "final_norm.weight": (dimension,),
    }
    for layer in range(config.num_layers):
        prefix = f"blocks.{layer}"
        shapes.update(
            {
                f"{prefix}.attention_norm.weight": (dimension,),
                f"{prefix}.attention_norm.bias": (dimension,),
                f"{prefix}.attention.qkv.weight": (3 * dimension, dimension),
                f"{prefix}.attention.output.weight": (dimension, dimension),
                f"{prefix}.mlp_norm.weight": (dimension,),
                f"{prefix}.mlp_norm.bias": (dimension,),
                f"{prefix}.mlp.layers.0.weight": (hidden, dimension),
                f"{prefix}.mlp.layers.2.weight": (dimension, hidden),
            }
        )
        if config.bias:
            shapes.update(
                {
                    f"{prefix}.attention.qkv.bias": (3 * dimension,),
                    f"{prefix}.attention.output.bias": (dimension,),
                    f"{prefix}.mlp.layers.0.bias": (hidden,),
                    f"{prefix}.mlp.layers.2.bias": (dimension,),
                }
            )
    return shapes


def _validate_model_structure(model: MiniGPT) -> None:
    if type(model) is not MiniGPT:
        raise TypeError("training model must be MiniGPT")
    config = model.config
    expected_shapes = _expected_parameter_shapes(config)
    named_parameters = dict(model.named_parameters())
    if (
        model.lm_head.weight is not model.token_embedding.weight
        or set(named_parameters) != set(expected_shapes)
        or any(
            tuple(named_parameters[name].shape) != shape
            for name, shape in expected_shapes.items()
        )
    ):
        raise ValueError("training MiniGPT parameter/tied contract drifted")
    if any(
        parameter.device.type != "cpu"
        or parameter.dtype != torch.float32
        or not torch.all(torch.isfinite(parameter))
        for parameter in named_parameters.values()
    ):
        raise ValueError("training MiniGPT parameters must be finite CPU float32")
    if len(model.blocks) != config.num_layers:
        raise ValueError("training MiniGPT block count drifted")
    if not math.isclose(float(model.dropout.p), float(config.dropout)):
        raise ValueError("training MiniGPT embedding dropout/config drifted")
    expected_mask = torch.tril(
        torch.ones(config.context_length, config.context_length, dtype=torch.bool)
    ).view(1, 1, config.context_length, config.context_length)
    for raw_block in model.blocks:
        block = cast(TransformerBlock, raw_block)
        if (
            block.attention.num_heads != config.num_heads
            or block.attention.head_dim != config.model_dim // config.num_heads
            or not math.isclose(
                float(block.attention.attention_dropout), float(config.dropout)
            )
            or not math.isclose(
                float(block.attention.residual_dropout.p), float(config.dropout)
            )
            or not torch.equal(block.attention.causal_mask.cpu(), expected_mask)
        ):
            raise ValueError("training MiniGPT attention/config contract drifted")
        if (
            block.attention_norm.eps != 1e-5
            or block.mlp_norm.eps != 1e-5
            or getattr(block.mlp.layers[1], "approximate", None) != "tanh"
            or not math.isclose(
                float(cast(torch.nn.Dropout, block.mlp.layers[3]).p),
                float(config.dropout),
            )
        ):
            raise ValueError("training MiniGPT norm/MLP architecture drifted")
    if model.final_norm.eps != 1e-5:
        raise ValueError("training MiniGPT final norm architecture drifted")


def _validate_model_size(
    shapes: dict[str, tuple[int, ...]], limits: MiniGPTTrainingCheckpointLimits
) -> None:
    if sum(math.prod(shape) for shape in shapes.values()) > (
        limits.max_model_parameter_count
    ):
        raise ValueError("training checkpoint model parameter count exceeds limit")


def _limits(
    value: MiniGPTTrainingCheckpointLimits | None,
) -> MiniGPTTrainingCheckpointLimits:
    if value is None:
        return MiniGPTTrainingCheckpointLimits()
    if not isinstance(value, MiniGPTTrainingCheckpointLimits):
        raise TypeError("limits must be MiniGPTTrainingCheckpointLimits")
    return value


def _record(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return cast(int, value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return cast(int, value)


def _finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _identity_string(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
        or len(value.encode("utf-8")) > 4096
    ):
        raise ValueError(f"{name} must be trimmed, NUL-free, and <=4096 UTF-8 bytes")
    return value


def _tensor_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
        or len(value.encode("utf-8")) > 2048
    ):
        raise ValueError("training tensor name is invalid")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be canonical sha256:<lowercase-hex>")
    return value
