"""Strict JAX/Optax checkpoint and cross-process resume control."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import multiprocessing as mp
import os
import struct
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty
from typing import Any, NoReturn, cast

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax  # type: ignore[import-untyped]
from jax import Array

from about_llm.from_scratch.gpt_jax import (
    JAXGPTConfig,
    PyTree,
    adamw_optimizer,
    causal_self_attention,
    cross_entropy_loss,
    init_params,
    rms_norm,
)

JAX_TRAINING_RESUME_VERSION = "about-llm.jax-training-resume.v1"
JAX_CHECKPOINT_FORMAT_VERSION = "about-llm.jax-checkpoint.v1"
MAGIC = b"ALLMJAX1"
HEADER = struct.Struct("<8sQ")
DIGEST_BYTES = 32
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
TOTAL_STEPS = 6
SPLIT_STEP = 3
MODEL_SEED = 17
DATA_SEED = 29
DROPOUT_SEED = 41
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
DROPOUT_RATE = 0.2
BATCH_SIZE = 2


@dataclass(frozen=True)
class JAXTrainingState:
    params: PyTree
    optimizer_state: optax.OptState
    dropout_key: Array
    data_key: Array
    permutation: np.ndarray
    cursor: int
    epoch: int
    global_step: int


@dataclass(frozen=True)
class TrainingTrace:
    sample_ids: tuple[tuple[int, ...], ...]
    losses: tuple[float, ...]
    gradient_norms: tuple[float, ...]


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _strict_json(payload: bytes) -> dict[str, object]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("checkpoint manifest must be UTF-8") from error
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("checkpoint manifest is invalid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    if _canonical_bytes(parsed) != payload:
        raise ValueError("checkpoint manifest must use canonical JSON encoding")
    return parsed


def _exact_keys(
    record: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(record) != expected:
        raise ValueError(
            f"{label} fields mismatch: expected {sorted(expected)}, "
            f"got {sorted(record)}"
        )


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be finite")
    return result


def _model_config() -> JAXGPTConfig:
    return JAXGPTConfig(
        vocab_size=13,
        context_length=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2,
    )


def _model_config_payload(config: JAXGPTConfig) -> dict[str, int]:
    return {
        "context_length": config.context_length,
        "mlp_ratio": config.mlp_ratio,
        "model_dim": config.model_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "vocab_size": config.vocab_size,
    }


def _parse_model_config(value: object) -> JAXGPTConfig:
    if not isinstance(value, dict):
        raise ValueError("model_config must be an object")
    _exact_keys(
        value,
        {
            "context_length",
            "mlp_ratio",
            "model_dim",
            "num_heads",
            "num_layers",
            "vocab_size",
        },
        "model_config",
    )
    return JAXGPTConfig(
        vocab_size=_integer(value["vocab_size"], "model_config.vocab_size", minimum=1),
        context_length=_integer(
            value["context_length"],
            "model_config.context_length",
            minimum=1,
        ),
        model_dim=_integer(value["model_dim"], "model_config.model_dim", minimum=1),
        num_heads=_integer(value["num_heads"], "model_config.num_heads", minimum=1),
        num_layers=_integer(
            value["num_layers"],
            "model_config.num_layers",
            minimum=1,
        ),
        mlp_ratio=_integer(value["mlp_ratio"], "model_config.mlp_ratio", minimum=1),
    )


def _optimizer_payload() -> dict[str, float | str]:
    return {
        "kind": "clip-global-norm+adamw",
        "learning_rate": LEARNING_RATE,
        "max_grad_norm": MAX_GRAD_NORM,
        "weight_decay": WEIGHT_DECAY,
    }


def _parse_optimizer(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("optimizer must be an object")
    _exact_keys(
        value,
        {"kind", "learning_rate", "max_grad_norm", "weight_decay"},
        "optimizer",
    )
    if value["kind"] != "clip-global-norm+adamw":
        raise ValueError("optimizer kind mismatch")
    if not math.isclose(
        _finite_float(value["learning_rate"], "optimizer.learning_rate", positive=True),
        LEARNING_RATE,
        rel_tol=0,
        abs_tol=0,
    ):
        raise ValueError("optimizer learning rate mismatch")
    if not math.isclose(
        _finite_float(value["max_grad_norm"], "optimizer.max_grad_norm", positive=True),
        MAX_GRAD_NORM,
        rel_tol=0,
        abs_tol=0,
    ):
        raise ValueError("optimizer max grad norm mismatch")
    if not math.isclose(
        _finite_float(value["weight_decay"], "optimizer.weight_decay"),
        WEIGHT_DECAY,
        rel_tol=0,
        abs_tol=0,
    ):
        raise ValueError("optimizer weight decay mismatch")


def _optimizer() -> optax.GradientTransformation:
    return adamw_optimizer(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
    )


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    input_ids = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
            [6, 7, 8, 9],
        ],
        dtype=np.int32,
    )
    targets = (input_ids + 1).astype(np.int32)
    return input_ids, targets


def _dataset_fingerprint() -> str:
    input_ids, targets = _dataset()
    digest = hashlib.sha256()
    digest.update(input_ids.astype("<i4", copy=False).tobytes(order="C"))
    digest.update(targets.astype("<i4", copy=False).tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _new_permutation(key: Array, count: int) -> tuple[Array, np.ndarray]:
    next_key, permutation_key = jax.random.split(key)
    permutation = np.asarray(
        jax.random.permutation(permutation_key, count),
        dtype=np.int32,
    )
    return next_key, permutation


def create_jax_training_state() -> JAXTrainingState:
    """Create the deterministic initial state used by the resume fixture."""

    config = _model_config()
    params = init_params(jax.random.key(MODEL_SEED), config)
    optimizer_state = _optimizer().init(params)
    data_key, permutation = _new_permutation(
        jax.random.key(DATA_SEED),
        len(_dataset()[0]),
    )
    return JAXTrainingState(
        params=params,
        optimizer_state=optimizer_state,
        dropout_key=jax.random.key(DROPOUT_SEED),
        data_key=data_key,
        permutation=permutation,
        cursor=0,
        epoch=0,
        global_step=0,
    )


def _dropout_forward(
    params: PyTree,
    input_ids: Array,
    dropout_key: Array,
    config: JAXGPTConfig,
) -> Array:
    sequence_length = input_ids.shape[1]
    hidden_states = params["token_embedding"][input_ids]
    hidden_states = hidden_states + params["position_embedding"][
        jnp.arange(sequence_length)
    ]
    keep_probability = 1.0 - DROPOUT_RATE
    keep = jax.random.bernoulli(
        dropout_key,
        keep_probability,
        hidden_states.shape,
    )
    hidden_states = jnp.where(
        keep,
        hidden_states / keep_probability,
        jnp.zeros_like(hidden_states),
    )
    for block in params["blocks"]:
        attention_input = rms_norm(hidden_states, block["attention_norm"])
        hidden_states = hidden_states + causal_self_attention(
            attention_input,
            block,
            config,
        )
        mlp_input = rms_norm(hidden_states, block["mlp_norm"])
        activated = jax.nn.gelu(
            mlp_input @ block["up"],
            approximate=True,
        )
        hidden_states = hidden_states + activated @ block["down"]
    hidden_states = rms_norm(hidden_states, params["final_norm"])
    return cast(Array, hidden_states @ params["token_embedding"].T)


def _make_train_step() -> Any:
    config = _model_config()
    optimizer = _optimizer()

    def step(
        params: PyTree,
        optimizer_state: optax.OptState,
        dropout_key: Array,
        input_ids: Array,
        targets: Array,
    ) -> tuple[PyTree, optax.OptState, Array, Array, Array]:
        next_key, step_key = jax.random.split(dropout_key)

        def loss_function(current_params: PyTree) -> Array:
            logits = _dropout_forward(
                current_params,
                input_ids,
                step_key,
                config,
            )
            return cross_entropy_loss(logits, targets)

        loss, gradients = jax.value_and_grad(loss_function)(params)
        gradient_norm = optax.tree.norm(gradients)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            params,
        )
        next_params = optax.apply_updates(params, updates)
        return (
            cast(PyTree, next_params),
            next_optimizer_state,
            next_key,
            loss,
            gradient_norm,
        )

    return jax.jit(step)


def _next_batch(
    state: JAXTrainingState,
) -> tuple[JAXTrainingState, np.ndarray]:
    dataset_size = len(_dataset()[0])
    selected: list[int] = []
    permutation = state.permutation.copy()
    cursor = state.cursor
    epoch = state.epoch
    data_key = state.data_key
    while len(selected) < BATCH_SIZE:
        if cursor == dataset_size:
            epoch += 1
            data_key, permutation = _new_permutation(data_key, dataset_size)
            cursor = 0
        take = min(BATCH_SIZE - len(selected), dataset_size - cursor)
        selected.extend(int(value) for value in permutation[cursor : cursor + take])
        cursor += take
    return (
        replace(
            state,
            data_key=data_key,
            permutation=permutation,
            cursor=cursor,
            epoch=epoch,
        ),
        np.asarray(selected, dtype=np.int32),
    )


def train_jax_steps(
    state: JAXTrainingState,
    *,
    steps: int,
) -> tuple[JAXTrainingState, TrainingTrace]:
    """Train a fixed number of steps while advancing RNG and data state."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    train_step = _make_train_step()
    input_table, target_table = _dataset()
    sample_ids: list[tuple[int, ...]] = []
    losses: list[float] = []
    gradient_norms: list[float] = []
    current = state
    for _ in range(steps):
        current, indices = _next_batch(current)
        params, optimizer_state, dropout_key, loss, gradient_norm = train_step(
            current.params,
            current.optimizer_state,
            current.dropout_key,
            jnp.asarray(input_table[indices]),
            jnp.asarray(target_table[indices]),
        )
        loss.block_until_ready()
        current = replace(
            current,
            params=params,
            optimizer_state=optimizer_state,
            dropout_key=dropout_key,
            global_step=current.global_step + 1,
        )
        sample_ids.append(tuple(int(index) for index in indices))
        losses.append(float(loss))
        gradient_norms.append(float(gradient_norm))
    return current, TrainingTrace(
        sample_ids=tuple(sample_ids),
        losses=tuple(losses),
        gradient_norms=tuple(gradient_norms),
    )


def _portable_array(value: object) -> tuple[np.ndarray, str]:
    array = np.asarray(value)
    dtype_map = {
        np.dtype("float32"): (np.dtype("<f4"), "float32"),
        np.dtype("int32"): (np.dtype("<i4"), "int32"),
        np.dtype("uint32"): (np.dtype("<u4"), "uint32"),
    }
    if array.dtype not in dtype_map:
        raise ValueError(f"unsupported checkpoint dtype: {array.dtype}")
    portable_dtype, label = dtype_map[array.dtype]
    portable = np.asarray(array, dtype=portable_dtype, order="C")
    if portable.dtype.kind == "f" and not np.all(np.isfinite(portable)):
        raise ValueError("checkpoint arrays must be finite")
    return portable, label


def _checkpoint_arrays(state: JAXTrainingState) -> list[tuple[str, object]]:
    param_leaves = jax.tree.leaves(state.params)
    optimizer_leaves = jax.tree.leaves(state.optimizer_state)
    arrays: list[tuple[str, object]] = []
    arrays.extend(
        (f"params/{index:04d}", value)
        for index, value in enumerate(param_leaves)
    )
    arrays.extend(
        (f"optimizer/{index:04d}", value)
        for index, value in enumerate(optimizer_leaves)
    )
    arrays.extend(
        [
            ("dropout_key", jax.random.key_data(state.dropout_key)),
            ("data_key", jax.random.key_data(state.data_key)),
            ("permutation", state.permutation),
        ]
    )
    return arrays


def serialize_jax_training_checkpoint(state: JAXTrainingState) -> bytes:
    """Serialize a strict deterministic checkpoint for the authored fixture."""

    _validate_state(state)
    payload_parts: list[bytes] = []
    descriptors: list[dict[str, object]] = []
    offset = 0
    for name, value in _checkpoint_arrays(state):
        array, dtype_label = _portable_array(value)
        content = array.tobytes(order="C")
        descriptors.append(
            {
                "dtype": dtype_label,
                "name": name,
                "nbytes": len(content),
                "offset": offset,
                "sha256": hashlib.sha256(content).hexdigest(),
                "shape": list(array.shape),
            }
        )
        payload_parts.append(content)
        offset += len(content)
    manifest: dict[str, object] = {
        "arrays": descriptors,
        "cursor": state.cursor,
        "dataset_fingerprint": _dataset_fingerprint(),
        "epoch": state.epoch,
        "format_version": JAX_CHECKPOINT_FORMAT_VERSION,
        "global_step": state.global_step,
        "model_config": _model_config_payload(_model_config()),
        "optimizer": _optimizer_payload(),
    }
    manifest_bytes = _canonical_bytes(manifest)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("checkpoint manifest exceeds the configured limit")
    prefix = (
        HEADER.pack(MAGIC, len(manifest_bytes))
        + manifest_bytes
        + b"".join(payload_parts)
    )
    artifact = prefix + hashlib.sha256(prefix).digest()
    if len(artifact) > MAX_ARTIFACT_BYTES:
        raise ValueError("checkpoint artifact exceeds the configured limit")
    return artifact


def write_jax_training_checkpoint(
    path: str | Path,
    state: JAXTrainingState,
) -> dict[str, object]:
    """Write an exclusive checkpoint file and fsync its contents."""

    target = Path(path)
    artifact = serialize_jax_training_checkpoint(state)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(artifact)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "artifact_bytes": len(artifact),
        "artifact_sha256": "sha256:" + hashlib.sha256(artifact).hexdigest(),
    }


def _parse_descriptors(
    value: object,
    payload: bytes,
) -> dict[str, np.ndarray]:
    if not isinstance(value, list) or not value:
        raise ValueError("arrays must be a non-empty list")
    arrays: dict[str, np.ndarray] = {}
    expected_offset = 0
    dtype_map = {
        "float32": np.dtype("<f4"),
        "int32": np.dtype("<i4"),
        "uint32": np.dtype("<u4"),
    }
    for index, descriptor_value in enumerate(value):
        if not isinstance(descriptor_value, dict):
            raise ValueError(f"arrays[{index}] must be an object")
        _exact_keys(
            descriptor_value,
            {"dtype", "name", "nbytes", "offset", "sha256", "shape"},
            f"arrays[{index}]",
        )
        name = descriptor_value["name"]
        if not isinstance(name, str) or not name or name in arrays:
            raise ValueError("array names must be unique non-empty strings")
        dtype_label = descriptor_value["dtype"]
        if dtype_label not in dtype_map:
            raise ValueError(f"unsupported array dtype: {dtype_label}")
        shape_value = descriptor_value["shape"]
        if not isinstance(shape_value, list):
            raise ValueError("array shape must be a list")
        shape = tuple(
            _integer(dimension, "array dimension") for dimension in shape_value
        )
        offset = _integer(descriptor_value["offset"], "array offset")
        nbytes = _integer(descriptor_value["nbytes"], "array nbytes")
        if offset != expected_offset:
            raise ValueError("array payload offsets must be contiguous")
        dtype = dtype_map[cast(str, dtype_label)]
        expected_nbytes = math.prod(shape) * dtype.itemsize
        if nbytes != expected_nbytes or offset + nbytes > len(payload):
            raise ValueError("array nbytes/shape exceeds payload")
        content = payload[offset : offset + nbytes]
        expected_hash = descriptor_value["sha256"]
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or hashlib.sha256(content).hexdigest() != expected_hash
        ):
            raise ValueError("array payload digest mismatch")
        array = np.frombuffer(content, dtype=dtype).reshape(shape).copy()
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError("checkpoint arrays must be finite")
        arrays[name] = array
        expected_offset += nbytes
    if expected_offset != len(payload):
        raise ValueError("unreferenced bytes remain in checkpoint payload")
    return arrays


def _expected_array_names(
    param_count: int,
    optimizer_count: int,
) -> list[str]:
    return [
        *(f"params/{index:04d}" for index in range(param_count)),
        *(f"optimizer/{index:04d}" for index in range(optimizer_count)),
        "dropout_key",
        "data_key",
        "permutation",
    ]


def parse_jax_training_checkpoint(artifact: bytes) -> JAXTrainingState:
    """Parse and validate a checkpoint before constructing JAX arrays."""

    if not isinstance(artifact, bytes):
        raise TypeError("artifact must be bytes")
    if len(artifact) > MAX_ARTIFACT_BYTES:
        raise ValueError("checkpoint artifact exceeds the configured limit")
    if len(artifact) < HEADER.size + DIGEST_BYTES:
        raise ValueError("checkpoint artifact is truncated")
    magic, manifest_length = HEADER.unpack_from(artifact)
    if magic != MAGIC:
        raise ValueError("checkpoint magic mismatch")
    if manifest_length > MAX_MANIFEST_BYTES:
        raise ValueError("checkpoint manifest exceeds the configured limit")
    manifest_end = HEADER.size + manifest_length
    if manifest_end > len(artifact) - DIGEST_BYTES:
        raise ValueError("checkpoint manifest is truncated")
    prefix = artifact[:-DIGEST_BYTES]
    if not hmac.compare_digest(
        hashlib.sha256(prefix).digest(),
        artifact[-DIGEST_BYTES:],
    ):
        raise ValueError("checkpoint outer digest mismatch")
    manifest_bytes = artifact[HEADER.size:manifest_end]
    manifest = _strict_json(manifest_bytes)
    _exact_keys(
        manifest,
        {
            "arrays",
            "cursor",
            "dataset_fingerprint",
            "epoch",
            "format_version",
            "global_step",
            "model_config",
            "optimizer",
        },
        "manifest",
    )
    if manifest["format_version"] != JAX_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint format version mismatch")
    if manifest["dataset_fingerprint"] != _dataset_fingerprint():
        raise ValueError("checkpoint dataset binding mismatch")
    config = _parse_model_config(manifest["model_config"])
    if config != _model_config():
        raise ValueError("checkpoint model config mismatch")
    _parse_optimizer(manifest["optimizer"])
    payload = artifact[manifest_end:-DIGEST_BYTES]
    arrays = _parse_descriptors(manifest["arrays"], payload)

    template_params = init_params(jax.random.key(MODEL_SEED), config)
    template_optimizer_state = _optimizer().init(template_params)
    param_leaves, param_tree = jax.tree.flatten(template_params)
    optimizer_leaves, optimizer_tree = jax.tree.flatten(
        template_optimizer_state
    )
    expected_names = _expected_array_names(
        len(param_leaves),
        len(optimizer_leaves),
    )
    if list(arrays) != expected_names:
        raise ValueError("checkpoint array ordering/schema mismatch")

    restored_param_leaves: list[Array] = []
    for index, template in enumerate(param_leaves):
        array = arrays[f"params/{index:04d}"]
        if array.shape != template.shape or str(array.dtype) != str(template.dtype):
            raise ValueError("checkpoint parameter shape/dtype mismatch")
        restored_param_leaves.append(jnp.asarray(array))
    restored_optimizer_leaves: list[Array] = []
    for index, template in enumerate(optimizer_leaves):
        array = arrays[f"optimizer/{index:04d}"]
        if array.shape != template.shape or str(array.dtype) != str(template.dtype):
            raise ValueError("checkpoint optimizer shape/dtype mismatch")
        restored_optimizer_leaves.append(jnp.asarray(array))

    dropout_key_data = arrays["dropout_key"]
    data_key_data = arrays["data_key"]
    if dropout_key_data.shape != (2,) or data_key_data.shape != (2,):
        raise ValueError("checkpoint PRNG key-data shape mismatch")
    permutation = arrays["permutation"]
    dataset_size = len(_dataset()[0])
    if (
        permutation.shape != (dataset_size,)
        or permutation.dtype != np.dtype("int32")
        or sorted(int(value) for value in permutation)
        != list(range(dataset_size))
    ):
        raise ValueError("checkpoint permutation is invalid")
    state = JAXTrainingState(
        params=cast(PyTree, jax.tree.unflatten(param_tree, restored_param_leaves)),
        optimizer_state=jax.tree.unflatten(
            optimizer_tree,
            restored_optimizer_leaves,
        ),
        dropout_key=jax.random.wrap_key_data(jnp.asarray(dropout_key_data)),
        data_key=jax.random.wrap_key_data(jnp.asarray(data_key_data)),
        permutation=permutation.copy(),
        cursor=_integer(manifest["cursor"], "cursor"),
        epoch=_integer(manifest["epoch"], "epoch"),
        global_step=_integer(manifest["global_step"], "global_step"),
    )
    _validate_state(state)
    return state


def load_jax_training_checkpoint(path: str | Path) -> JAXTrainingState:
    target = Path(path)
    artifact_size = target.stat().st_size
    if artifact_size > MAX_ARTIFACT_BYTES:
        raise ValueError("checkpoint artifact exceeds the configured limit")
    return parse_jax_training_checkpoint(target.read_bytes())


def _validate_state(state: JAXTrainingState) -> None:
    if not isinstance(state, JAXTrainingState):
        raise TypeError("state must be JAXTrainingState")
    dataset_size = len(_dataset()[0])
    if state.permutation.shape != (dataset_size,):
        raise ValueError("state permutation shape mismatch")
    if sorted(int(value) for value in state.permutation) != list(
        range(dataset_size)
    ):
        raise ValueError("state permutation must contain each sample once")
    if state.cursor < 0 or state.cursor > dataset_size:
        raise ValueError("state cursor is out of range")
    if state.epoch < 0 or state.global_step < 0:
        raise ValueError("state counters must be non-negative")
    key_shapes = (
        jax.random.key_data(state.dropout_key).shape,
        jax.random.key_data(state.data_key).shape,
    )
    if key_shapes != ((2,), (2,)):
        raise ValueError("state PRNG keys must be scalar typed keys")


def _state_fingerprint(state: JAXTrainingState) -> str:
    digest = hashlib.sha256()
    for name, value in _checkpoint_arrays(state):
        array, dtype_label = _portable_array(value)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(dtype_label.encode("ascii") + b"\0")
        digest.update(_canonical_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    digest.update(
        _canonical_bytes(
            {
                "cursor": state.cursor,
                "epoch": state.epoch,
                "global_step": state.global_step,
            }
        )
    )
    return "sha256:" + digest.hexdigest()


def _parameter_max_difference(
    left: JAXTrainingState,
    right: JAXTrainingState,
) -> float:
    differences: list[float] = []
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left.params),
        jax.tree.leaves(right.params),
        strict=True,
    ):
        left_array = np.asarray(left_leaf, dtype=np.float64)
        right_array = np.asarray(right_leaf, dtype=np.float64)
        differences.append(float(np.max(np.abs(left_array - right_array))))
    return max(differences, default=0.0)


def _trace_payload(trace: TrainingTrace) -> dict[str, object]:
    return {
        "gradient_norms": list(trace.gradient_norms),
        "losses": list(trace.losses),
        "sample_ids": [list(batch) for batch in trace.sample_ids],
    }


def _worker_checkpoint(
    checkpoint_path: str,
    output_queue: Any,
) -> None:
    state = create_jax_training_state()
    state, trace = train_jax_steps(state, steps=SPLIT_STEP)
    artifact = write_jax_training_checkpoint(checkpoint_path, state)
    output_queue.put(
        {
            "artifact": artifact,
            "checkpoint_state_fingerprint": _state_fingerprint(state),
            "pid": os.getpid(),
            "trace": _trace_payload(trace),
        }
    )


def _worker_resume(
    checkpoint_path: str,
    output_queue: Any,
) -> None:
    restored = load_jax_training_checkpoint(checkpoint_path)
    remaining = TOTAL_STEPS - restored.global_step
    correct, correct_trace = train_jax_steps(restored, steps=remaining)

    wrong_rng_start = replace(
        restored,
        dropout_key=jax.random.key(DROPOUT_SEED),
    )
    wrong_rng, wrong_rng_trace = train_jax_steps(
        wrong_rng_start,
        steps=remaining,
    )
    wrong_cursor_start = replace(restored, cursor=0)
    wrong_cursor, wrong_cursor_trace = train_jax_steps(
        wrong_cursor_start,
        steps=remaining,
    )
    output_queue.put(
        {
            "correct_final_fingerprint": _state_fingerprint(correct),
            "correct_trace": _trace_payload(correct_trace),
            "pid": os.getpid(),
            "wrong_cursor_final_fingerprint": _state_fingerprint(wrong_cursor),
            "wrong_cursor_parameter_max_abs_difference": (
                _parameter_max_difference(correct, wrong_cursor)
            ),
            "wrong_cursor_trace": _trace_payload(wrong_cursor_trace),
            "wrong_rng_final_fingerprint": _state_fingerprint(wrong_rng),
            "wrong_rng_parameter_max_abs_difference": (
                _parameter_max_difference(correct, wrong_rng)
            ),
            "wrong_rng_trace": _trace_payload(wrong_rng_trace),
        }
    )


def _run_worker(
    target: Any,
    *args: str,
    timeout: float = 180.0,
) -> dict[str, object]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=target, args=(*args, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(10)
        raise RuntimeError("JAX resume worker timed out")
    if process.exitcode != 0:
        raise RuntimeError(f"JAX resume worker exited with {process.exitcode}")
    try:
        result = queue.get(timeout=10)
    except Empty as error:
        raise RuntimeError("JAX resume worker returned no report") from error
    finally:
        queue.close()
        queue.join_thread()
    if not isinstance(result, dict):
        raise RuntimeError("JAX resume worker report is invalid")
    return cast(dict[str, object], result)


def run_jax_training_resume_control() -> dict[str, object]:
    """Compare uninterrupted and cross-process checkpoint-resumed training."""

    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("a JAX CPU device is required")
    with jax.default_device(cpu_devices[0]):
        initial = create_jax_training_state()
        uninterrupted, uninterrupted_trace = train_jax_steps(
            initial,
            steps=TOTAL_STEPS,
        )
        uninterrupted_fingerprint = _state_fingerprint(uninterrupted)

    with tempfile.TemporaryDirectory(prefix="about-llm-jax-resume-") as temporary:
        checkpoint_path = str(Path(temporary) / "checkpoint.allmjax")
        first = _run_worker(_worker_checkpoint, checkpoint_path)
        second = _run_worker(_worker_resume, checkpoint_path)

    first_trace = cast(dict[str, object], first["trace"])
    correct_trace = cast(dict[str, object], second["correct_trace"])
    resumed_sample_ids = cast(list[object], first_trace["sample_ids"]) + cast(
        list[object], correct_trace["sample_ids"]
    )
    resumed_losses = cast(list[float], first_trace["losses"]) + cast(
        list[float], correct_trace["losses"]
    )
    resumed_gradient_norms = cast(
        list[float], first_trace["gradient_norms"]
    ) + cast(list[float], correct_trace["gradient_norms"])
    uninterrupted_payload = _trace_payload(uninterrupted_trace)
    correct_fingerprint = second["correct_final_fingerprint"]
    assertions = {
        "phase_workers_are_distinct_processes": first["pid"] != second["pid"],
        "split_checkpoint_state_is_at_committed_step_three": (
            first_trace["sample_ids"]
            == cast(list[object], uninterrupted_payload["sample_ids"])[:SPLIT_STEP]
            and first["checkpoint_state_fingerprint"]
            != uninterrupted_fingerprint
        ),
        "cross_process_resume_is_bit_exact": (
            correct_fingerprint == uninterrupted_fingerprint
            and resumed_sample_ids == uninterrupted_payload["sample_ids"]
            and resumed_losses == uninterrupted_payload["losses"]
            and resumed_gradient_norms == uninterrupted_payload["gradient_norms"]
        ),
        "reset_dropout_prng_diverges": (
            second["wrong_rng_final_fingerprint"] != correct_fingerprint
            and cast(float, second["wrong_rng_parameter_max_abs_difference"])
            > 0
        ),
        "reset_data_cursor_diverges": (
            second["wrong_cursor_final_fingerprint"] != correct_fingerprint
            and cast(float, second["wrong_cursor_parameter_max_abs_difference"])
            > 0
            and cast(dict[str, object], second["wrong_cursor_trace"])[
                "sample_ids"
            ]
            != correct_trace["sample_ids"]
        ),
    }
    if not all(assertions.values()):
        raise AssertionError(f"JAX resume assertion failed: {assertions}")

    report: dict[str, object] = {
        "schema_version": JAX_TRAINING_RESUME_VERSION,
        "runtime": {
            "jax_backend": "cpu",
            "jax_device": str(cpu_devices[0]),
            "jax_version": jax.__version__,
            "jaxlib_version": getattr(jaxlib, "__version__", "unknown"),
            "optax_version": optax.__version__,
            "process_start_method": "spawn",
            "dtype": "float32",
        },
        "fixture": {
            "model_config": _model_config_payload(_model_config()),
            "optimizer": _optimizer_payload(),
            "dropout_rate": DROPOUT_RATE,
            "batch_size": BATCH_SIZE,
            "dataset_examples": len(_dataset()[0]),
            "dataset_fingerprint": _dataset_fingerprint(),
            "model_seed": MODEL_SEED,
            "data_seed": DATA_SEED,
            "dropout_seed": DROPOUT_SEED,
            "total_steps": TOTAL_STEPS,
            "split_step": SPLIT_STEP,
        },
        "artifact": first["artifact"],
        "process_observation": {
            "distinct_phase_worker_count": 2,
            "raw_process_ids_published": False,
        },
        "uninterrupted": {
            "final_state_fingerprint": uninterrupted_fingerprint,
            "trace": uninterrupted_payload,
        },
        "resumed": {
            "final_state_fingerprint": correct_fingerprint,
            "trace": {
                "gradient_norms": resumed_gradient_norms,
                "losses": resumed_losses,
                "sample_ids": resumed_sample_ids,
            },
        },
        "counterfactuals": {
            "reset_dropout_prng": {
                "final_state_fingerprint": second[
                    "wrong_rng_final_fingerprint"
                ],
                "parameter_max_abs_difference": second[
                    "wrong_rng_parameter_max_abs_difference"
                ],
                "trace": second["wrong_rng_trace"],
            },
            "reset_data_cursor": {
                "final_state_fingerprint": second[
                    "wrong_cursor_final_fingerprint"
                ],
                "parameter_max_abs_difference": second[
                    "wrong_cursor_parameter_max_abs_difference"
                ],
                "trace": second["wrong_cursor_trace"],
            },
        },
        "assertions": assertions,
        "scope": {
            "strict_canonical_manifest_and_outer_digest_executed": True,
            "parameter_and_optax_state_restored": True,
            "dropout_prng_and_data_shuffle_state_restored": True,
            "cross_process_split_resume_executed": True,
            "bit_exact_full_state_and_trace_compared": True,
            "wrong_prng_and_cursor_counterfactuals_executed": True,
            "exclusive_create_and_file_fsync_executed": True,
            "directory_fsync_or_power_loss_atomicity_proved": False,
            "orbax_flax_tensorstore_or_distributed_checkpoint_executed": False,
            "cuda_tpu_multi_device_or_sharding_executed": False,
            "python_numpy_worker_or_accelerator_rng_restored": False,
            "target_model_dataset_convergence_or_performance_proved": False,
            "artifact_origin_authentication_or_confidentiality_proved": False,
        },
    }
    report["report_fingerprint"] = "sha256:" + hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    return report


__all__ = [
    "JAX_CHECKPOINT_FORMAT_VERSION",
    "JAX_TRAINING_RESUME_VERSION",
    "JAXTrainingState",
    "create_jax_training_state",
    "load_jax_training_checkpoint",
    "parse_jax_training_checkpoint",
    "run_jax_training_resume_control",
    "serialize_jax_training_checkpoint",
    "train_jax_steps",
    "write_jax_training_checkpoint",
]
