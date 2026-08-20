"""Collect and verify the pinned Qwen3-0.6B through nano-vLLM study."""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import hmac
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Final, cast

MANIFEST_VERSION: Final = "about-llm.nano-vllm-study-manifest.v1"
REPORT_VERSION: Final = "about-llm.nano-vllm-study.v1"
WORKER_VERSION: Final = "about-llm.nano-vllm-study-worker.v1"
MAX_MANIFEST_BYTES: Final = 64_000
MAX_REPORT_BYTES: Final = 64_000_000
SUBPROCESS_TIMEOUT_SECONDS: Final = 7_200

SOURCE_REPOSITORY: Final = "https://github.com/GeeeekExplorer/nano-vllm.git"
SOURCE_REVISION: Final = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"
MODEL_REPOSITORY: Final = "Qwen/Qwen3-0.6B"
MODEL_REVISION: Final = "c1899de289a04d12100db370d81485cdf75e47ca"

NANO_VLLM_STUDY_EVIDENCE_BOUNDARY: Final = (
    "This report is a local, version-bound observation of Qwen/Qwen3-0.6B at revision "
    "c1899de289a04d12100db370d81485cdf75e47ca running through GeeeekExplorer/nano-vllm "
    "at commit bb823b3e06983d71485a8e1f23715ebd87d98ef8. It records scheduler, sequence, "
    "Paged KV, prefix-cache, eager/CUDA Graph, timing, throughput, and CUDA memory "
    "observations for the declared synthetic token workload only. It does not publish "
    "raw prompts, measure model quality, prove production capacity or SLOs, compare "
    "against Transformers or vLLM, generalize to another driver/GPU/version/workload, "
    "or prove the absence of hidden synchronization and measurement effects. Artifact "
    "hashing happens before worker execution and does not remove local TOCTOU risk. The "
    "unkeyed report fingerprint detects accidental drift but does not authenticate the "
    "recorder or publisher. Failed cases are evidence of that run's terminal state, not "
    "zero performance."
)

_LOCKED_SOURCE: Final = {
    "repository": SOURCE_REPOSITORY,
    "revision": SOURCE_REVISION,
    "package_version": "0.2.0",
    "require_clean_checkout": True,
}
_LOCKED_MODEL_IDENTITY: Final = {
    "repository": MODEL_REPOSITORY,
    "revision": MODEL_REVISION,
    "architecture": "Qwen3ForCausalLM",
    "model_type": "qwen3",
    "torch_dtype": "bfloat16",
    "vocab_size": 151936,
}
_LOCKED_ARTIFACTS: Final = [
    {
        "filename": "config.json",
        "size_bytes": 726,
        "sha256": "sha256:660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    },
    {
        "filename": "generation_config.json",
        "size_bytes": 239,
        "sha256": "sha256:2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    },
    {
        "filename": "merges.txt",
        "size_bytes": 1_671_853,
        "sha256": "sha256:8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    },
    {
        "filename": "model.safetensors",
        "size_bytes": 1_503_300_328,
        "sha256": "sha256:f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    },
    {
        "filename": "tokenizer.json",
        "size_bytes": 11_422_654,
        "sha256": "sha256:aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    },
    {
        "filename": "tokenizer_config.json",
        "size_bytes": 9_732,
        "sha256": "sha256:d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    },
    {
        "filename": "vocab.json",
        "size_bytes": 2_776_833,
        "sha256": "sha256:ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
]
_LOCKED_ENGINE: Final = {
    "tensor_parallel_size": 1,
    "max_num_seqs": 8,
    "max_model_len": 1024,
    "gpu_memory_utilization": 0.8,
    "kvcache_block_size": 256,
}
_LOCKED_WORKLOAD: Final = {
    "prompt_tokens": 768,
    "output_tokens": 8,
    "temperature": 1.0,
    "prefix_cached_blocks": 2,
    "drift_token_index": 256,
    "execution_modes": ["eager", "cuda_graph"],
    "prefix_variants": ["exact", "one_token_drift"],
    "max_num_batched_tokens": [256, 1024],
    "concurrency": [1, 2, 4, 8],
    "seed": 20260820,
}
_LOCKED_COLLECTION: Final = {
    "warmup_runs": 1,
    "measurement_runs": 5,
    "subprocess_isolation": True,
    "publish_raw_prompts": False,
}

_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "study_id",
        "source",
        "model",
        "engine",
        "workload",
        "collection",
        "manifest_fingerprint",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "report_version",
        "study_id",
        "manifest_fingerprint",
        "collected_at",
        "source",
        "model",
        "runtime",
        "hardware",
        "collection",
        "cases",
        "evidence_boundary",
        "report_fingerprint",
    }
)
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "execution_mode",
        "max_num_batched_tokens",
        "prefix_variant",
        "concurrency",
        "status",
        "failure",
        "engine",
        "samples",
        "summary",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "kind",
        "sample_index",
        "seed",
        "status",
        "failure",
        "primer",
        "engine_finished_ns",
        "requests",
        "prefix_hits",
        "trace",
        "metrics",
        "kv_released",
    }
)
_METRIC_NAMES: Final = (
    "ttft_ms",
    "tpot_ms",
    "e2e_ms",
    "output_tokens_per_second",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _load_strict_object(path: Path, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label}") from error
    if not raw or len(raw) > maximum_bytes:
        raise ValueError(f"{label} size is invalid")
    return _strict_json_bytes(raw, label=label)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value is not canonical finite JSON") from error


def _fingerprint(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def fingerprint_document(value: Mapping[str, Any], *, field: str) -> str:
    """Return the canonical SHA-256 after removing one fingerprint field."""

    unsigned = copy.deepcopy(dict(value))
    supplied = unsigned.pop(field, None)
    if supplied is not None and not isinstance(supplied, str):
        raise ValueError(f"{field} must be a string")
    return _fingerprint(unsigned)


def _exact(value: Mapping[str, Any], fields: frozenset[str], location: str) -> None:
    if frozenset(value) != fields:
        raise ValueError(f"{location} fields are invalid")


def _as_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return cast(Mapping[str, Any], value)


def _as_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    return value


def _finite_number(value: Any, location: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{location} is outside its valid range")
    return number


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer")
    return cast(int, value)


def _nonnegative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return cast(int, value)


def load_study_manifest(path: Path) -> dict[str, Any]:
    """Load the one reviewed study manifest with strict identity checks."""

    manifest = _load_strict_object(
        path, maximum_bytes=MAX_MANIFEST_BYTES, label="nano-vLLM study manifest"
    )
    _exact(manifest, _MANIFEST_FIELDS, "manifest")
    supplied = manifest.get("manifest_fingerprint")
    if not isinstance(supplied, str) or not hmac.compare_digest(
        supplied, fingerprint_document(manifest, field="manifest_fingerprint")
    ):
        raise ValueError("manifest fingerprint mismatch")
    if (
        manifest.get("manifest_version") != MANIFEST_VERSION
        or manifest.get("study_id") != "qwen3-0.6b-through-nano-vllm"
        or manifest.get("source") != _LOCKED_SOURCE
        or manifest.get("engine") != _LOCKED_ENGINE
        or manifest.get("workload") != _LOCKED_WORKLOAD
        or manifest.get("collection") != _LOCKED_COLLECTION
    ):
        raise ValueError("manifest study identity or workload drift")
    model = _as_mapping(manifest.get("model"), "manifest.model")
    identity = {key: model.get(key) for key in _LOCKED_MODEL_IDENTITY}
    if identity != _LOCKED_MODEL_IDENTITY or model.get("artifacts") != _LOCKED_ARTIFACTS:
        raise ValueError("manifest model revision or artifact drift")
    if frozenset(model) != frozenset({*_LOCKED_MODEL_IDENTITY, "artifacts"}):
        raise ValueError("manifest.model fields are invalid")
    return copy.deepcopy(manifest)


def _run_git(source_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise ValueError("cannot inspect nano-vLLM git checkout") from error
    return result.stdout.strip()


def _normalize_repository(url: str) -> str:
    normalized = url.strip().replace("git@github.com:", "https://github.com/")
    return normalized.removesuffix("/").removesuffix(".git").lower()


def inspect_source_checkout(source_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind collection to the reviewed clean nano-vLLM checkout."""

    root = source_root.resolve()
    if not root.is_dir():
        raise ValueError("nano-vLLM source root is not a directory")
    source = _as_mapping(manifest.get("source"), "manifest.source")
    head = _run_git(root, "rev-parse", "HEAD")
    if head != source.get("revision") or head != SOURCE_REVISION:
        raise ValueError("nano-vLLM source revision drift")
    dirty = _run_git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    if dirty:
        raise ValueError("nano-vLLM source checkout is dirty")
    origin = _run_git(root, "remote", "get-url", "origin")
    expected = cast(str, source["repository"])
    if _normalize_repository(origin) != _normalize_repository(expected):
        raise ValueError("nano-vLLM origin repository drift")
    return {
        "repository": expected,
        "revision": head,
        "package_version": source["package_version"],
        "clean_checkout": True,
        "origin_verified": True,
        "source_path_published": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError("cannot hash model snapshot artifact") from error
    return "sha256:" + digest.hexdigest()


def inspect_model_snapshot(model_snapshot: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Hash every model/runtime artifact selected by the reviewed manifest."""

    root = model_snapshot.resolve()
    if not root.is_dir():
        raise ValueError("model snapshot is not a directory")
    model = _as_mapping(manifest.get("model"), "manifest.model")
    observed: list[dict[str, Any]] = []
    for item_value in _as_list(model.get("artifacts"), "manifest.model.artifacts"):
        item = _as_mapping(item_value, "manifest.model.artifacts[]")
        filename = item.get("filename")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError("model artifact filename is unsafe")
        artifact = root / filename
        if not artifact.is_file():
            raise ValueError(f"model snapshot artifact is missing: {filename}")
        size = artifact.stat().st_size
        if size != item.get("size_bytes"):
            raise ValueError(f"model snapshot artifact size drift: {filename}")
        sha256 = _sha256_file(artifact)
        expected_sha = item.get("sha256")
        if not isinstance(expected_sha, str) or not hmac.compare_digest(sha256, expected_sha):
            raise ValueError(f"model snapshot artifact hash drift: {filename}")
        observed.append(
            {
                "filename": filename,
                "size_bytes": size,
                "sha256": sha256,
                "verified": True,
            }
        )
    config = _load_strict_object(root / "config.json", maximum_bytes=64_000, label="Qwen config")
    if (
        config.get("architectures") != [model.get("architecture")]
        or config.get("model_type") != model.get("model_type")
        or config.get("torch_dtype") != model.get("torch_dtype")
        or config.get("vocab_size") != model.get("vocab_size")
    ):
        raise ValueError("Qwen model config identity drift")
    return {
        "repository": model["repository"],
        "revision": model["revision"],
        "architecture": model["architecture"],
        "model_type": model["model_type"],
        "torch_dtype": model["torch_dtype"],
        "vocab_size": model["vocab_size"],
        "artifacts": observed,
        "selected_file_count": len(observed),
        "selected_total_bytes": sum(cast(int, item["size_bytes"]) for item in observed),
        "snapshot_path_published": False,
        "raw_prompt_published": False,
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _nvidia_driver_version() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise ValueError("cannot record NVIDIA driver identity") from error
    version = result.stdout.strip().splitlines()
    if len(version) != 1 or not version[0]:
        raise ValueError("NVIDIA driver identity is ambiguous")
    return version[0]


def _runtime_and_hardware(torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if torch.cuda.is_available() is not True or torch.cuda.device_count() < 1:
        raise ValueError("CUDA device 0 is required for nano-vLLM collection")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    cuda_runtime = torch.version.cuda
    if not isinstance(cuda_runtime, str) or not cuda_runtime:
        raise ValueError("PyTorch CUDA runtime identity is missing")
    runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": cast(str, torch.__version__),
        "cuda_runtime": cuda_runtime,
        "nccl": ".".join(str(part) for part in torch.cuda.nccl.version()),
        "transformers": _package_version("transformers"),
        "flash_attn": _package_version("flash-attn"),
        "triton": _package_version("triton"),
        "xxhash": _package_version("xxhash"),
        "nano_vllm": _LOCKED_SOURCE["package_version"],
        "cuda_available": True,
    }
    hardware = {
        "device_index": 0,
        "device_name": cast(str, properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "driver_version": _nvidia_driver_version(),
        "device_count_visible": int(torch.cuda.device_count()),
    }
    return runtime, hardware


def _case_id(mode: str, batched_tokens: int, prefix_variant: str, concurrency: int) -> str:
    return f"{mode}-mbt{batched_tokens}-{prefix_variant}-c{concurrency}"


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _prompt_tokens(vocab_size: int, length: int, namespace: int) -> list[int]:
    lower = 1_000
    width = vocab_size - 2 * lower
    if width <= 0:
        raise ValueError("model vocabulary is too small for synthetic workload")
    return [lower + ((namespace * 7_919 + index * 104_729) % width) for index in range(length)]


def _active_sequences(scheduler: Any) -> list[Any]:
    by_id: dict[int, Any] = {}
    for sequence in [*scheduler.waiting, *scheduler.running]:
        by_id[int(sequence.seq_id)] = sequence
    return [by_id[key] for key in sorted(by_id)]


def _sequence_projection(sequence: Any, request_ids: Mapping[int, str]) -> dict[str, Any]:
    return {
        "request_id": request_ids.get(int(sequence.seq_id), "unmeasured"),
        "sequence_id": int(sequence.seq_id),
        "status": str(sequence.status.name).lower(),
        "prompt_tokens": int(sequence.num_prompt_tokens),
        "completion_tokens": int(sequence.num_completion_tokens),
        "cached_tokens": int(sequence.num_cached_tokens),
        "scheduled_tokens": int(sequence.num_scheduled_tokens),
        "block_count": len(sequence.block_table),
        "is_prefill": bool(sequence.is_prefill),
    }


def _sequence_list(sequences: Sequence[Any], request_ids: Mapping[int, str]) -> list[Any]:
    return [
        _sequence_projection(sequence, request_ids)
        for sequence in sorted(sequences, key=lambda item: int(item.seq_id))
    ]


def _kv_snapshot(scheduler: Any) -> dict[str, Any]:
    manager = scheduler.block_manager
    ref_counts = [int(block.ref_count) for block in manager.blocks]
    active = _active_sequences(scheduler)
    return {
        "total_blocks": len(manager.blocks),
        "free_blocks": len(manager.free_block_ids),
        "used_blocks": len(manager.used_block_ids),
        "ref_count_total": sum(ref_counts),
        "block_table_references": sum(len(sequence.block_table) for sequence in active),
        "shared_blocks": sum(ref_count > 1 for ref_count in ref_counts),
        "max_ref_count": max(ref_counts, default=0),
        "cached_hash_entries": len(manager.hash_to_block_id),
    }


def _assert_live_kv_invariants(snapshot: Mapping[str, Any]) -> None:
    if (
        snapshot["free_blocks"] + snapshot["used_blocks"] != snapshot["total_blocks"]
        or snapshot["ref_count_total"] != snapshot["block_table_references"]
        or (snapshot["used_blocks"] == 0 and snapshot["ref_count_total"] != 0)
    ):
        raise RuntimeError("nano-vLLM KV ledger invariant failed during collection")


def _capture_schedule(
    original_schedule: Any,
    scheduler: Any,
    request_ids: Mapping[int, str],
    first_prefix_hits: dict[int, int],
    block_size: int,
    mode: str,
    captured: dict[str, Any],
) -> tuple[list[Any], bool]:
    scheduled, is_prefill = original_schedule()
    kv_scheduled = _kv_snapshot(scheduler)
    _assert_live_kv_invariants(kv_scheduled)
    scheduled_tokens = sum(int(sequence.num_scheduled_tokens) for sequence in scheduled)
    if is_prefill:
        incomplete = any(
            int(sequence.num_cached_tokens) + int(sequence.num_scheduled_tokens)
            < int(sequence.num_tokens)
            for sequence in scheduled
        )
        phase = "chunked_prefill" if incomplete else "prefill"
        execution_path = "eager"
        for sequence in scheduled:
            first_prefix_hits.setdefault(
                int(sequence.seq_id), int(sequence.num_cached_tokens) // block_size
            )
    else:
        phase = "decode"
        execution_path = "eager" if mode == "eager" else "cuda_graph"
    captured.update(
        {
            "scheduled": scheduled,
            "is_prefill": is_prefill,
            "kv_scheduled": kv_scheduled,
            "scheduled_tokens": scheduled_tokens,
            "phase": phase,
            "execution_path": execution_path,
            "before_completion": {
                int(sequence.seq_id): int(sequence.num_completion_tokens) for sequence in scheduled
            },
            "scheduled_projection": _sequence_list(scheduled, request_ids),
        }
    )
    return cast(tuple[list[Any], bool], (scheduled, is_prefill))


def _prime_prefix(engine: Any, prompt: list[int], sampling_type: Any, torch: Any) -> dict[str, Any]:
    params = sampling_type(temperature=1.0, max_tokens=1, ignore_eos=True)
    engine.add_request(prompt, params)
    while not engine.is_finished():
        engine.step()
    torch.cuda.synchronize()
    snapshot = _kv_snapshot(engine.scheduler)
    _assert_live_kv_invariants(snapshot)
    if snapshot["used_blocks"] != 0 or snapshot["ref_count_total"] != 0:
        raise RuntimeError("prefix primer did not release every KV block")
    return {
        "status": "success",
        "cached_block_entries": snapshot["cached_hash_entries"],
        "kv_after": snapshot,
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute a median from no values")
    return float(statistics.median(values))


def _run_measurement(
    engine: Any,
    prompts: list[list[int]],
    sampling_type: Any,
    torch: Any,
    *,
    seed: int,
    mode: str,
    block_size: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter_ns()
    request_states: dict[int, dict[str, Any]] = {}
    request_ids: dict[int, str] = {}
    params = sampling_type(temperature=1.0, max_tokens=8, ignore_eos=True)
    for index, prompt in enumerate(prompts):
        engine.add_request(prompt, params)
        sequence = engine.scheduler.waiting[-1]
        sequence_id = int(sequence.seq_id)
        request_id = f"req-{index}"
        request_ids[sequence_id] = request_id
        request_states[sequence_id] = {
            "request_id": request_id,
            "sequence_id": sequence_id,
            "added_ns": time.perf_counter_ns() - started,
            "first_token_ns": None,
            "finished_ns": None,
            "completion_tokens": 0,
        }

    trace: list[dict[str, Any]] = []
    first_prefix_hits: dict[int, int] = {}
    previous_step_finished = 0
    while not engine.is_finished():
        scheduler = engine.scheduler
        before_sequences = _active_sequences(scheduler)
        kv_before = _kv_snapshot(scheduler)
        _assert_live_kv_invariants(kv_before)
        step_started = time.perf_counter_ns() - started
        captured: dict[str, Any] = {}
        original_schedule = scheduler.schedule
        scheduler.schedule = partial(
            _capture_schedule,
            original_schedule,
            scheduler,
            request_ids,
            first_prefix_hits,
            block_size,
            mode,
            captured,
        )
        try:
            _, engine_num_tokens = engine.step()
        finally:
            scheduler.schedule = original_schedule
        step_finished = time.perf_counter_ns() - started
        scheduled = cast(list[Any], captured["scheduled"])
        is_prefill = cast(bool, captured["is_prefill"])
        kv_scheduled = cast(dict[str, Any], captured["kv_scheduled"])
        scheduled_tokens = cast(int, captured["scheduled_tokens"])
        phase = cast(str, captured["phase"])
        execution_path = cast(str, captured["execution_path"])
        before_completion = cast(dict[int, int], captured["before_completion"])
        scheduled_projection = cast(list[Any], captured["scheduled_projection"])
        expected_engine_tokens = scheduled_tokens if is_prefill else -len(scheduled)
        if engine_num_tokens != expected_engine_tokens:
            raise RuntimeError("nano-vLLM step token-accounting drift")
        committed = 0
        for sequence in scheduled:
            sequence_id = int(sequence.seq_id)
            current = int(sequence.num_completion_tokens)
            prior = before_completion[sequence_id]
            if current > prior:
                committed += current - prior
                if request_states[sequence_id]["first_token_ns"] is None:
                    request_states[sequence_id]["first_token_ns"] = step_finished
            request_states[sequence_id]["completion_tokens"] = current
            if bool(sequence.is_finished):
                request_states[sequence_id]["finished_ns"] = step_finished
        after_sequences = _active_sequences(scheduler)
        kv_after = _kv_snapshot(scheduler)
        _assert_live_kv_invariants(kv_after)
        if step_started < previous_step_finished or step_finished < step_started:
            raise RuntimeError("monotonic step timing order failed during collection")
        previous_step_finished = step_finished
        trace.append(
            {
                "step_index": len(trace),
                "phase": phase,
                "execution_path": execution_path,
                "started_ns": step_started,
                "finished_ns": step_finished,
                "scheduled_tokens": scheduled_tokens,
                "scheduled_sequence_ids": [int(sequence.seq_id) for sequence in scheduled],
                "sequences_before": _sequence_list(before_sequences, request_ids),
                "sequences_scheduled": scheduled_projection,
                "sequences_after": _sequence_list(after_sequences, request_ids),
                "kv_before": kv_before,
                "kv_scheduled": kv_scheduled,
                "kv_after": kv_after,
                "committed_token_count": committed,
            }
        )

    torch.cuda.synchronize()
    engine_finished = time.perf_counter_ns() - started
    requests = [request_states[key] for key in sorted(request_states)]
    if any(
        request["first_token_ns"] is None
        or request["finished_ns"] is None
        or request["completion_tokens"] != 8
        for request in requests
    ):
        raise RuntimeError("nano-vLLM request did not reach the reviewed terminal state")
    ttft_values = [
        (cast(int, item["first_token_ns"]) - cast(int, item["added_ns"])) / 1e6 for item in requests
    ]
    tpot_values = [
        (cast(int, item["finished_ns"]) - cast(int, item["first_token_ns"]))
        / (cast(int, item["completion_tokens"]) - 1)
        / 1e6
        for item in requests
    ]
    e2e_values = [
        (cast(int, item["finished_ns"]) - cast(int, item["added_ns"])) / 1e6 for item in requests
    ]
    total_output_tokens = sum(cast(int, item["completion_tokens"]) for item in requests)
    duration_ms = engine_finished / 1e6
    metrics = {
        "duration_ms": duration_ms,
        "total_output_tokens": total_output_tokens,
        "ttft_ms": _median(ttft_values),
        "tpot_ms": _median(tpot_values),
        "e2e_ms": _median(e2e_values),
        "output_tokens_per_second": total_output_tokens / (engine_finished / 1e9),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    released = _kv_snapshot(engine.scheduler)
    _assert_live_kv_invariants(released)
    prefix_hits = [
        {
            "request_id": request_ids[sequence_id],
            "sequence_id": sequence_id,
            "hit_blocks": first_prefix_hits[sequence_id],
        }
        for sequence_id in sorted(request_ids)
    ]
    return {
        "engine_finished_ns": engine_finished,
        "requests": requests,
        "prefix_hits": prefix_hits,
        "trace": trace,
        "metrics": metrics,
        "kv_released": released,
    }


def _failed_sample(
    kind: str, index: int, seed: int, stage: str, error: Exception
) -> dict[str, Any]:
    return {
        "kind": kind,
        "sample_index": index,
        "seed": seed,
        "status": "failed",
        "failure": {"stage": stage, "error_type": type(error).__name__},
        "primer": None,
        "engine_finished_ns": None,
        "requests": [],
        "prefix_hits": [],
        "trace": [],
        "metrics": None,
        "kv_released": None,
    }


def _run_sample(
    engine: Any,
    sampling_type: Any,
    torch: Any,
    manifest: Mapping[str, Any],
    *,
    kind: str,
    index: int,
    seed: int,
    mode: str,
    prefix_variant: str,
    concurrency: int,
) -> dict[str, Any]:
    workload = _as_mapping(manifest["workload"], "manifest.workload")
    length = cast(int, workload["prompt_tokens"])
    vocab_size = cast(int, _as_mapping(manifest["model"], "manifest.model")["vocab_size"])
    namespace = seed
    primer_prompt = _prompt_tokens(vocab_size, length, namespace)
    measured_prompt = list(primer_prompt)
    if prefix_variant == "one_token_drift":
        drift_index = cast(int, workload["drift_token_index"])
        measured_prompt[drift_index] = (measured_prompt[drift_index] + 1) % vocab_size
        if measured_prompt[drift_index] == primer_prompt[drift_index]:
            raise RuntimeError("one-token drift construction failed")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        primer = _prime_prefix(engine, primer_prompt, sampling_type, torch)
    except Exception as error:
        return _failed_sample(kind, index, seed, "prefix_primer", error)
    try:
        measured = _run_measurement(
            engine,
            [list(measured_prompt) for _ in range(concurrency)],
            sampling_type,
            torch,
            seed=seed + 1,
            mode=mode,
            block_size=cast(
                int, _as_mapping(manifest["engine"], "manifest.engine")["kvcache_block_size"]
            ),
        )
    except Exception as error:
        failed = _failed_sample(kind, index, seed, "measurement", error)
        failed["primer"] = primer
        return failed
    return {
        "kind": kind,
        "sample_index": index,
        "seed": seed,
        "status": "success",
        "failure": None,
        "primer": primer,
        **measured,
    }


def _summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    measured = [sample for sample in samples if sample.get("kind") == "measurement"]
    if not measured or any(sample.get("status") != "success" for sample in measured):
        return None
    raw: dict[str, list[Any]] = {name: [] for name in _METRIC_NAMES}
    for sample in measured:
        metrics = _as_mapping(sample.get("metrics"), "sample.metrics")
        for name in _METRIC_NAMES:
            raw[name].append(metrics[name])
    medians = {name: statistics.median(values) for name, values in raw.items()}
    return {
        "measurement_count": len(measured),
        "raw": raw,
        "median": medians,
    }


def _case_failure(stage: str, error_type: str) -> dict[str, str]:
    return {"stage": stage, "error_type": error_type}


def _unrun_case(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    batched_tokens: int,
    prefix_variant: str,
    concurrency: int,
    failure: Mapping[str, str],
) -> dict[str, Any]:
    engine_config = _as_mapping(manifest["engine"], "manifest.engine")
    return {
        "case_id": _case_id(mode, batched_tokens, prefix_variant, concurrency),
        "execution_mode": mode,
        "max_num_batched_tokens": batched_tokens,
        "prefix_variant": prefix_variant,
        "concurrency": concurrency,
        "status": "failed",
        "failure": dict(failure),
        "engine": {
            **copy.deepcopy(dict(engine_config)),
            "enforce_eager": mode == "eager",
            "num_kvcache_blocks": None,
        },
        "samples": [],
        "summary": None,
    }


def _run_worker(
    manifest_path: Path,
    source_root: Path,
    model_snapshot: Path,
    output: Path,
    *,
    mode: str,
    batched_tokens: int,
) -> int:
    manifest = load_study_manifest(manifest_path)
    inspect_source_checkout(source_root, manifest)
    sys.path.insert(0, str(source_root.resolve()))
    torch = importlib.import_module("torch")
    nano_vllm = importlib.import_module("nanovllm")
    transformers = importlib.import_module("transformers")
    llm_type = nano_vllm.LLM
    sampling_type = nano_vllm.SamplingParams

    runtime, hardware = _runtime_and_hardware(torch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_snapshot, use_fast=True, local_files_only=True
    )
    tokenizer_identity = {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
    }
    workload = _as_mapping(manifest["workload"], "manifest.workload")
    engine_manifest = _as_mapping(manifest["engine"], "manifest.engine")
    cases: list[dict[str, Any]] = []
    engine: Any | None = None
    engine_error: Exception | None = None
    try:
        engine = llm_type(
            str(model_snapshot.resolve()),
            enforce_eager=mode == "eager",
            max_num_batched_tokens=batched_tokens,
            max_num_seqs=engine_manifest["max_num_seqs"],
            max_model_len=engine_manifest["max_model_len"],
            gpu_memory_utilization=engine_manifest["gpu_memory_utilization"],
            tensor_parallel_size=engine_manifest["tensor_parallel_size"],
            kvcache_block_size=engine_manifest["kvcache_block_size"],
        )
    except Exception as error:
        engine_error = error

    for prefix_variant_value in cast(list[Any], workload["prefix_variants"]):
        prefix_variant = cast(str, prefix_variant_value)
        for concurrency_value in cast(list[Any], workload["concurrency"]):
            concurrency = cast(int, concurrency_value)
            if engine is None:
                assert engine_error is not None
                cases.append(
                    _unrun_case(
                        manifest,
                        mode=mode,
                        batched_tokens=batched_tokens,
                        prefix_variant=prefix_variant,
                        concurrency=concurrency,
                        failure=_case_failure("engine_init", type(engine_error).__name__),
                    )
                )
                continue
            case_id = _case_id(mode, batched_tokens, prefix_variant, concurrency)
            samples: list[dict[str, Any]] = []
            base_seed = cast(int, workload["seed"])
            warmup_seed = _stable_seed(base_seed, case_id, "warmup")
            samples.append(
                _run_sample(
                    engine,
                    sampling_type,
                    torch,
                    manifest,
                    kind="warmup",
                    index=0,
                    seed=warmup_seed,
                    mode=mode,
                    prefix_variant=prefix_variant,
                    concurrency=concurrency,
                )
            )
            for index in range(
                cast(int, _as_mapping(manifest["collection"], "collection")["measurement_runs"])
            ):
                sample_seed = _stable_seed(base_seed, case_id, "measurement", str(index))
                samples.append(
                    _run_sample(
                        engine,
                        sampling_type,
                        torch,
                        manifest,
                        kind="measurement",
                        index=index,
                        seed=sample_seed,
                        mode=mode,
                        prefix_variant=prefix_variant,
                        concurrency=concurrency,
                    )
                )
            success = all(sample["status"] == "success" for sample in samples)
            first_failure = next(
                (
                    cast(Mapping[str, str], sample["failure"])
                    for sample in samples
                    if sample["status"] != "success"
                ),
                None,
            )
            cases.append(
                {
                    "case_id": case_id,
                    "execution_mode": mode,
                    "max_num_batched_tokens": batched_tokens,
                    "prefix_variant": prefix_variant,
                    "concurrency": concurrency,
                    "status": "success" if success else "failed",
                    "failure": None if success else dict(cast(Mapping[str, str], first_failure)),
                    "engine": {
                        **copy.deepcopy(dict(engine_manifest)),
                        "enforce_eager": mode == "eager",
                        "num_kvcache_blocks": int(engine.model_runner.config.num_kvcache_blocks),
                    },
                    "samples": samples,
                    "summary": _summarize_samples(samples) if success else None,
                }
            )
    if engine is not None:
        atexit.unregister(engine.exit)
        engine.exit()
    worker = {
        "worker_version": WORKER_VERSION,
        "execution_mode": mode,
        "max_num_batched_tokens": batched_tokens,
        "runtime": runtime,
        "hardware": hardware,
        "tokenizer": tokenizer_identity,
        "cases": cases,
    }
    _write_json(output, worker)
    return 0


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def collect_study(
    manifest_path: Path,
    source_root: Path,
    model_snapshot: Path,
    output: Path,
) -> dict[str, Any]:
    """Run four isolated engine workers and write one self-verifying report."""

    manifest = load_study_manifest(manifest_path)
    source = inspect_source_checkout(source_root, manifest)
    model = inspect_model_snapshot(model_snapshot, manifest)
    workload = _as_mapping(manifest["workload"], "manifest.workload")
    worker_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="about-llm-nano-vllm-study-") as directory:
        temporary_root = Path(directory)
        for mode_value in cast(list[Any], workload["execution_modes"]):
            mode = cast(str, mode_value)
            for batched_value in cast(list[Any], workload["max_num_batched_tokens"]):
                batched_tokens = cast(int, batched_value)
                worker_output = temporary_root / f"{mode}-{batched_tokens}.json"
                command = [
                    sys.executable,
                    "-m",
                    "about_llm.inference.nano_vllm_study",
                    "_worker",
                    "--manifest",
                    str(manifest_path.resolve()),
                    "--source-root",
                    str(source_root.resolve()),
                    "--model-snapshot",
                    str(model_snapshot.resolve()),
                    "--output",
                    str(worker_output),
                    "--mode",
                    mode,
                    "--max-num-batched-tokens",
                    str(batched_tokens),
                ]
                try:
                    result = subprocess.run(
                        command,
                        check=False,
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        timeout=SUBPROCESS_TIMEOUT_SECONDS,
                    )
                except (OSError, subprocess.SubprocessError) as error:
                    raise RuntimeError("nano-vLLM study worker could not run") from error
                if result.returncode != 0 or not worker_output.is_file():
                    raise RuntimeError(
                        f"nano-vLLM study worker failed for {mode}/{batched_tokens} "
                        f"with exit code {result.returncode}"
                    )
                worker = _load_strict_object(
                    worker_output,
                    maximum_bytes=MAX_REPORT_BYTES,
                    label="nano-vLLM worker report",
                )
                if (
                    worker.get("worker_version") != WORKER_VERSION
                    or worker.get("execution_mode") != mode
                    or worker.get("max_num_batched_tokens") != batched_tokens
                ):
                    raise ValueError("nano-vLLM worker identity drift")
                worker_results.append(worker)
    if not worker_results:
        raise RuntimeError("nano-vLLM study produced no workers")
    runtime = worker_results[0].get("runtime")
    hardware = worker_results[0].get("hardware")
    tokenizer = worker_results[0].get("tokenizer")
    if any(
        worker.get("runtime") != runtime
        or worker.get("hardware") != hardware
        or worker.get("tokenizer") != tokenizer
        for worker in worker_results[1:]
    ):
        raise ValueError("runtime, hardware, or tokenizer identity drift across workers")
    model.update(cast(Mapping[str, Any], tokenizer))
    cases = [
        cast(dict[str, Any], case)
        for worker in worker_results
        for case in _as_list(worker.get("cases"), "worker.cases")
    ]
    success_count = sum(case.get("status") == "success" for case in cases)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "study_id": manifest["study_id"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "model": model,
        "runtime": runtime,
        "hardware": hardware,
        "collection": {
            **copy.deepcopy(cast(dict[str, Any], manifest["collection"])),
            "worker_count": len(worker_results),
            "successful_cases": success_count,
            "failed_cases": len(cases) - success_count,
        },
        "cases": cases,
        "evidence_boundary": NANO_VLLM_STUDY_EVIDENCE_BOUNDARY,
    }
    report["report_fingerprint"] = _fingerprint(report)
    verified = verify_study_report(manifest, report)
    _write_json(output, verified)
    return verified


def _verify_kv(snapshot_value: Any, location: str, *, require_released: bool = False) -> None:
    snapshot = _as_mapping(snapshot_value, location)
    fields = frozenset(
        {
            "total_blocks",
            "free_blocks",
            "used_blocks",
            "ref_count_total",
            "block_table_references",
            "shared_blocks",
            "max_ref_count",
            "cached_hash_entries",
        }
    )
    _exact(snapshot, fields, location)
    values = {name: _nonnegative_int(snapshot[name], f"{location}.{name}") for name in fields}
    if values["total_blocks"] <= 0:
        raise ValueError(f"{location}.total_blocks must be positive")
    if (
        values["free_blocks"] + values["used_blocks"] != values["total_blocks"]
        or values["ref_count_total"] != values["block_table_references"]
        or values["shared_blocks"] > values["used_blocks"]
        or (values["used_blocks"] == 0 and values["ref_count_total"] != 0)
        or (values["ref_count_total"] == 0 and values["max_ref_count"] != 0)
    ):
        raise ValueError(f"{location} KV ledger invariant failed")
    if require_released and (
        values["used_blocks"] != 0
        or values["ref_count_total"] != 0
        or values["free_blocks"] != values["total_blocks"]
    ):
        raise ValueError(f"{location} does not prove complete KV release")


def _verify_sequence_projection(value: Any, location: str) -> None:
    projection = _as_mapping(value, location)
    _exact(
        projection,
        frozenset(
            {
                "request_id",
                "sequence_id",
                "status",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "scheduled_tokens",
                "block_count",
                "is_prefill",
            }
        ),
        location,
    )
    if not isinstance(projection["request_id"], str) or not projection["request_id"]:
        raise ValueError(f"{location}.request_id is invalid")
    _nonnegative_int(projection["sequence_id"], f"{location}.sequence_id")
    if projection["status"] not in {"waiting", "running", "finished"}:
        raise ValueError(f"{location}.status is invalid")
    _positive_int(projection["prompt_tokens"], f"{location}.prompt_tokens")
    for name in ("completion_tokens", "cached_tokens", "scheduled_tokens", "block_count"):
        _nonnegative_int(projection[name], f"{location}.{name}")
    if not isinstance(projection["is_prefill"], bool):
        raise ValueError(f"{location}.is_prefill must be boolean")


def _verify_trace(
    trace_value: Any,
    *,
    location: str,
    engine_finished_ns: int,
    execution_mode: str,
    max_num_batched_tokens: int,
) -> tuple[dict[int, int], int]:
    trace = _as_list(trace_value, location)
    if not trace:
        raise ValueError(f"{location} must not be empty")
    first_cached_tokens: dict[int, int] = {}
    committed_total = 0
    prior_finished = 0
    for index, step_value in enumerate(trace):
        step = _as_mapping(step_value, f"{location}[{index}]")
        _exact(
            step,
            frozenset(
                {
                    "step_index",
                    "phase",
                    "execution_path",
                    "started_ns",
                    "finished_ns",
                    "scheduled_tokens",
                    "scheduled_sequence_ids",
                    "sequences_before",
                    "sequences_scheduled",
                    "sequences_after",
                    "kv_before",
                    "kv_scheduled",
                    "kv_after",
                    "committed_token_count",
                }
            ),
            f"{location}[{index}]",
        )
        if step["step_index"] != index or step["phase"] not in {
            "prefill",
            "chunked_prefill",
            "decode",
        }:
            raise ValueError(f"{location}[{index}] phase/index drift")
        expected_path = (
            "eager" if step["phase"] != "decode" or execution_mode == "eager" else "cuda_graph"
        )
        if step["execution_path"] != expected_path:
            raise ValueError(f"{location}[{index}] execution-path drift")
        started = _nonnegative_int(step["started_ns"], f"{location}[{index}].started_ns")
        finished = _nonnegative_int(step["finished_ns"], f"{location}[{index}].finished_ns")
        if started < prior_finished or finished < started or finished > engine_finished_ns:
            raise ValueError(f"{location}[{index}] timing order failed")
        prior_finished = finished
        scheduled_tokens = _positive_int(
            step["scheduled_tokens"], f"{location}[{index}].scheduled_tokens"
        )
        if step["phase"] != "decode" and scheduled_tokens > max_num_batched_tokens:
            raise ValueError(f"{location}[{index}] prefill token budget exceeded")
        committed = _nonnegative_int(
            step["committed_token_count"],
            f"{location}[{index}].committed_token_count",
        )
        committed_total += committed
        scheduled_ids = _as_list(
            step["scheduled_sequence_ids"], f"{location}[{index}].scheduled_sequence_ids"
        )
        if not scheduled_ids:
            raise ValueError(f"{location}[{index}] scheduled no sequences")
        for name in ("sequences_before", "sequences_scheduled", "sequences_after"):
            projections = _as_list(step[name], f"{location}[{index}].{name}")
            for projection_index, projection in enumerate(projections):
                _verify_sequence_projection(
                    projection, f"{location}[{index}].{name}[{projection_index}]"
                )
                if name == "sequences_scheduled" and step["phase"] != "decode":
                    projection_map = cast(Mapping[str, Any], projection)
                    first_cached_tokens.setdefault(
                        cast(int, projection_map["sequence_id"]),
                        cast(int, projection_map["cached_tokens"]),
                    )
            if name == "sequences_scheduled":
                projection_ids = [
                    cast(Mapping[str, Any], projection)["sequence_id"] for projection in projections
                ]
                projection_tokens = sum(
                    cast(int, cast(Mapping[str, Any], projection)["scheduled_tokens"])
                    for projection in projections
                )
                if projection_ids != scheduled_ids or projection_tokens != scheduled_tokens:
                    raise ValueError(f"{location}[{index}] scheduled projection drift")
                if committed > len(projections):
                    raise ValueError(f"{location}[{index}] committed too many tokens")
                if step["phase"] == "decode" and scheduled_tokens != len(projections):
                    raise ValueError(
                        f"{location}[{index}] decode must schedule one token per sequence"
                    )
        for name in ("kv_before", "kv_scheduled", "kv_after"):
            _verify_kv(step[name], f"{location}[{index}].{name}")
    return first_cached_tokens, committed_total


def _assert_close(observed: Any, expected: float, location: str) -> None:
    number = _finite_number(observed, location, minimum=0.0)
    if not math.isclose(number, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{location} arithmetic drift")


def _verify_successful_sample(
    sample: Mapping[str, Any],
    *,
    location: str,
    concurrency: int,
    prefix_variant: str,
    execution_mode: str,
    block_size: int,
    expected_hit_blocks: int,
    max_num_batched_tokens: int,
) -> None:
    primer = _as_mapping(sample["primer"], f"{location}.primer")
    _exact(primer, frozenset({"status", "cached_block_entries", "kv_after"}), f"{location}.primer")
    if primer["status"] != "success":
        raise ValueError(f"{location}.primer status drift")
    _positive_int(primer["cached_block_entries"], f"{location}.primer.cached_block_entries")
    _verify_kv(primer["kv_after"], f"{location}.primer.kv_after", require_released=True)
    engine_finished = _positive_int(sample["engine_finished_ns"], f"{location}.engine_finished_ns")
    requests = _as_list(sample["requests"], f"{location}.requests")
    if len(requests) != concurrency:
        raise ValueError(f"{location}.requests concurrency drift")
    ttft_values: list[float] = []
    tpot_values: list[float] = []
    e2e_values: list[float] = []
    sequence_ids: set[int] = set()
    total_tokens = 0
    for index, request_value in enumerate(requests):
        request = _as_mapping(request_value, f"{location}.requests[{index}]")
        _exact(
            request,
            frozenset(
                {
                    "request_id",
                    "sequence_id",
                    "added_ns",
                    "first_token_ns",
                    "finished_ns",
                    "completion_tokens",
                }
            ),
            f"{location}.requests[{index}]",
        )
        if request["request_id"] != f"req-{index}":
            raise ValueError(f"{location}.requests order drift")
        sequence_id = _nonnegative_int(
            request["sequence_id"], f"{location}.requests[{index}].sequence_id"
        )
        if sequence_id in sequence_ids:
            raise ValueError(f"{location}.requests duplicate sequence id")
        sequence_ids.add(sequence_id)
        added = _nonnegative_int(request["added_ns"], f"{location}.requests[{index}].added_ns")
        first = _nonnegative_int(
            request["first_token_ns"], f"{location}.requests[{index}].first_token_ns"
        )
        finished = _nonnegative_int(
            request["finished_ns"], f"{location}.requests[{index}].finished_ns"
        )
        completion_tokens = _positive_int(
            request["completion_tokens"],
            f"{location}.requests[{index}].completion_tokens",
        )
        if not added <= first <= finished <= engine_finished:
            raise ValueError(f"{location}.requests[{index}] timing order failed")
        if completion_tokens <= 1:
            raise ValueError(f"{location}.requests[{index}] TPOT is undefined")
        ttft_values.append((first - added) / 1e6)
        tpot_values.append((finished - first) / (completion_tokens - 1) / 1e6)
        e2e_values.append((finished - added) / 1e6)
        total_tokens += completion_tokens
    first_cached, committed_total = _verify_trace(
        sample["trace"],
        location=f"{location}.trace",
        engine_finished_ns=engine_finished,
        execution_mode=execution_mode,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    if committed_total != total_tokens:
        raise ValueError(f"{location}.trace committed-token arithmetic drift")
    prefix_hits = _as_list(sample["prefix_hits"], f"{location}.prefix_hits")
    if len(prefix_hits) != concurrency:
        raise ValueError(f"{location}.prefix_hits concurrency drift")
    for index, hit_value in enumerate(prefix_hits):
        hit = _as_mapping(hit_value, f"{location}.prefix_hits[{index}]")
        _exact(
            hit,
            frozenset({"request_id", "sequence_id", "hit_blocks"}),
            f"{location}.prefix_hits[{index}]",
        )
        sequence_id = cast(int, hit["sequence_id"])
        if (
            hit["request_id"] != f"req-{index}"
            or sequence_id not in sequence_ids
            or hit["hit_blocks"] != expected_hit_blocks
            or first_cached.get(sequence_id) != expected_hit_blocks * block_size
        ):
            raise ValueError(f"{location}.prefix_hits observed prefix drift")
    metrics = _as_mapping(sample["metrics"], f"{location}.metrics")
    _exact(
        metrics,
        frozenset(
            {
                "duration_ms",
                "total_output_tokens",
                *_METRIC_NAMES,
            }
        ),
        f"{location}.metrics",
    )
    _assert_close(metrics["duration_ms"], engine_finished / 1e6, f"{location}.metrics.duration_ms")
    if metrics["total_output_tokens"] != total_tokens:
        raise ValueError(f"{location}.metrics total-output arithmetic drift")
    _assert_close(metrics["ttft_ms"], _median(ttft_values), f"{location}.metrics.ttft_ms")
    _assert_close(metrics["tpot_ms"], _median(tpot_values), f"{location}.metrics.tpot_ms")
    _assert_close(metrics["e2e_ms"], _median(e2e_values), f"{location}.metrics.e2e_ms")
    _assert_close(
        metrics["output_tokens_per_second"],
        total_tokens / (engine_finished / 1e9),
        f"{location}.metrics.output_tokens_per_second",
    )
    allocated = _nonnegative_int(
        metrics["peak_allocated_bytes"], f"{location}.metrics.peak_allocated_bytes"
    )
    reserved = _nonnegative_int(
        metrics["peak_reserved_bytes"], f"{location}.metrics.peak_reserved_bytes"
    )
    if allocated > reserved:
        raise ValueError(f"{location}.metrics allocated memory exceeds reserved memory")
    _verify_kv(sample["kv_released"], f"{location}.kv_released", require_released=True)
    if prefix_variant not in {"exact", "one_token_drift"}:
        raise ValueError(f"{location} prefix variant drift")


def _verify_sample(
    sample_value: Any,
    *,
    location: str,
    expected_kind: str,
    expected_index: int,
    concurrency: int,
    prefix_variant: str,
    execution_mode: str,
    block_size: int,
    expected_hit_blocks: int,
    max_num_batched_tokens: int,
) -> None:
    sample = _as_mapping(sample_value, location)
    _exact(sample, _SAMPLE_FIELDS, location)
    if (
        sample["kind"] != expected_kind
        or sample["sample_index"] != expected_index
        or sample["status"] not in {"success", "failed"}
    ):
        raise ValueError(f"{location} kind/index/status drift")
    _nonnegative_int(sample["seed"], f"{location}.seed")
    if sample["status"] == "failed":
        failure = _as_mapping(sample["failure"], f"{location}.failure")
        _exact(failure, frozenset({"stage", "error_type"}), f"{location}.failure")
        if failure["stage"] not in {"prefix_primer", "measurement"}:
            raise ValueError(f"{location}.failure stage drift")
        if not isinstance(failure["error_type"], str) or not failure["error_type"]:
            raise ValueError(f"{location}.failure error type is invalid")
        if any(sample[name] for name in ("requests", "prefix_hits", "trace")):
            raise ValueError(f"{location} failed sample published partial ambiguous data")
        if sample["engine_finished_ns"] is not None or sample["metrics"] is not None:
            raise ValueError(f"{location} failed sample metric drift")
        if sample["kv_released"] is not None:
            raise ValueError(f"{location} failed sample cannot claim KV release")
        return
    if sample["failure"] is not None:
        raise ValueError(f"{location} successful sample has a failure")
    _verify_successful_sample(
        sample,
        location=location,
        concurrency=concurrency,
        prefix_variant=prefix_variant,
        execution_mode=execution_mode,
        block_size=block_size,
        expected_hit_blocks=expected_hit_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
    )


def _verify_summary(
    case: Mapping[str, Any], measurement_samples: Sequence[Mapping[str, Any]]
) -> None:
    summary = _as_mapping(case["summary"], f"case[{case['case_id']}].summary")
    _exact(summary, frozenset({"measurement_count", "raw", "median"}), "case.summary")
    if summary["measurement_count"] != len(measurement_samples):
        raise ValueError("case.summary measurement count drift")
    raw = _as_mapping(summary["raw"], "case.summary.raw")
    medians = _as_mapping(summary["median"], "case.summary.median")
    if frozenset(raw) != frozenset(_METRIC_NAMES) or frozenset(medians) != frozenset(_METRIC_NAMES):
        raise ValueError("case.summary metric fields drift")
    for name in _METRIC_NAMES:
        expected = [
            cast(Mapping[str, Any], sample["metrics"])[name] for sample in measurement_samples
        ]
        if raw[name] != expected:
            raise ValueError(f"case.summary raw {name} drift")
        observed_median = medians[name]
        expected_median = statistics.median(expected)
        if isinstance(expected_median, float):
            _assert_close(observed_median, expected_median, f"case.summary.median.{name}")
        elif observed_median != expected_median:
            raise ValueError(f"case.summary median {name} drift")


def _expected_case_keys(manifest: Mapping[str, Any]) -> set[tuple[str, int, str, int]]:
    workload = _as_mapping(manifest["workload"], "manifest.workload")
    return {
        (cast(str, mode), cast(int, batched), cast(str, prefix), cast(int, concurrency))
        for mode in cast(list[Any], workload["execution_modes"])
        for batched in cast(list[Any], workload["max_num_batched_tokens"])
        for prefix in cast(list[Any], workload["prefix_variants"])
        for concurrency in cast(list[Any], workload["concurrency"])
    }


def _verify_case(
    case_value: Any, manifest: Mapping[str, Any], location: str
) -> tuple[str, int, str, int]:
    case = _as_mapping(case_value, location)
    _exact(case, _CASE_FIELDS, location)
    mode = case.get("execution_mode")
    batched = case.get("max_num_batched_tokens")
    prefix = case.get("prefix_variant")
    concurrency = case.get("concurrency")
    if not isinstance(mode, str) or not isinstance(prefix, str):
        raise ValueError(f"{location} case dimensions are invalid")
    batched_int = _positive_int(batched, f"{location}.max_num_batched_tokens")
    concurrency_int = _positive_int(concurrency, f"{location}.concurrency")
    key = (mode, batched_int, prefix, concurrency_int)
    if key not in _expected_case_keys(manifest):
        raise ValueError(f"{location} case matrix drift")
    if case.get("case_id") != _case_id(*key):
        raise ValueError(f"{location}.case_id drift")
    engine = _as_mapping(case.get("engine"), f"{location}.engine")
    engine_fields = frozenset({*_LOCKED_ENGINE, "enforce_eager", "num_kvcache_blocks"})
    _exact(engine, engine_fields, f"{location}.engine")
    for name, value in _LOCKED_ENGINE.items():
        if engine.get(name) != value:
            raise ValueError(f"{location}.engine.{name} drift")
    if engine["enforce_eager"] is not (mode == "eager"):
        raise ValueError(f"{location}.engine enforce_eager drift")
    status = case.get("status")
    if status not in {"success", "failed"}:
        raise ValueError(f"{location}.status is invalid")
    samples = _as_list(case.get("samples"), f"{location}.samples")
    if status == "failed" and not samples:
        failure = _as_mapping(case.get("failure"), f"{location}.failure")
        _exact(failure, frozenset({"stage", "error_type"}), f"{location}.failure")
        if failure.get("stage") != "engine_init" or engine["num_kvcache_blocks"] is not None:
            raise ValueError(f"{location} engine-init failure drift")
        if case.get("summary") is not None:
            raise ValueError(f"{location} failed case cannot have a summary")
        return key
    _positive_int(engine["num_kvcache_blocks"], f"{location}.engine.num_kvcache_blocks")
    collection = _as_mapping(manifest["collection"], "manifest.collection")
    expected_count = cast(int, collection["warmup_runs"]) + cast(
        int, collection["measurement_runs"]
    )
    if len(samples) != expected_count:
        raise ValueError(f"{location}.samples count drift")
    block_size = cast(int, _as_mapping(manifest["engine"], "manifest.engine")["kvcache_block_size"])
    expected_hit_blocks = cast(
        int, _as_mapping(manifest["workload"], "workload")["prefix_cached_blocks"]
    ) - (1 if prefix == "one_token_drift" else 0)
    for index, sample in enumerate(samples):
        kind = "warmup" if index == 0 else "measurement"
        sample_index = 0 if index == 0 else index - 1
        _verify_sample(
            sample,
            location=f"{location}.samples[{index}]",
            expected_kind=kind,
            expected_index=sample_index,
            concurrency=concurrency_int,
            prefix_variant=prefix,
            execution_mode=mode,
            block_size=block_size,
            expected_hit_blocks=expected_hit_blocks,
            max_num_batched_tokens=batched_int,
        )
    all_success = all(
        cast(Mapping[str, Any], sample).get("status") == "success" for sample in samples
    )
    if status == "success":
        if not all_success or case.get("failure") is not None:
            raise ValueError(f"{location} successful case terminal drift")
        measurement_samples = [cast(Mapping[str, Any], sample) for sample in samples[1:]]
        _verify_summary(case, measurement_samples)
    else:
        failure = _as_mapping(case.get("failure"), f"{location}.failure")
        if all_success or case.get("summary") is not None:
            raise ValueError(f"{location} failed case terminal drift")
        first_failure = next(
            cast(Mapping[str, Any], sample)["failure"]
            for sample in samples
            if cast(Mapping[str, Any], sample).get("status") == "failed"
        )
        if failure != first_failure:
            raise ValueError(f"{location}.failure does not bind first failed sample")
    return key


def verify_study_report(manifest: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Verify report shape, identities, timing arithmetic, and KV invariants offline."""

    _exact(report, _REPORT_FIELDS, "report")
    supplied = report.get("report_fingerprint")
    if not isinstance(supplied, str) or not hmac.compare_digest(
        supplied, fingerprint_document(report, field="report_fingerprint")
    ):
        raise ValueError("report fingerprint mismatch")
    if (
        report.get("report_version") != REPORT_VERSION
        or report.get("study_id") != manifest.get("study_id")
        or report.get("manifest_fingerprint") != manifest.get("manifest_fingerprint")
        or report.get("evidence_boundary") != NANO_VLLM_STUDY_EVIDENCE_BOUNDARY
    ):
        raise ValueError("report identity or evidence boundary drift")
    collected_at = report.get("collected_at")
    if not isinstance(collected_at, str):
        raise ValueError("report.collected_at is invalid")
    try:
        parsed = datetime.fromisoformat(collected_at)
    except ValueError as error:
        raise ValueError("report.collected_at is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("report.collected_at must be UTC")
    source = _as_mapping(report.get("source"), "report.source")
    _exact(
        source,
        frozenset(
            {
                "repository",
                "revision",
                "package_version",
                "clean_checkout",
                "origin_verified",
                "source_path_published",
            }
        ),
        "report.source",
    )
    if source != {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "package_version": "0.2.0",
        "clean_checkout": True,
        "origin_verified": True,
        "source_path_published": False,
    }:
        raise ValueError("report source revision or cleanliness drift")
    model = _as_mapping(report.get("model"), "report.model")
    _exact(
        model,
        frozenset(
            {
                *_LOCKED_MODEL_IDENTITY,
                "artifacts",
                "selected_file_count",
                "selected_total_bytes",
                "snapshot_path_published",
                "raw_prompt_published",
                "tokenizer_class",
                "tokenizer_vocab_size",
            }
        ),
        "report.model",
    )
    for name, value in _LOCKED_MODEL_IDENTITY.items():
        if model.get(name) != value:
            raise ValueError(f"report.model.{name} drift")
    expected_artifacts = [{**copy.deepcopy(item), "verified": True} for item in _LOCKED_ARTIFACTS]
    if (
        model.get("artifacts") != expected_artifacts
        or model.get("selected_file_count") != len(expected_artifacts)
        or model.get("selected_total_bytes")
        != sum(cast(int, item["size_bytes"]) for item in expected_artifacts)
        or model.get("snapshot_path_published") is not False
        or model.get("raw_prompt_published") is not False
        or not isinstance(model.get("tokenizer_class"), str)
    ):
        raise ValueError("report model artifacts or privacy boundary drift")
    _positive_int(model.get("tokenizer_vocab_size"), "report.model.tokenizer_vocab_size")
    runtime = _as_mapping(report.get("runtime"), "report.runtime")
    runtime_fields = frozenset(
        {
            "python",
            "python_implementation",
            "platform",
            "torch",
            "cuda_runtime",
            "nccl",
            "transformers",
            "flash_attn",
            "triton",
            "xxhash",
            "nano_vllm",
            "cuda_available",
        }
    )
    _exact(runtime, runtime_fields, "report.runtime")
    for name in runtime_fields - {"cuda_available"}:
        if not isinstance(runtime[name], str) or not runtime[name]:
            raise ValueError(f"report.runtime.{name} is invalid")
    if runtime["cuda_available"] is not True or runtime["nano_vllm"] != "0.2.0":
        raise ValueError("report runtime CUDA or nano-vLLM identity drift")
    hardware = _as_mapping(report.get("hardware"), "report.hardware")
    _exact(
        hardware,
        frozenset(
            {
                "device_index",
                "device_name",
                "compute_capability",
                "total_memory_bytes",
                "driver_version",
                "device_count_visible",
            }
        ),
        "report.hardware",
    )
    if hardware["device_index"] != 0 or not isinstance(hardware["device_name"], str):
        raise ValueError("report hardware device identity drift")
    capability = _as_list(hardware["compute_capability"], "report.hardware.compute_capability")
    if len(capability) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in capability
    ):
        raise ValueError("report hardware compute capability is invalid")
    _positive_int(hardware["total_memory_bytes"], "report.hardware.total_memory_bytes")
    _positive_int(hardware["device_count_visible"], "report.hardware.device_count_visible")
    if not isinstance(hardware["driver_version"], str) or not hardware["driver_version"]:
        raise ValueError("report hardware driver identity is invalid")
    collection = _as_mapping(report.get("collection"), "report.collection")
    _exact(
        collection,
        frozenset({*_LOCKED_COLLECTION, "worker_count", "successful_cases", "failed_cases"}),
        "report.collection",
    )
    for name, value in _LOCKED_COLLECTION.items():
        if collection.get(name) != value:
            raise ValueError(f"report.collection.{name} drift")
    if collection.get("worker_count") != 4:
        raise ValueError("report.collection worker isolation drift")
    cases = _as_list(report.get("cases"), "report.cases")
    expected_keys = _expected_case_keys(manifest)
    observed_keys: set[tuple[str, int, str, int]] = set()
    success_count = 0
    for index, case in enumerate(cases):
        key = _verify_case(case, manifest, f"report.cases[{index}]")
        if key in observed_keys:
            raise ValueError("report contains a duplicate study case")
        observed_keys.add(key)
        if cast(Mapping[str, Any], case).get("status") == "success":
            success_count += 1
    if observed_keys != expected_keys:
        raise ValueError("report case matrix is incomplete")
    if (
        collection.get("successful_cases") != success_count
        or collection.get("failed_cases") != len(cases) - success_count
    ):
        raise ValueError("report collection terminal counts drift")
    return copy.deepcopy(dict(report))


def load_and_verify_study_report(manifest_path: Path, report_path: Path) -> dict[str, Any]:
    manifest = load_study_manifest(manifest_path)
    report = _load_strict_object(
        report_path, maximum_bytes=MAX_REPORT_BYTES, label="nano-vLLM study report"
    )
    return verify_study_report(manifest, report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--manifest", required=True, type=Path)
    collect_parser.add_argument("--source-root", required=True, type=Path)
    collect_parser.add_argument("--model-snapshot", required=True, type=Path)
    collect_parser.add_argument("--output", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", required=True, type=Path)
    verify_parser.add_argument("--report", required=True, type=Path)
    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--manifest", required=True, type=Path)
    worker_parser.add_argument("--source-root", required=True, type=Path)
    worker_parser.add_argument("--model-snapshot", required=True, type=Path)
    worker_parser.add_argument("--output", required=True, type=Path)
    worker_parser.add_argument("--mode", required=True, choices=("eager", "cuda_graph"))
    worker_parser.add_argument(
        "--max-num-batched-tokens", required=True, type=int, choices=(256, 1024)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "collect":
        report = collect_study(args.manifest, args.source_root, args.model_snapshot, args.output)
    elif args.command == "verify":
        report = load_and_verify_study_report(args.manifest, args.report)
    else:
        return _run_worker(
            args.manifest,
            args.source_root,
            args.model_snapshot,
            args.output,
            mode=args.mode,
            batched_tokens=args.max_num_batched_tokens,
        )
    summary = {
        "report_version": report["report_version"],
        "report_fingerprint": report["report_fingerprint"],
        "successful_cases": report["collection"]["successful_cases"],
        "failed_cases": report["collection"]["failed_cases"],
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
