"""演示 DataLoader 预取为何会让“已发出位置”不同于“已完成位置”。

实验在独立 CPU 进程中先消费 3 个样本并保存 checkpoint。两个恢复对照分别从训练主循环
真正完成的 committed cursor，以及 sampler 已提前送进 worker 队列的 emitted cursor 继续。
前者不会漏样本，后者会跳过只被预取但尚未训练的样本。实验还比较 worker 局部随机数与
按 sample ID 派生的无状态随机数；它没有序列化 DataLoader 内部队列，也不能保证任意随机
数据增强都能精确恢复。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

CONTROL_VERSION = "about-llm.dataloader-prefetch-resume-control.v1"
CHECKPOINT_VERSION = "about-llm.dataloader-prefetch-checkpoint.v1"
DATASET_SIZE = 10
PERMUTATION = (8, 3, 1, 7, 0, 9, 4, 2, 6, 5)
SPLIT_COMMITTED_CURSOR = 3
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
BATCH_SIZE = 1
LOADER_GENERATOR_SEED = 20260814
SAMPLE_KEY_NAMESPACE = "about-llm.sample-keyed-randomness.v1"
MAX_CHECKPOINT_BYTES = 64 * 1024
EXPECTED_EMITTED_CURSOR = (
    SPLIT_COMMITTED_CURSOR + NUM_WORKERS * PREFETCH_FACTOR
)
WorkerKind = Literal[
    "uninterrupted",
    "phase1",
    "resume_committed",
    "resume_emitted",
]


def _dataset_identity() -> str:
    payload = {
        "dataset_size": DATASET_SIZE,
        "sample_ids": list(range(DATASET_SIZE)),
        "sample_key_namespace": SAMPLE_KEY_NAMESPACE,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class PrefetchDataset(Dataset[dict[str, int | float]]):
    """同时返回依赖 worker 状态和只依赖 sample ID 的两种随机观测。"""

    def __len__(self) -> int:
        return DATASET_SIZE

    def __getitem__(self, sample_id: int) -> dict[str, int | float]:
        if isinstance(sample_id, bool) or not isinstance(sample_id, int):
            raise TypeError("sample_id must be an integer")
        if not 0 <= sample_id < DATASET_SIZE:
            raise IndexError("sample_id is outside the authored dataset")
        worker = get_worker_info()
        if worker is None:
            raise RuntimeError("this fixture requires a real DataLoader worker")

        # 这个值取决于样本落到哪个 worker 以及该 worker 已消费多少次 RNG。
        worker_rng_value = float(torch.rand((), dtype=torch.float64).item())
        # sample-keyed 随机数只由样本身份派生，跨进程恢复后仍可重建同一个值。
        digest = hashlib.sha256(
            f"{SAMPLE_KEY_NAMESPACE}:{sample_id}".encode()
        ).digest()
        sample_seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_seed)
        sample_keyed_value = float(
            torch.rand((), dtype=torch.float64, generator=generator).item()
        )
        return {
            "sample_id": sample_id,
            "worker_id": worker.id,
            "worker_rng_value": worker_rng_value,
            "sample_keyed_value": sample_keyed_value,
        }


class TrackingOffsetSampler(Sampler[int]):
    """记录 DataLoader 已从固定顺序中请求到哪里，而非训练完成到哪里。"""

    def __init__(self, permutation: tuple[int, ...], start_cursor: int) -> None:
        if tuple(sorted(permutation)) != tuple(range(DATASET_SIZE)):
            raise ValueError("permutation must cover each authored sample exactly once")
        if isinstance(start_cursor, bool) or not isinstance(start_cursor, int):
            raise TypeError("start_cursor must be an integer")
        if not 0 <= start_cursor <= len(permutation):
            raise ValueError("start_cursor is outside the permutation")
        self.permutation = permutation
        self.start_cursor = start_cursor
        self.emitted_cursor = start_cursor

    def __iter__(self) -> Iterator[int]:
        while self.emitted_cursor < len(self.permutation):
            sample_id = self.permutation[self.emitted_cursor]
            self.emitted_cursor += 1
            yield sample_id

    def __len__(self) -> int:
        return len(self.permutation) - self.start_cursor


def _int_from_batch(batch: Mapping[str, Any], field: str) -> int:
    value = batch.get(field)
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise AssertionError(f"{field} must collate to one scalar tensor")
    return int(value.item())


def _float_from_batch(batch: Mapping[str, Any], field: str) -> float:
    value = batch.get(field)
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise AssertionError(f"{field} must collate to one scalar tensor")
    result = float(value.item())
    if not math.isfinite(result):
        raise AssertionError(f"{field} must be finite")
    return result


def _run_loader(
    *,
    worker_kind: WorkerKind,
    start_cursor: int,
    max_records: int | None,
) -> dict[str, object]:
    """运行一个 loader 阶段并分别记录返回样本与 sampler 发出位置。"""

    sampler = TrackingOffsetSampler(PERMUTATION, start_cursor)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(LOADER_GENERATOR_SEED)
    loader = DataLoader(
        PrefetchDataset(),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=False,
        pin_memory=False,
        timeout=30,
        multiprocessing_context="spawn",
        generator=loader_generator,
        in_order=True,
    )
    iterator = iter(loader)
    records: list[dict[str, int | float]] = []
    try:
        while max_records is None or len(records) < max_records:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if not isinstance(batch, Mapping):
                raise AssertionError("default collate must return a mapping")
            records.append(
                {
                    "sample_id": _int_from_batch(batch, "sample_id"),
                    "worker_id": _int_from_batch(batch, "worker_id"),
                    "worker_rng_value": _float_from_batch(
                        batch, "worker_rng_value"
                    ),
                    "sample_keyed_value": _float_from_batch(
                        batch, "sample_keyed_value"
                    ),
                }
            )
        # 即使主循环只拿到少量 batch，worker 预取也可能已让 sampler 向前移动多步。
        emitted_cursor = sampler.emitted_cursor
    finally:
        # CPython destroys the non-persistent iterator here and joins workers.
        # The control does not read or mutate private DataLoader queue fields.
        del iterator
        del loader
        gc.collect()

    return {
        "worker_kind": worker_kind,
        "pid": os.getpid(),
        "start_cursor": start_cursor,
        "requested_record_limit": max_records,
        "sampler_emitted_cursor_when_observed": emitted_cursor,
        "records": records,
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _encode_canonical(payload: object, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _parse_strict_json(raw: str) -> Any:
    return json.loads(
        raw,
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_pairs,
    )


def _checkpoint_payload(phase_report: dict[str, object]) -> dict[str, object]:
    """只保存可公开验证的恢复契约，不尝试保存私有队列状态。"""

    records = phase_report.get("records")
    if not isinstance(records, list):
        raise AssertionError("phase records must be a list")
    emitted_cursor = phase_report.get("sampler_emitted_cursor_when_observed")
    if isinstance(emitted_cursor, bool) or not isinstance(emitted_cursor, int):
        raise AssertionError("emitted cursor must be an integer")
    phase_pid = phase_report.get("pid")
    if isinstance(phase_pid, bool) or not isinstance(phase_pid, int):
        raise AssertionError("phase pid must be an integer")
    return {
        "implementation": CHECKPOINT_VERSION,
        "dataset_identity": _dataset_identity(),
        "permutation": list(PERMUTATION),
        "committed_cursor": SPLIT_COMMITTED_CURSOR,
        "sampler_emitted_cursor": emitted_cursor,
        "consumed_sample_ids": [int(record["sample_id"]) for record in records],
        "phase_pid": phase_pid,
        "loader_contract": {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "persistent_workers": False,
            "pin_memory": False,
            "multiprocessing_context": "spawn",
            "in_order": True,
            "loader_generator_seed": LOADER_GENERATOR_SEED,
        },
    }


def _write_checkpoint(path: Path, payload: dict[str, object]) -> int:
    encoded = _encode_canonical(payload).encode("utf-8")
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise AssertionError("checkpoint exceeds the authored resource cap")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.write_bytes(encoded)
    return len(encoded)


def _load_checkpoint(path: Path) -> tuple[dict[str, object], int]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint size is outside the authored resource cap")
    raw = path.read_text(encoding="utf-8")
    payload = _parse_strict_json(raw)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a JSON object")
    if raw != _encode_canonical(payload):
        raise ValueError("checkpoint must use the canonical JSON encoding")
    expected_fields = {
        "implementation",
        "dataset_identity",
        "permutation",
        "committed_cursor",
        "sampler_emitted_cursor",
        "consumed_sample_ids",
        "phase_pid",
        "loader_contract",
    }
    if set(payload) != expected_fields:
        raise ValueError("checkpoint fields do not match the closed schema")
    if payload["implementation"] != CHECKPOINT_VERSION:
        raise ValueError("checkpoint implementation drifted")
    if payload["dataset_identity"] != _dataset_identity():
        raise ValueError("checkpoint dataset identity drifted")
    if payload["permutation"] != list(PERMUTATION):
        raise ValueError("checkpoint permutation drifted")
    if payload["committed_cursor"] != SPLIT_COMMITTED_CURSOR:
        raise ValueError("checkpoint committed cursor drifted")
    emitted = payload["sampler_emitted_cursor"]
    if isinstance(emitted, bool) or not isinstance(emitted, int):
        raise ValueError("checkpoint emitted cursor must be an integer")
    if not SPLIT_COMMITTED_CURSOR < emitted <= DATASET_SIZE:
        raise ValueError("checkpoint must exhibit sampler prefetch-ahead")
    if payload["consumed_sample_ids"] != list(
        PERMUTATION[:SPLIT_COMMITTED_CURSOR]
    ):
        raise ValueError("checkpoint consumed sample IDs drifted")
    expected_contract = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "persistent_workers": False,
        "pin_memory": False,
        "multiprocessing_context": "spawn",
        "in_order": True,
        "loader_generator_seed": LOADER_GENERATOR_SEED,
    }
    if payload["loader_contract"] != expected_contract:
        raise ValueError("checkpoint loader contract drifted")
    phase_pid = payload["phase_pid"]
    if isinstance(phase_pid, bool) or not isinstance(phase_pid, int) or phase_pid <= 0:
        raise ValueError("checkpoint phase pid is invalid")
    return payload, size


def _worker_main(worker_kind: WorkerKind, checkpoint_path: Path) -> None:
    """在独立进程中运行不间断、首阶段或某一种恢复路径。"""

    if worker_kind == "uninterrupted":
        report = _run_loader(
            worker_kind=worker_kind,
            start_cursor=0,
            max_records=None,
        )
    elif worker_kind == "phase1":
        report = _run_loader(
            worker_kind=worker_kind,
            start_cursor=0,
            max_records=SPLIT_COMMITTED_CURSOR,
        )
        payload = _checkpoint_payload(report)
        report["checkpoint_size_bytes"] = _write_checkpoint(
            checkpoint_path, payload
        )
    else:
        checkpoint, checkpoint_size = _load_checkpoint(checkpoint_path)
        # 正确路径从主循环已完成的位置恢复；错误对照从 sampler 预取到的位置恢复。
        cursor_field = (
            "committed_cursor"
            if worker_kind == "resume_committed"
            else "sampler_emitted_cursor"
        )
        start_cursor = checkpoint[cursor_field]
        if isinstance(start_cursor, bool) or not isinstance(start_cursor, int):
            raise AssertionError("validated checkpoint cursor must be an integer")
        report = _run_loader(
            worker_kind=worker_kind,
            start_cursor=start_cursor,
            max_records=None,
        )
        report["checkpoint_size_bytes"] = checkpoint_size
        report["checkpoint_phase_pid"] = checkpoint["phase_pid"]
    sys.stdout.buffer.write(_encode_canonical(report).encode("utf-8"))


def _run_worker_process(
    script_path: Path,
    worker_kind: WorkerKind,
    checkpoint_path: Path,
) -> dict[str, object]:
    """启动新解释器，确保恢复阶段不能偷用上一阶段的进程内状态。"""

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--worker-kind",
            worker_kind,
            "--checkpoint",
            str(checkpoint_path),
        ],
        cwd=script_path.parents[2],
        check=False,
        capture_output=True,
        timeout=180,
    )
    stderr = completed.stderr.decode("utf-8", errors="strict")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{worker_kind} worker failed ({completed.returncode}): {stderr}"
        )
    if stderr:
        raise AssertionError(f"{worker_kind} worker wrote stderr: {stderr}")
    stdout = completed.stdout.decode("utf-8", errors="strict")
    report = _parse_strict_json(stdout)
    if not isinstance(report, dict):
        raise AssertionError(f"{worker_kind} worker report must be an object")
    return report


def _records(report: dict[str, object]) -> list[dict[str, int | float]]:
    raw = report.get("records")
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise AssertionError("worker records must be a list of objects")
    records: list[dict[str, int | float]] = []
    for item in raw:
        if set(item) != {
            "sample_id",
            "worker_id",
            "worker_rng_value",
            "sample_keyed_value",
        }:
            raise AssertionError("worker record fields drifted")
        sample_id = item["sample_id"]
        worker_id = item["worker_id"]
        worker_value = item["worker_rng_value"]
        keyed_value = item["sample_keyed_value"]
        if any(isinstance(value, bool) for value in item.values()):
            raise AssertionError("worker record values cannot be booleans")
        if not isinstance(sample_id, int) or not isinstance(worker_id, int):
            raise AssertionError("worker sample and worker IDs must be integers")
        if not isinstance(worker_value, (int, float)) or not isinstance(
            keyed_value, (int, float)
        ):
            raise AssertionError("worker random observations must be numeric")
        floats = (float(worker_value), float(keyed_value))
        if not all(math.isfinite(value) for value in floats):
            raise AssertionError("worker random observations must be finite")
        records.append(
            {
                "sample_id": sample_id,
                "worker_id": worker_id,
                "worker_rng_value": floats[0],
                "sample_keyed_value": floats[1],
            }
        )
    return records


def _pid(report: dict[str, object]) -> int:
    pid = report.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise AssertionError("worker pid must be a positive integer")
    return pid


def run_control(script_path: Path | None = None) -> dict[str, object]:
    """运行未中断、committed-resume 与 emitted-resume 三个进程。"""

    entry = Path(__file__).resolve() if script_path is None else script_path.resolve()
    with tempfile.TemporaryDirectory(prefix="about-llm-dataloader-prefetch-") as temp:
        checkpoint_path = Path(temp) / "prefetch-checkpoint.json"
        uninterrupted = _run_worker_process(
            entry, "uninterrupted", checkpoint_path
        )
        phase1 = _run_worker_process(entry, "phase1", checkpoint_path)
        resume_committed = _run_worker_process(
            entry, "resume_committed", checkpoint_path
        )
        resume_emitted = _run_worker_process(
            entry, "resume_emitted", checkpoint_path
        )
        checkpoint, checkpoint_size = _load_checkpoint(checkpoint_path)

    full_records = _records(uninterrupted)
    prefix_records = _records(phase1)
    committed_records = _records(resume_committed)
    emitted_records = _records(resume_emitted)
    full_ids = [int(item["sample_id"]) for item in full_records]
    prefix_ids = [int(item["sample_id"]) for item in prefix_records]
    committed_ids = [int(item["sample_id"]) for item in committed_records]
    emitted_ids = [int(item["sample_id"]) for item in emitted_records]
    emitted_cursor = checkpoint["sampler_emitted_cursor"]
    if isinstance(emitted_cursor, bool) or not isinstance(emitted_cursor, int):
        raise AssertionError("validated emitted cursor must be an integer")
    skipped_ids = list(PERMUTATION[SPLIT_COMMITTED_CURSOR:emitted_cursor])

    full_prefix_worker_values = [
        float(item["worker_rng_value"])
        for item in full_records[:SPLIT_COMMITTED_CURSOR]
    ]
    phase_worker_values = [
        float(item["worker_rng_value"]) for item in prefix_records
    ]
    full_tail_worker_values = [
        float(item["worker_rng_value"])
        for item in full_records[SPLIT_COMMITTED_CURSOR:]
    ]
    resumed_worker_values = [
        float(item["worker_rng_value"]) for item in committed_records
    ]
    full_tail_keyed_values = [
        float(item["sample_keyed_value"])
        for item in full_records[SPLIT_COMMITTED_CURSOR:]
    ]
    resumed_keyed_values = [
        float(item["sample_keyed_value"]) for item in committed_records
    ]
    worker_rng_max_abs_difference = max(
        abs(left - right)
        for left, right in zip(
            full_tail_worker_values,
            resumed_worker_values,
            strict=True,
        )
    )
    sample_keyed_max_abs_difference = max(
        abs(left - right)
        for left, right in zip(
            full_tail_keyed_values,
            resumed_keyed_values,
            strict=True,
        )
    )
    process_ids = [
        _pid(uninterrupted),
        _pid(phase1),
        _pid(resume_committed),
        _pid(resume_emitted),
    ]
    assertions = {
        "uninterrupted_ids_equal_fixed_permutation": full_ids == list(PERMUTATION),
        "phase1_consumes_only_committed_prefix": prefix_ids
        == list(PERMUTATION[:SPLIT_COMMITTED_CURSOR]),
        "sampler_emitted_cursor_leads_committed_cursor_by_prefetch_window": (
            emitted_cursor == EXPECTED_EMITTED_CURSOR
        ),
        "prefetched_but_uncommitted_ids_are_explicit": skipped_ids
        == list(PERMUTATION[3:7]),
        "committed_cursor_resume_restores_sample_order": (
            prefix_ids + committed_ids == full_ids
        ),
        "emitted_cursor_resume_skips_prefetched_uncommitted_samples": (
            prefix_ids + emitted_ids != full_ids
            and emitted_ids == list(PERMUTATION[emitted_cursor:])
        ),
        "same_seed_independent_phase_replays_worker_rng_prefix": (
            phase_worker_values == full_prefix_worker_values
        ),
        "fresh_workers_do_not_replay_uninterrupted_worker_rng_tail": (
            resumed_worker_values != full_tail_worker_values
            and worker_rng_max_abs_difference > 0.0
        ),
        "sample_keyed_randomness_replays_tail_bit_exact": (
            resumed_keyed_values == full_tail_keyed_values
            and sample_keyed_max_abs_difference == 0.0
        ),
        "all_control_segments_use_distinct_operating_system_processes": (
            len(set(process_ids)) == len(process_ids)
        ),
        "resume_workers_loaded_checkpoint_from_phase1_pid": (
            resume_committed["checkpoint_phase_pid"] == _pid(phase1)
            and resume_emitted["checkpoint_phase_pid"] == _pid(phase1)
        ),
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise AssertionError(f"DataLoader prefetch resume control failed: {failed}")

    return {
        "implementation": CONTROL_VERSION,
        "runtime": {
            "torch_version": torch.__version__,
            "device": "cpu",
            "dataset_kind": "torch.utils.data.Dataset-map-style",
            "dataloader": "torch.utils.data.DataLoader",
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "persistent_workers": False,
            "pin_memory": False,
            "multiprocessing_context": "spawn",
            "in_order": True,
            "loader_generator_seed": LOADER_GENERATOR_SEED,
        },
        "fixture": {
            "dataset_size": DATASET_SIZE,
            "dataset_identity": _dataset_identity(),
            "permutation": list(PERMUTATION),
            "split_committed_cursor": SPLIT_COMMITTED_CURSOR,
            "sampler_emitted_cursor_at_split": emitted_cursor,
            "prefetched_but_uncommitted_sample_ids": skipped_ids,
            "sample_key_namespace": SAMPLE_KEY_NAMESPACE,
        },
        "processes": {
            "uninterrupted_pid": process_ids[0],
            "phase1_pid": process_ids[1],
            "resume_committed_pid": process_ids[2],
            "resume_emitted_pid": process_ids[3],
            "all_distinct": True,
        },
        "checkpoint": {
            "implementation": checkpoint["implementation"],
            "size_bytes": checkpoint_size,
            "fields": sorted(checkpoint),
            "canonical_strict_json": True,
            "preload_size_cap_bytes": MAX_CHECKPOINT_BYTES,
        },
        "paths": {
            "uninterrupted": uninterrupted,
            "phase1": phase1,
            "resume_from_committed_cursor": resume_committed,
            "resume_from_sampler_emitted_cursor_negative_control": resume_emitted,
        },
        "comparisons": {
            "uninterrupted_sample_ids": full_ids,
            "committed_resume_combined_sample_ids": prefix_ids + committed_ids,
            "emitted_resume_combined_sample_ids": prefix_ids + emitted_ids,
            "worker_rng_tail_max_abs_difference": worker_rng_max_abs_difference,
            "sample_keyed_tail_max_abs_difference": sample_keyed_max_abs_difference,
        },
        "assertions": assertions,
        "scope": {
            "real_dataloader_worker_processes_executed": True,
            "num_workers_two_and_prefetch_factor_two_executed": True,
            "sampler_prefetch_ahead_of_committed_consumption_observed": True,
            "cross_pid_checkpoint_and_resume_executed": True,
            "committed_sample_cursor_order_resume_executed": True,
            "sampler_emitted_cursor_skip_negative_control_executed": True,
            "worker_local_torch_rng_nonreplay_observed": True,
            "sample_keyed_stateless_randomness_exact_replay_executed": True,
            "strict_canonical_json_checkpoint_executed": True,
            "private_dataloader_queue_fields_read_or_mutated": False,
            "prefetched_queue_payload_or_worker_process_state_checkpointed": False,
            "worker_local_rng_state_restored": False,
            "arbitrary_stochastic_transform_exact_resume_proved": False,
            "multi_epoch_or_repeated_sample_randomness_policy_executed": False,
            "prefetch_depth_as_public_stable_api_contract_proved": False,
            "persistent_workers_pin_memory_or_iterable_dataset_executed": False,
            "optimizer_scheduler_scaler_or_model_training_executed": False,
            "sample_consumption_and_optimizer_commit_atomicity_proved": False,
            "distributed_sampler_ddp_fsdp_zero_or_sharded_state_executed": False,
            "cuda_gpu_or_target_trainer_dataset_executed": False,
            "checkpoint_crash_power_loss_atomicity_or_authentication_proved": False,
            "throughput_memory_quality_or_convergence_proved": False,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-kind",
        choices=(
            "uninterrupted",
            "phase1",
            "resume_committed",
            "resume_emitted",
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker_kind is None:
        if args.checkpoint is not None:
            raise SystemExit("--checkpoint requires --worker-kind")
        payload = run_control()
        sys.stdout.buffer.write(_encode_canonical(payload, pretty=True).encode("utf-8"))
        return
    if args.checkpoint is None:
        raise SystemExit("--worker-kind requires --checkpoint")
    _worker_main(args.worker_kind, args.checkpoint)


if __name__ == "__main__":
    main()
