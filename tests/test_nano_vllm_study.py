from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from about_llm.inference.nano_vllm_study import (
    NANO_VLLM_STUDY_EVIDENCE_BOUNDARY,
    REPORT_VERSION,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    fingerprint_document,
    load_and_verify_study_report,
    load_study_manifest,
    main,
    verify_study_report,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "inference-serving"
MANIFEST = PROJECT / "nano-vllm-qwen3-0.6b.study.json"
pytestmark = pytest.mark.contract


def _rehash(document: dict[str, Any], field: str) -> None:
    document.pop(field, None)
    document[field] = fingerprint_document(document, field=field)


def _kv(*, used: int = 0, references: int = 0, cached: int = 3) -> dict[str, int]:
    total = 100
    return {
        "total_blocks": total,
        "free_blocks": total - used,
        "used_blocks": used,
        "ref_count_total": references,
        "block_table_references": references,
        "shared_blocks": 0,
        "max_ref_count": 1 if references else 0,
        "cached_hash_entries": cached,
    }


def _sequence(
    *,
    completion_tokens: int,
    cached_tokens: int,
    scheduled_tokens: int,
    block_count: int,
    is_prefill: bool,
    status: str = "running",
) -> dict[str, Any]:
    return {
        "request_id": "req-0",
        "sequence_id": 1,
        "status": status,
        "prompt_tokens": 768,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "scheduled_tokens": scheduled_tokens,
        "block_count": block_count,
        "is_prefill": is_prefill,
    }


def _trace() -> list[dict[str, Any]]:
    released = _kv()
    trace: list[dict[str, Any]] = []
    before = _sequence(
        completion_tokens=0,
        cached_tokens=0,
        scheduled_tokens=0,
        block_count=0,
        is_prefill=True,
        status="waiting",
    )
    scheduled = _sequence(
        completion_tokens=0,
        cached_tokens=512,
        scheduled_tokens=256,
        block_count=3,
        is_prefill=True,
    )
    after = _sequence(
        completion_tokens=1,
        cached_tokens=768,
        scheduled_tokens=0,
        block_count=3,
        is_prefill=True,
    )
    trace.append(
        {
            "step_index": 0,
            "phase": "prefill",
            "execution_path": "eager",
            "started_ns": 100,
            "finished_ns": 1_000_000,
            "scheduled_tokens": 256,
            "scheduled_sequence_ids": [1],
            "sequences_before": [before],
            "sequences_scheduled": [scheduled],
            "sequences_after": [after],
            "kv_before": released,
            "kv_scheduled": _kv(used=3, references=3),
            "kv_after": _kv(used=3, references=3),
            "committed_token_count": 1,
        }
    )
    for decode_index in range(1, 8):
        completion_before = decode_index
        before_blocks = 3 if decode_index == 1 else 4
        before_kv = _kv(used=before_blocks, references=before_blocks)
        scheduled_decode = _sequence(
            completion_tokens=completion_before,
            cached_tokens=768 + decode_index - 1,
            scheduled_tokens=1,
            block_count=4,
            is_prefill=False,
        )
        is_final = decode_index == 7
        after_decode = []
        after_kv = released
        if not is_final:
            after_decode = [
                _sequence(
                    completion_tokens=completion_before + 1,
                    cached_tokens=768 + decode_index,
                    scheduled_tokens=0,
                    block_count=4,
                    is_prefill=False,
                )
            ]
            after_kv = _kv(used=4, references=4)
        trace.append(
            {
                "step_index": decode_index,
                "phase": "decode",
                "execution_path": "eager",
                "started_ns": decode_index * 1_000_000 + 100,
                "finished_ns": (decode_index + 1) * 1_000_000,
                "scheduled_tokens": 1,
                "scheduled_sequence_ids": [1],
                "sequences_before": [
                    _sequence(
                        completion_tokens=completion_before,
                        cached_tokens=768 + decode_index - 1,
                        scheduled_tokens=0,
                        block_count=before_blocks,
                        is_prefill=decode_index != 1,
                    )
                ],
                "sequences_scheduled": [scheduled_decode],
                "sequences_after": after_decode,
                "kv_before": before_kv,
                "kv_scheduled": _kv(used=4, references=4),
                "kv_after": after_kv,
                "committed_token_count": 1,
            }
        )
    return trace


def _sample(kind: str, index: int, seed: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "sample_index": index,
        "seed": seed,
        "status": "success",
        "failure": None,
        "primer": {
            "status": "success",
            "cached_block_entries": 3,
            "kv_after": _kv(),
        },
        "engine_finished_ns": 10_000_000,
        "requests": [
            {
                "request_id": "req-0",
                "sequence_id": 1,
                "added_ns": 0,
                "first_token_ns": 1_000_000,
                "finished_ns": 8_000_000,
                "completion_tokens": 8,
            }
        ],
        "prefix_hits": [{"request_id": "req-0", "sequence_id": 1, "hit_blocks": 2}],
        "trace": _trace(),
        "metrics": {
            "duration_ms": 10.0,
            "total_output_tokens": 8,
            "ttft_ms": 1.0,
            "tpot_ms": 1.0,
            "e2e_ms": 8.0,
            "output_tokens_per_second": 800.0,
            "peak_allocated_bytes": 1_000,
            "peak_reserved_bytes": 2_000,
        },
        "kv_released": _kv(),
    }


def _successful_case(manifest: Mapping[str, Any]) -> dict[str, Any]:
    samples = [_sample("warmup", 0, 1)] + [
        _sample("measurement", index, index + 2) for index in range(5)
    ]
    metric_names = (
        "ttft_ms",
        "tpot_ms",
        "e2e_ms",
        "output_tokens_per_second",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    measurements = samples[1:]
    raw = {name: [sample["metrics"][name] for sample in measurements] for name in metric_names}
    return {
        "case_id": "eager-mbt256-exact-c1",
        "execution_mode": "eager",
        "max_num_batched_tokens": 256,
        "prefix_variant": "exact",
        "concurrency": 1,
        "status": "success",
        "failure": None,
        "engine": {
            **copy.deepcopy(manifest["engine"]),
            "enforce_eager": True,
            "num_kvcache_blocks": 100,
        },
        "samples": samples,
        "summary": {
            "measurement_count": 5,
            "raw": raw,
            "median": {name: values[2] for name, values in raw.items()},
        },
    }


def _failed_case(
    manifest: Mapping[str, Any], mode: str, batched: int, prefix: str, concurrency: int
) -> dict[str, Any]:
    return {
        "case_id": f"{mode}-mbt{batched}-{prefix}-c{concurrency}",
        "execution_mode": mode,
        "max_num_batched_tokens": batched,
        "prefix_variant": prefix,
        "concurrency": concurrency,
        "status": "failed",
        "failure": {"stage": "engine_init", "error_type": "OutOfMemoryError"},
        "engine": {
            **copy.deepcopy(manifest["engine"]),
            "enforce_eager": mode == "eager",
            "num_kvcache_blocks": None,
        },
        "samples": [],
        "summary": None,
    }


def _report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for mode in manifest["workload"]["execution_modes"]:
        for batched in manifest["workload"]["max_num_batched_tokens"]:
            for prefix in manifest["workload"]["prefix_variants"]:
                for concurrency in manifest["workload"]["concurrency"]:
                    if (mode, batched, prefix, concurrency) == (
                        "eager",
                        256,
                        "exact",
                        1,
                    ):
                        cases.append(_successful_case(manifest))
                    else:
                        cases.append(_failed_case(manifest, mode, batched, prefix, concurrency))
    artifacts = [
        {**copy.deepcopy(item), "verified": True} for item in manifest["model"]["artifacts"]
    ]
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "study_id": manifest["study_id"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "collected_at": "2026-08-20T00:00:00+00:00",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "package_version": "0.2.0",
            "clean_checkout": True,
            "origin_verified": True,
            "source_path_published": False,
        },
        "model": {
            **{
                key: copy.deepcopy(value)
                for key, value in manifest["model"].items()
                if key != "artifacts"
            },
            "artifacts": artifacts,
            "selected_file_count": len(artifacts),
            "selected_total_bytes": sum(item["size_bytes"] for item in artifacts),
            "snapshot_path_published": False,
            "raw_prompt_published": False,
            "tokenizer_class": "Qwen2TokenizerFast",
            "tokenizer_vocab_size": 151669,
        },
        "runtime": {
            "python": "3.12.10",
            "python_implementation": "CPython",
            "platform": "Linux-WSL2",
            "torch": "2.7.1+cu128",
            "cuda_runtime": "12.8",
            "nccl": "2.26.2",
            "transformers": "4.53.0",
            "flash_attn": "2.8.0",
            "triton": "3.3.1",
            "xxhash": "3.5.0",
            "nano_vllm": "0.2.0",
            "cuda_available": True,
        },
        "hardware": {
            "device_index": 0,
            "device_name": "Synthetic RTX fixture",
            "compute_capability": [8, 6],
            "total_memory_bytes": 8_000_000_000,
            "driver_version": "fixture-driver",
            "device_count_visible": 1,
        },
        "collection": {
            **copy.deepcopy(manifest["collection"]),
            "worker_count": 4,
            "successful_cases": 1,
            "failed_cases": len(cases) - 1,
        },
        "cases": cases,
        "evidence_boundary": NANO_VLLM_STUDY_EVIDENCE_BOUNDARY,
    }
    _rehash(report, "report_fingerprint")
    return report


def test_reviewed_manifest_pins_source_model_and_matrix() -> None:
    manifest = load_study_manifest(MANIFEST)

    assert manifest["source"]["revision"] == SOURCE_REVISION
    assert manifest["model"]["revision"] == ("c1899de289a04d12100db370d81485cdf75e47ca")
    assert manifest["engine"]["kvcache_block_size"] == 256
    assert manifest["workload"]["max_num_batched_tokens"] == [256, 1024]
    assert manifest["workload"]["concurrency"] == [1, 2, 4, 8]
    assert manifest["collection"] == {
        "warmup_runs": 1,
        "measurement_runs": 5,
        "subprocess_isolation": True,
        "publish_raw_prompts": False,
    }


def test_manifest_rejects_duplicate_nonfinite_and_revision_drift(tmp_path: Path) -> None:
    invalid_payloads = (
        b'{"manifest_version":"x","manifest_version":"y"}',
        b'{"value":NaN}',
        b"{\xff}",
    )
    for index, payload in enumerate(invalid_payloads):
        path = tmp_path / f"invalid-{index}.json"
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="strict UTF-8 JSON"):
            load_study_manifest(path)

    drifted = json.loads(MANIFEST.read_text(encoding="utf-8"))
    drifted["source"]["revision"] = "0" * 40
    _rehash(drifted, "manifest_fingerprint")
    path = tmp_path / "revision-drift.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="identity or workload drift"):
        load_study_manifest(path)


def test_synthetic_report_proves_success_and_preserves_failure_terminals() -> None:
    manifest = load_study_manifest(MANIFEST)
    report = _report(manifest)

    verified = verify_study_report(manifest, report)

    assert verified["collection"]["successful_cases"] == 1
    assert verified["collection"]["failed_cases"] == 31
    assert verified["cases"][0]["summary"]["median"]["tpot_ms"] == 1.0
    assert verified["model"]["raw_prompt_published"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["source"].update({"revision": "0" * 40}),
            "source revision",
        ),
        (
            lambda report: report["cases"][0]["samples"][1]["requests"][0].update(
                {"first_token_ns": 9_000_000}
            ),
            "timing order",
        ),
        (
            lambda report: report["cases"][0]["samples"][1]["metrics"].update(
                {"output_tokens_per_second": 1.0}
            ),
            "arithmetic drift",
        ),
        (
            lambda report: report["cases"][0]["samples"][1]["kv_released"].update(
                {"free_blocks": 99}
            ),
            "KV ledger invariant",
        ),
        (
            lambda report: report["cases"][0]["samples"][1]["trace"][0].update(
                {"scheduled_tokens": 257}
            ),
            "token budget exceeded",
        ),
    ],
)
def test_verifier_rejects_cooperatively_rehashed_semantic_drift(mutate: Any, message: str) -> None:
    manifest = load_study_manifest(MANIFEST)
    report = _report(manifest)
    mutate(report)
    _rehash(report, "report_fingerprint")

    with pytest.raises(ValueError, match=message):
        verify_study_report(manifest, report)


def test_report_loader_rejects_duplicate_nonfinite_and_invalid_utf8(tmp_path: Path) -> None:
    invalid_payloads = (
        b'{"report_version":"x","report_version":"y"}',
        b'{"value":Infinity}',
        b"{\xff}",
    )
    for index, payload in enumerate(invalid_payloads):
        path = tmp_path / f"invalid-report-{index}.json"
        path.write_bytes(payload)
        with pytest.raises(ValueError, match="strict UTF-8 JSON"):
            load_and_verify_study_report(MANIFEST, path)


@pytest.mark.smoke
def test_verify_cli_is_cpu_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = load_study_manifest(MANIFEST)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report(manifest), ensure_ascii=False), encoding="utf-8")

    assert main(["verify", "--manifest", str(MANIFEST), "--report", str(report_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["successful_cases"] == 1
    assert summary["failed_cases"] == 31
