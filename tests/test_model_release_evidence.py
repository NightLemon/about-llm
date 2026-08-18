from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from about_llm.model_release_evidence import (
    MODEL_RELEASE_EVIDENCE_BOUNDARY,
    fetch_release_artifact,
    verify_model_release_evidence,
)

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "projects" / "transformers-basics" / "release-evidence"
MANIFEST = EVIDENCE / "manifest.json"
SCRIPT = ROOT / "projects" / "transformers-basics" / "verify_release_evidence.py"


def _copy_evidence(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    target = tmp_path / "release-evidence"
    shutil.copytree(EVIDENCE, target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path, manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _network_free_upstream_fixture(
    manifest_path: Path, manifest: dict[str, Any]
) -> dict[str, bytes]:
    llama_projection = json.loads(
        manifest_path.with_name("llama-3.2-text.projection.json").read_text(
            encoding="utf-8"
        )
    )
    llama_bytes = ("\n".join(llama_projection["source_fragments"]) + "\n").encode()
    llama_record = manifest["records"][0]
    llama_record["upstream_size_bytes"] = len(llama_bytes)
    llama_record["upstream_sha256"] = "sha256:" + hashlib.sha256(llama_bytes).hexdigest()
    _write_manifest(manifest_path, manifest)
    return {
        llama_record["source_url"]: llama_bytes,
        manifest["records"][1]["source_url"]: manifest_path.with_name(
            "qwen2.5-0.5b-instruct.config.json"
        ).read_bytes(),
        manifest["records"][2]["source_url"]: manifest_path.with_name(
            "deepseek-v3.config.json"
        ).read_bytes(),
    }


def test_offline_release_evidence_locks_three_distinct_evidence_types() -> None:
    report = verify_model_release_evidence(MANIFEST)
    payload = report.to_dict()

    assert report.manifest_fingerprint == (
        "sha256:74166133716bfebddb444587e9f9a012b4beada923f5209482308ff61194953b"
    )
    assert report.projection_fingerprint == (
        "sha256:40b3fe7b2a9c054ea6aa17e9e747d1831b8ae41ee3d55130c916f818acbe4638"
    )
    assert payload["upstream_verified"] is False
    assert payload["evidence_boundary"] == MODEL_RELEASE_EVIDENCE_BOUNDARY
    records = {record["record_id"]: record for record in payload["records"]}

    llama = records["llama-3.2-text-model-card"]
    assert llama["source_fragments_verified"] is False
    assert llama["vendor_reported"]["evidence_type"] == (
        "vendor_model_card_claims_not_independent_measurements"
    )
    assert llama["vendor_reported"]["reported_context_length"] == "128k"

    llama["family"] = "attacker-rewrite"
    assert report.records[0]["family"] == "Llama"
    with pytest.raises(TypeError):
        report.records[0]["family"] = "attacker-rewrite"  # type: ignore[index]

    qwen = records["qwen2.5-0.5b-instruct-config"]
    assert qwen["contract"] == {
        "architectures": ["Qwen2ForCausalLM"],
        "attention_kind": "gqa",
        "head_dim": 64,
        "known_mla_markers_present": False,
        "known_moe_markers_present": False,
        "model_type": "qwen2",
        "query_heads_per_kv_head": 7,
        "standard_kv_applicable": True,
    }
    assert qwen["standard_kv_estimates"][0]["total_bytes"] == 402_653_184
    assert qwen["standard_kv_estimates"][0]["ideal_tensor_payload_only"] is True

    deepseek = records["deepseek-v3-config"]
    assert deepseek["contract"]["known_mla_markers_present"] is True
    assert deepseek["contract"]["known_moe_markers_present"] is True
    assert deepseek["estimate_refused"] is True
    assert deepseek["standard_kv_estimates"] == []
    assert "standard dense K/V formula must not be applied" in deepseek[
        "estimate_refusal_reason"
    ]


def test_optional_upstream_mode_checks_bytes_json_and_vendor_fragments(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _copy_evidence(tmp_path)
    upstream = _network_free_upstream_fixture(manifest_path, manifest)

    report = verify_model_release_evidence(
        manifest_path,
        verify_upstream=True,
        fetcher=upstream.__getitem__,
    )

    assert report.upstream_verified is True
    assert all(record["upstream_verified"] is True for record in report.records)
    assert report.records[0]["source_fragments_verified"] is True


def test_local_semantic_snapshot_tampering_fails_closed(tmp_path: Path) -> None:
    manifest_path, _ = _copy_evidence(tmp_path)
    qwen_path = manifest_path.with_name("qwen2.5-0.5b-instruct.config.json")
    config = json.loads(qwen_path.read_text(encoding="utf-8"))
    config["num_key_value_heads"] = 1
    qwen_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="local semantic snapshot hash mismatch"):
        verify_model_release_evidence(manifest_path)


def test_upstream_raw_byte_tampering_fails_before_semantic_comparison(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _copy_evidence(tmp_path)
    upstream = _network_free_upstream_fixture(manifest_path, manifest)
    qwen_url = manifest["records"][1]["source_url"]
    upstream[qwen_url] += b"\n"

    with pytest.raises(ValueError, match="upstream byte length mismatch"):
        verify_model_release_evidence(
            manifest_path,
            verify_upstream=True,
            fetcher=upstream.__getitem__,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["records"][1]["expected_contract"].update(
                {"head_dim": 128}
            ),
            "expected_contract does not match inspection",
        ),
        (
            lambda manifest: manifest["records"][1]["kv_scenarios"][0].update(
                {"expected_total_bytes": 1}
            ),
            "total KV estimate drift",
        ),
        (
            lambda manifest: manifest["records"][1].update(
                {"local_snapshot_path": "../qwen.json"}
            ),
            "must be contained",
        ),
        (
            lambda manifest: manifest["records"][1].update({"unexpected": True}),
            "field set mismatch",
        ),
    ],
)
def test_manifest_drift_is_rejected(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    manifest_path, manifest = _copy_evidence(tmp_path)
    mutation(manifest)
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        verify_model_release_evidence(manifest_path)


def test_duplicate_snapshot_json_is_rejected_before_hash_comparison(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _copy_evidence(tmp_path)
    qwen_path = manifest_path.with_name("qwen2.5-0.5b-instruct.config.json")
    qwen_path.write_text('{"model_type":"qwen2","model_type":"drift"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        verify_model_release_evidence(manifest_path)


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/model/config.json",
        "https://example.com/model/config.json",
        "https://user:secret@huggingface.co/model/config.json",
    ],
)
def test_fetcher_rejects_untrusted_source_url_without_network(url: str) -> None:
    with pytest.raises(ValueError, match="allowlisted public HTTPS origin"):
        fetch_release_artifact(url)

