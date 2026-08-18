from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file

from about_llm.finetuning.peft_export import (
    PEFT_EXPORT_MANIFEST_FILENAME,
    PEFTExportIdentity,
    PEFTExportLimits,
    verify_peft_export_directory,
    write_peft_export_manifest_new,
)
from about_llm.llmops import canonical_json_bytes

pytestmark = pytest.mark.contract


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _fixture(root: Path) -> PEFTExportIdentity:
    base_config = {
        "architectures": ["GPT2LMHeadModel"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "model_type": "gpt2",
        "pad_token_id": 0,
        "vocab_size": 4,
    }
    _write_json(root / "base" / "config.json", base_config)
    save_file(
        {"transformer.wte.weight": np.zeros((4, 2), dtype=np.float32)},
        root / "base" / "model.safetensors",
    )
    _write_json(root / "merged" / "config.json", base_config)
    save_file(
        {"transformer.wte.weight": np.ones((4, 2), dtype=np.float32)},
        root / "merged" / "model.safetensors",
    )
    _write_json(
        root / "adapter" / "adapter_config.json",
        {
            "base_model_name_or_path": "authored-base",
            "peft_type": "LORA",
            "target_modules": ["c_attn"],
            "task_type": "CAUSAL_LM",
        },
    )
    save_file(
        {
            "base_model.model.h.0.c_attn.lora_A.weight": np.zeros(
                (2, 2), dtype=np.float32
            ),
            "base_model.model.h.0.c_attn.lora_B.weight": np.zeros(
                (6, 2), dtype=np.float32
            ),
        },
        root / "adapter" / "adapter_model.safetensors",
    )
    _write_json(
        root / "tokenizer" / "tokenizer_config.json",
        {
            "bos_token": "<bos>",
            "chat_template": "{{ messages[0]['content'] }}",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        },
    )
    _write_json(
        root / "tokenizer" / "tokenizer.json",
        {
            "model": {
                "type": "WordLevel",
                "vocab": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3},
            }
        },
    )
    return PEFTExportIdentity(
        artifact_id="authored-peft-export",
        architecture="GPT2LMHeadModel",
        base_model_id="authored-base",
        base_revision="fixture-seed-31",
        tokenizer_revision="wordlevel-v1",
    )


def _manifest(root: Path) -> dict[str, Any]:
    value: Any = json.loads((root / PEFT_EXPORT_MANIFEST_FILENAME).read_bytes())
    assert isinstance(value, dict)
    return value


def _rewrite_manifest(root: Path, manifest: dict[str, Any]) -> None:
    manifest["file_set_sha256"] = "sha256:" + hashlib.sha256(
        canonical_json_bytes(manifest["files"])
    ).hexdigest()
    (root / PEFT_EXPORT_MANIFEST_FILENAME).write_bytes(canonical_json_bytes(manifest))


def _refresh_file(root: Path, relative_path: str) -> None:
    manifest = _manifest(root)
    descriptor = next(
        item for item in manifest["files"] if item["path"] == relative_path
    )
    payload = (root / relative_path).read_bytes()
    descriptor["bytes"] = len(payload)
    descriptor["sha256"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    _rewrite_manifest(root, manifest)


def test_peft_export_manifest_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    identity = _fixture(tmp_path)
    report = write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    assert report.identity == identity
    assert report.file_count == 8
    assert report.total_file_bytes == sum(
        path.stat().st_size
        for path in tmp_path.rglob("*")
        if path.is_file() and path.name != PEFT_EXPORT_MANIFEST_FILENAME
    )
    assert report.files == tuple(sorted(report.files))
    assert report.file_set_sha256.startswith("sha256:")
    assert verify_peft_export_directory(tmp_path) == report
    with pytest.raises(FileExistsError):
        write_peft_export_manifest_new(
            tmp_path, identity=identity, target_modules=("c_attn",)
        )


@pytest.mark.parametrize("relative_path", ["base/model.safetensors", "tokenizer/tokenizer.json"])
def test_peft_export_rejects_file_content_drift(
    tmp_path: Path, relative_path: str
) -> None:
    identity = _fixture(tmp_path)
    write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    with (tmp_path / relative_path).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="file set, size, or digest"):
        verify_peft_export_directory(tmp_path)


def test_peft_export_rejects_extra_and_missing_files(tmp_path: Path) -> None:
    identity = _fixture(tmp_path)
    write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    (tmp_path / "adapter" / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        verify_peft_export_directory(tmp_path)
    (tmp_path / "adapter" / "extra.json").unlink()
    (tmp_path / "merged" / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="file set"):
        verify_peft_export_directory(tmp_path)


def test_peft_export_rejects_noncanonical_duplicate_and_unknown_manifest(
    tmp_path: Path,
) -> None:
    identity = _fixture(tmp_path)
    write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    manifest = _manifest(tmp_path)
    path = tmp_path / PEFT_EXPORT_MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON"):
        verify_peft_export_directory(tmp_path)
    path.write_bytes(
        canonical_json_bytes(manifest).replace(
            b'"schema_version":', b'"schema_version":"duplicate","schema_version":', 1
        )
    )
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        verify_peft_export_directory(tmp_path)
    manifest["unknown"] = True
    path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="fields are invalid"):
        verify_peft_export_directory(tmp_path)


def test_peft_export_rejects_traversal_duplicate_and_unsorted_descriptors(
    tmp_path: Path,
) -> None:
    identity = _fixture(tmp_path)
    write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    manifest = _manifest(tmp_path)
    manifest["files"][0]["path"] = "adapter/../outside"
    _rewrite_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="traverse"):
        verify_peft_export_directory(tmp_path)

    manifest = _manifest_from_fresh(tmp_path, identity)
    manifest["files"][1] = dict(manifest["files"][0])
    _rewrite_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="unique and path-sorted"):
        verify_peft_export_directory(tmp_path)

    manifest = _manifest_from_fresh(tmp_path, identity)
    manifest["files"][0], manifest["files"][1] = (
        manifest["files"][1],
        manifest["files"][0],
    )
    _rewrite_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="unique and path-sorted"):
        verify_peft_export_directory(tmp_path)


def _manifest_from_fresh(root: Path, identity: PEFTExportIdentity) -> dict[str, Any]:
    (root / PEFT_EXPORT_MANIFEST_FILENAME).unlink()
    write_peft_export_manifest_new(
        root, identity=identity, target_modules=("c_attn",)
    )
    return _manifest(root)


@pytest.mark.parametrize(
    ("relative_path", "field", "value", "message"),
    [
        (
            "adapter/adapter_config.json",
            "base_model_name_or_path",
            "wrong-base",
            "base_model_name_or_path",
        ),
        ("adapter/adapter_config.json", "task_type", "SEQ_CLS", "task_type"),
        ("merged/config.json", "vocab_size", 5, "vocab_size"),
        ("tokenizer/tokenizer_config.json", "chat_template", "", "chat template"),
    ],
)
def test_peft_export_rejects_semantic_drift_after_manifest_rehash(
    tmp_path: Path,
    relative_path: str,
    field: str,
    value: object,
    message: str,
) -> None:
    identity = _fixture(tmp_path)
    write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    path = tmp_path / relative_path
    payload: Any = json.loads(path.read_bytes())
    payload[field] = value
    path.write_bytes(canonical_json_bytes(payload))
    _refresh_file(tmp_path, relative_path)
    with pytest.raises(ValueError, match=message):
        verify_peft_export_directory(tmp_path)


def test_peft_export_rejects_tokenizer_vocab_and_special_token_drift(
    tmp_path: Path,
) -> None:
    identity = _fixture(tmp_path)
    write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    tokenizer_path = tmp_path / "tokenizer" / "tokenizer.json"
    tokenizer: Any = json.loads(tokenizer_path.read_bytes())
    del tokenizer["model"]["vocab"]["<unk>"]
    tokenizer_path.write_bytes(canonical_json_bytes(tokenizer))
    _refresh_file(tmp_path, "tokenizer/tokenizer.json")
    with pytest.raises(ValueError, match="effective vocabulary"):
        verify_peft_export_directory(tmp_path)

    second = tmp_path / "second"
    second.mkdir()
    second_identity = _fixture(second)
    write_peft_export_manifest_new(
        second, identity=second_identity, target_modules=("c_attn",)
    )
    config_path = second / "tokenizer" / "tokenizer_config.json"
    config: Any = json.loads(config_path.read_bytes())
    config["eos_token"] = "<unk>"
    config_path.write_bytes(canonical_json_bytes(config))
    _refresh_file(second, "tokenizer/tokenizer_config.json")
    with pytest.raises(ValueError, match="eos_token id"):
        verify_peft_export_directory(second)


def test_peft_export_rejects_full_config_drift_after_manifest_rehash(
    tmp_path: Path,
) -> None:
    identity = _fixture(tmp_path)
    write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    merged_path = tmp_path / "merged" / "config.json"
    merged: Any = json.loads(merged_path.read_bytes())
    merged["n_layer"] = 99
    merged_path.write_bytes(canonical_json_bytes(merged))
    _refresh_file(tmp_path, "merged/config.json")
    with pytest.raises(ValueError, match="config payload mismatch"):
        verify_peft_export_directory(tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "base/model.safetensors",
        "adapter/adapter_model.safetensors",
        "merged/model.safetensors",
    ],
)
def test_peft_export_rejects_invalid_safetensors_after_manifest_rehash(
    tmp_path: Path, relative_path: str
) -> None:
    identity = _fixture(tmp_path)
    write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    (tmp_path / relative_path).write_bytes(b"coordinated invalid replacement")
    _refresh_file(tmp_path, relative_path)
    with pytest.raises(ValueError, match="valid safetensors"):
        verify_peft_export_directory(tmp_path)


def test_peft_export_rejects_weight_signature_and_lora_coverage_drift(
    tmp_path: Path,
) -> None:
    identity = _fixture(tmp_path)
    write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    merged_path = tmp_path / "merged" / "model.safetensors"
    save_file(
        {"transformer.wte.weight": np.ones((5, 2), dtype=np.float32)}, merged_path
    )
    _refresh_file(tmp_path, "merged/model.safetensors")
    with pytest.raises(ValueError, match="tensor signatures mismatch"):
        verify_peft_export_directory(tmp_path)

    second = tmp_path / "second"
    second.mkdir()
    second_identity = _fixture(second)
    write_peft_export_manifest_new(
        second, identity=second_identity, target_modules=("c_attn",)
    )
    adapter_path = second / "adapter" / "adapter_model.safetensors"
    save_file(
        {
            "base_model.model.h.0.c_attn.lora_A.weight": np.zeros(
                (2, 2), dtype=np.float32
            )
        },
        adapter_path,
    )
    _refresh_file(second, "adapter/adapter_model.safetensors")
    with pytest.raises(ValueError, match="LoRA A/B tensors"):
        verify_peft_export_directory(second)


def test_peft_export_resource_limits_and_target_module_contract(tmp_path: Path) -> None:
    identity = _fixture(tmp_path)
    with pytest.raises(ValueError, match="target_modules"):
        write_peft_export_manifest_new(
            tmp_path, identity=identity, target_modules=("z", "a")
        )
    with pytest.raises(ValueError, match="target module"):
        write_peft_export_manifest_new(
            tmp_path,
            identity=identity,
            target_modules=("a", 1),  # type: ignore[arg-type]
        )
    report = write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    with pytest.raises(ValueError, match="manifest exceeds"):
        verify_peft_export_directory(
            tmp_path,
            limits=PEFTExportLimits(max_manifest_bytes=report.manifest_bytes - 1),
        )
    with pytest.raises(ValueError, match="file count"):
        verify_peft_export_directory(tmp_path, limits=PEFTExportLimits(max_files=1))
    with pytest.raises(ValueError, match="file exceeds"):
        verify_peft_export_directory(
            tmp_path, limits=PEFTExportLimits(max_file_bytes=1)
        )
    with pytest.raises(ValueError, match="total file bytes"):
        verify_peft_export_directory(
            tmp_path, limits=PEFTExportLimits(max_total_file_bytes=1)
        )


def test_peft_export_rejects_symlink_when_supported(tmp_path: Path) -> None:
    identity = _fixture(tmp_path)
    link = tmp_path / "base" / "linked.safetensors"
    try:
        os.symlink(tmp_path / "base" / "model.safetensors", link)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows configuration")
    with pytest.raises(ValueError, match="symlink"):
        write_peft_export_manifest_new(
            tmp_path, identity=identity, target_modules=("c_attn",)
        )


def test_peft_export_accepts_consistent_added_token_vocabulary(tmp_path: Path) -> None:
    identity = _fixture(tmp_path)
    tokenizer_path = tmp_path / "tokenizer" / "tokenizer.json"
    tokenizer: Any = json.loads(tokenizer_path.read_bytes())
    del tokenizer["model"]["vocab"]["<unk>"]
    tokenizer["added_tokens"] = [{"content": "<unk>", "id": 3}]
    tokenizer_path.write_bytes(canonical_json_bytes(tokenizer))
    report = write_peft_export_manifest_new(
        tmp_path, identity=identity, target_modules=("c_attn",)
    )
    assert report.file_count == 8
