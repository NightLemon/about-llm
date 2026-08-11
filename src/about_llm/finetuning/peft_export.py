"""Strict, complete-file-set verification for a PEFT deployment export directory."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

from about_llm.llmops import canonical_json_bytes

PEFT_EXPORT_MANIFEST_VERSION = "about-llm.peft-export-manifest.v1"
PEFT_EXPORT_MANIFEST_FILENAME = "about-llm-export-manifest.json"

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_FIELDS = {
    "contract",
    "file_set_sha256",
    "files",
    "identity",
    "schema_version",
}
_IDENTITY_FIELDS = {
    "architecture",
    "artifact_id",
    "base_model_id",
    "base_revision",
    "tokenizer_revision",
}
_CONTRACT_FIELDS = {
    "adapter_directory",
    "adapter_type",
    "base_directory",
    "merged_directory",
    "target_modules",
    "task_type",
    "tokenizer_directory",
}
_FILE_FIELDS = {"bytes", "path", "sha256"}
_REQUIRED_FILES = {
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "base/config.json",
    "base/model.safetensors",
    "merged/config.json",
    "merged/model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
}
_ALLOWED_ROOT_DIRECTORIES = {"adapter", "base", "merged", "tokenizer"}


@dataclass(frozen=True)
class PEFTExportIdentity:
    artifact_id: str
    architecture: str
    base_model_id: str
    base_revision: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _identity_string(value, name)

    def to_dict(self) -> dict[str, str]:
        return cast(dict[str, str], asdict(self))


@dataclass(frozen=True)
class PEFTExportLimits:
    max_manifest_bytes: int = 4 * 1024 * 1024
    max_files: int = 10_000
    max_file_bytes: int = 16 * 1024 * 1024 * 1024
    max_total_file_bytes: int = 64 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PEFTExportVerification:
    identity: PEFTExportIdentity
    file_count: int
    total_file_bytes: int
    manifest_bytes: int
    file_set_sha256: str
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "file_count": self.file_count,
            "total_file_bytes": self.total_file_bytes,
            "manifest_bytes": self.manifest_bytes,
            "file_set_sha256": self.file_set_sha256,
            "files": list(self.files),
        }


def write_peft_export_manifest_new(
    root: Path,
    *,
    identity: PEFTExportIdentity,
    target_modules: tuple[str, ...],
    limits: PEFTExportLimits | None = None,
) -> PEFTExportVerification:
    """Write a canonical manifest without replacing an existing target."""

    exact_limits = PEFTExportLimits() if limits is None else limits
    _validate_root(root)
    manifest_path = root / PEFT_EXPORT_MANIFEST_FILENAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"refusing to replace existing manifest: {manifest_path}")
    modules = _target_modules(target_modules)
    files = _describe_directory_files(root, limits=exact_limits)
    _required_files(files)
    contract = _contract(modules)
    _validate_semantics(root, identity=identity, contract=contract)
    file_set_sha256 = _file_set_sha256(files)
    manifest = {
        "schema_version": PEFT_EXPORT_MANIFEST_VERSION,
        "identity": identity.to_dict(),
        "contract": contract,
        "files": files,
        "file_set_sha256": file_set_sha256,
    }
    payload = canonical_json_bytes(manifest)
    if len(payload) > exact_limits.max_manifest_bytes:
        raise ValueError("PEFT export manifest exceeds configured limit")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return verify_peft_export_directory(root, limits=exact_limits)


def verify_peft_export_directory(
    root: Path,
    *,
    limits: PEFTExportLimits | None = None,
) -> PEFTExportVerification:
    """Fail closed unless the complete directory matches its strict manifest."""

    exact_limits = PEFTExportLimits() if limits is None else limits
    _validate_root(root)
    manifest_path = root / PEFT_EXPORT_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("PEFT export manifest must be a regular file")
    manifest_size = manifest_path.stat().st_size
    if manifest_size > exact_limits.max_manifest_bytes:
        raise ValueError("PEFT export manifest exceeds configured limit")
    manifest_bytes = manifest_path.read_bytes()
    record = _strict_json_object(manifest_bytes, "PEFT export manifest")
    _exact_fields(record, _MANIFEST_FIELDS, "PEFT export manifest")
    if record["schema_version"] != PEFT_EXPORT_MANIFEST_VERSION:
        raise ValueError("unsupported PEFT export manifest schema_version")
    identity = _parse_identity(record["identity"])
    contract = _parse_contract(record["contract"])
    files = _parse_file_descriptors(record["files"], limits=exact_limits)
    _required_files(files)
    expected_file_set = _file_set_sha256(files)
    if record["file_set_sha256"] != expected_file_set:
        raise ValueError("PEFT export file_set_sha256 mismatch")
    actual = _describe_directory_files(root, limits=exact_limits)
    if actual != files:
        raise ValueError("PEFT export directory file set, size, or digest mismatch")
    _validate_semantics(root, identity=identity, contract=contract)
    total = sum(cast(int, descriptor["bytes"]) for descriptor in files)
    return PEFTExportVerification(
        identity=identity,
        file_count=len(files),
        total_file_bytes=total,
        manifest_bytes=len(manifest_bytes),
        file_set_sha256=expected_file_set,
        files=tuple(cast(str, descriptor["path"]) for descriptor in files),
    )


def _contract(target_modules: tuple[str, ...]) -> dict[str, object]:
    return {
        "base_directory": "base",
        "adapter_directory": "adapter",
        "merged_directory": "merged",
        "tokenizer_directory": "tokenizer",
        "adapter_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": list(target_modules),
    }


def _describe_directory_files(
    root: Path, *, limits: PEFTExportLimits
) -> list[dict[str, object]]:
    descriptors: list[dict[str, object]] = []
    total = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            path = current_path / directory
            if path.is_symlink():
                raise ValueError(f"PEFT export directory contains symlink: {path}")
        for filename in filenames:
            path = current_path / filename
            relative = path.relative_to(root).as_posix()
            if relative == PEFT_EXPORT_MANIFEST_FILENAME:
                continue
            if path.is_symlink():
                raise ValueError(f"PEFT export contains symlink: {path}")
            if not path.is_file():
                raise ValueError(f"PEFT export contains non-regular file: {path}")
            _safe_relative_path(relative)
            size = path.stat().st_size
            if size > limits.max_file_bytes:
                raise ValueError(f"PEFT export file exceeds configured limit: {relative}")
            total += size
            if total > limits.max_total_file_bytes:
                raise ValueError("PEFT export total file bytes exceed configured limit")
            descriptors.append(
                {
                    "path": relative,
                    "bytes": size,
                    "sha256": _sha256_file(path),
                }
            )
            if len(descriptors) > limits.max_files:
                raise ValueError("PEFT export file count exceeds configured limit")
    descriptors.sort(key=lambda item: cast(str, item["path"]))
    return descriptors


def _parse_file_descriptors(
    value: object, *, limits: PEFTExportLimits
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("PEFT export files must be a non-empty array")
    if len(value) > limits.max_files:
        raise ValueError("PEFT export file count exceeds configured limit")
    descriptors: list[dict[str, object]] = []
    total = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"PEFT export files[{index}] must be an object")
        record = cast(dict[str, Any], item)
        _exact_fields(record, _FILE_FIELDS, f"PEFT export files[{index}]")
        path = _safe_relative_path(record["path"])
        size = _nonnegative_integer(record["bytes"], f"files[{index}].bytes")
        if size > limits.max_file_bytes:
            raise ValueError(f"PEFT export file exceeds configured limit: {path}")
        digest = record["sha256"]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"files[{index}].sha256 must be lowercase SHA-256")
        total += size
        if total > limits.max_total_file_bytes:
            raise ValueError("PEFT export total file bytes exceed configured limit")
        descriptors.append({"path": path, "bytes": size, "sha256": digest})
    paths = [cast(str, item["path"]) for item in descriptors]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("PEFT export file descriptors must be unique and path-sorted")
    return descriptors


def _parse_identity(value: object) -> PEFTExportIdentity:
    if not isinstance(value, dict):
        raise ValueError("PEFT export identity must be an object")
    record = cast(dict[str, Any], value)
    _exact_fields(record, _IDENTITY_FIELDS, "PEFT export identity")
    return PEFTExportIdentity(
        artifact_id=_identity_string(record["artifact_id"], "artifact_id"),
        architecture=_identity_string(record["architecture"], "architecture"),
        base_model_id=_identity_string(record["base_model_id"], "base_model_id"),
        base_revision=_identity_string(record["base_revision"], "base_revision"),
        tokenizer_revision=_identity_string(
            record["tokenizer_revision"], "tokenizer_revision"
        ),
    )


def _parse_contract(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("PEFT export contract must be an object")
    record = cast(dict[str, Any], value)
    _exact_fields(record, _CONTRACT_FIELDS, "PEFT export contract")
    expected_directories = {
        "base_directory": "base",
        "adapter_directory": "adapter",
        "merged_directory": "merged",
        "tokenizer_directory": "tokenizer",
    }
    for field, expected in expected_directories.items():
        if record[field] != expected:
            raise ValueError(f"PEFT export {field} must be {expected!r}")
    if record["adapter_type"] != "LORA" or record["task_type"] != "CAUSAL_LM":
        raise ValueError("PEFT export adapter/task contract is unsupported")
    modules_value = record["target_modules"]
    if not isinstance(modules_value, list):
        raise ValueError("PEFT export target_modules must be an array")
    modules = _target_modules(tuple(modules_value))
    return _contract(modules)


def _validate_semantics(
    root: Path,
    *,
    identity: PEFTExportIdentity,
    contract: dict[str, object],
) -> None:
    base = _load_json_file(root / "base" / "config.json", "base config")
    merged = _load_json_file(root / "merged" / "config.json", "merged config")
    adapter = _load_json_file(
        root / "adapter" / "adapter_config.json", "adapter config"
    )
    tokenizer_config = _load_json_file(
        root / "tokenizer" / "tokenizer_config.json", "tokenizer config"
    )
    tokenizer = _load_json_file(
        root / "tokenizer" / "tokenizer.json", "tokenizer payload"
    )
    model_type = _identity_string(base.get("model_type"), "base model_type")
    if merged.get("model_type") != model_type:
        raise ValueError("base/merged config semantic mismatch for model_type")
    vocab_size = _nonnegative_integer(base.get("vocab_size"), "base vocab_size")
    if vocab_size == 0 or merged.get("vocab_size") != vocab_size:
        raise ValueError("base/merged config semantic mismatch for vocab_size")
    token_ids: dict[str, int] = {}
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = _nonnegative_integer(base.get(field), f"base {field}")
        if token_id >= vocab_size or merged.get(field) != token_id:
            raise ValueError(f"base/merged config semantic mismatch for {field}")
        token_ids[field] = token_id
    architectures = base.get("architectures")
    merged_architectures = merged.get("architectures")
    if (
        not isinstance(architectures, list)
        or identity.architecture not in architectures
        or merged_architectures != architectures
    ):
        raise ValueError("base/merged architecture identity mismatch")
    if merged != base:
        raise ValueError("base/merged config payload mismatch")
    if adapter.get("base_model_name_or_path") != identity.base_model_id:
        raise ValueError("adapter config base_model_name_or_path identity mismatch")
    if adapter.get("peft_type") != contract["adapter_type"]:
        raise ValueError("adapter config peft_type mismatch")
    if adapter.get("task_type") != contract["task_type"]:
        raise ValueError("adapter config task_type mismatch")
    target_modules_value = adapter.get("target_modules")
    if not isinstance(target_modules_value, list):
        raise ValueError("adapter config target_modules must be an array")
    target_modules = _target_modules(tuple(target_modules_value))
    if list(target_modules) != contract["target_modules"]:
        raise ValueError("adapter config target_modules mismatch")
    _validate_weight_semantics(root, target_modules=target_modules)
    chat_template = tokenizer_config.get("chat_template")
    template_path = root / "tokenizer" / "chat_template.jinja"
    if not (
        (isinstance(chat_template, str) and chat_template.strip())
        or (
            template_path.is_file()
            and template_path.read_text(encoding="utf-8").strip()
        )
    ):
        raise ValueError("tokenizer export is missing a non-empty chat template")
    vocab = _effective_tokenizer_vocabulary(tokenizer)
    if set(vocab.values()) != set(range(vocab_size)):
        raise ValueError("tokenizer effective vocabulary ids differ from model config")
    for token_field, id_field in (
        ("bos_token", "bos_token_id"),
        ("eos_token", "eos_token_id"),
        ("pad_token", "pad_token_id"),
    ):
        token = _token_content(tokenizer_config.get(token_field), token_field)
        if vocab.get(token) != token_ids[id_field]:
            raise ValueError(f"tokenizer {token_field} id differs from model config")


def _validate_weight_semantics(
    root: Path, *, target_modules: tuple[str, ...]
) -> None:
    base = _safetensors_signature(
        root / "base" / "model.safetensors", "base model weights"
    )
    merged = _safetensors_signature(
        root / "merged" / "model.safetensors", "merged model weights"
    )
    if base != merged:
        raise ValueError("base/merged safetensors tensor signatures mismatch")
    adapter = _safetensors_signature(
        root / "adapter" / "adapter_model.safetensors", "adapter weights"
    )
    for module in target_modules:
        a_marker = f".{module}.lora_A."
        b_marker = f".{module}.lora_B."
        has_a = any(a_marker in name or name.startswith(a_marker[1:]) for name in adapter)
        has_b = any(b_marker in name or name.startswith(b_marker[1:]) for name in adapter)
        if not has_a or not has_b:
            raise ValueError(
                f"adapter weights must contain LoRA A/B tensors for target module {module!r}"
            )


def _safetensors_signature(
    path: Path, label: str
) -> dict[str, tuple[str, tuple[int, ...]]]:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - exercised by packaging, not CI
        raise RuntimeError(
            "strict PEFT export verification requires the safetensors package"
        ) from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular safetensors file")
    signature: dict[str, tuple[str, tuple[int, ...]]] = {}
    try:
        with safe_open(path, framework="numpy") as handle:
            names = handle.keys()
            for name in names:
                tensor = handle.get_slice(name)
                signature[name] = (
                    str(tensor.get_dtype()),
                    tuple(tensor.get_shape()),
                )
    except Exception as error:
        raise ValueError(f"{label} must be a valid safetensors file") from error
    if not signature:
        raise ValueError(f"{label} must contain at least one tensor")
    return signature


def _effective_tokenizer_vocabulary(tokenizer: dict[str, Any]) -> dict[str, int]:
    model = tokenizer.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise ValueError("tokenizer payload must contain model.vocab")
    vocab: dict[str, int] = {}
    ids: dict[int, str] = {}
    for token, value in cast(dict[str, Any], model["vocab"]).items():
        _add_vocabulary_entry(vocab, ids, token=token, value=value, label="model.vocab")
    added_tokens = tokenizer.get("added_tokens", [])
    if not isinstance(added_tokens, list):
        raise ValueError("tokenizer added_tokens must be an array")
    for index, item in enumerate(added_tokens):
        if not isinstance(item, dict):
            raise ValueError(f"tokenizer added_tokens[{index}] must be an object")
        _add_vocabulary_entry(
            vocab,
            ids,
            token=item.get("content"),
            value=item.get("id"),
            label=f"added_tokens[{index}]",
        )
    return vocab


def _add_vocabulary_entry(
    vocab: dict[str, int],
    ids: dict[int, str],
    *,
    token: object,
    value: object,
    label: str,
) -> None:
    token_value = _identity_string(token, f"tokenizer {label} token")
    token_id = _nonnegative_integer(value, f"tokenizer {label} id")
    existing_id = vocab.get(token_value)
    existing_token = ids.get(token_id)
    if existing_id is not None and existing_id != token_id:
        raise ValueError(f"tokenizer {label} changes an existing token id")
    if existing_token is not None and existing_token != token_value:
        raise ValueError(f"tokenizer {label} reuses an id for a different token")
    vocab[token_value] = token_id
    ids[token_id] = token_value


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return _strict_json_object(path.read_bytes(), label, require_canonical=False)


def _strict_json_object(
    value: bytes, label: str, *, require_canonical: bool = True
) -> dict[str, Any]:
    try:
        decoded = value.decode("utf-8", errors="strict")
        payload: Any = json.loads(
            decoded,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    record = cast(dict[str, Any], payload)
    if require_canonical and canonical_json_bytes(record) != value:
        raise ValueError(f"{label} must use canonical JSON encoding")
    return record


def _required_files(files: list[dict[str, object]]) -> None:
    paths = {cast(str, item["path"]) for item in files}
    missing = sorted(_REQUIRED_FILES - paths)
    if missing:
        raise ValueError(f"PEFT export is missing required files: {missing}")


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError("PEFT export file path must be a non-empty bounded string")
    if "\\" in value or ":" in value:
        raise ValueError("PEFT export file path must use safe POSIX-relative syntax")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("PEFT export file path must not traverse or be absolute")
    if path.as_posix() != value or path.parts[0] not in _ALLOWED_ROOT_DIRECTORIES:
        raise ValueError("PEFT export file path is outside the allowed directories")
    return value


def _target_modules(value: tuple[object, ...]) -> tuple[str, ...]:
    modules = tuple(_identity_string(item, "target module") for item in value)
    if not modules or len(modules) != len(set(modules)) or modules != tuple(sorted(modules)):
        raise ValueError("target_modules must be non-empty, unique, and sorted")
    return modules


def _file_set_sha256(files: list[dict[str, object]]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(files)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _token_content(value: object, name: str) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str) and content:
            return content
    raise ValueError(f"tokenizer {name} must identify a non-empty token")


def _identity_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{name} must be a non-empty bounded string")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _exact_fields(record: dict[str, Any], expected: set[str], label: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{label} fields are invalid")


def _validate_root(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("PEFT export root must be an existing regular directory")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
