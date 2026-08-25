"""Publish and verify a lightweight PEFT adapter bundle for SFT runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

import numpy as np

from about_llm.finetuning.training_runtime import write_strict_json
from about_llm.llmops import artifact_fingerprint

SFT_ADAPTER_BUNDLE_VERSION = "about-llm.sft-adapter-bundle.v1"
SFT_ADAPTER_BUNDLE_MANIFEST = "about-llm-adapter-bundle.json"

_REPORT_FILES = (
    "sft-data-audit.json",
    "sft-training-readiness.json",
    "sft-template-mask-audit.json",
    "sft-final-label-audit.json",
    "sft-training-run.json",
)
_MANIFEST_FIELDS = {
    "bundle_fingerprint",
    "contract",
    "evidence",
    "files",
    "identity",
    "schema_version",
}
_IDENTITY_FIELDS = {"model_id", "revision"}
_CONTRACT_FIELDS = {
    "adapter_directory",
    "adapter_type",
    "alpha",
    "quantized_base",
    "rank",
    "target_modules",
    "task_type",
    "tokenizer_directory",
}
_EVIDENCE_FIELDS = {
    "assistant_mask_manifest_fingerprint",
    "final_labels_fingerprint",
    "optimizer_step_count",
    "readiness_manifest_fingerprint",
    "training_report_version",
}
_FILE_FIELDS = {"bytes", "path", "sha256"}
_TOKENIZER_PAYLOAD_NAMES = {
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
}
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_FILES = 1_000
_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class SFTAdapterBundleIdentity:
    model_id: str
    revision: str

    def __post_init__(self) -> None:
        _nonempty_string(self.model_id, "model_id")
        _nonempty_string(self.revision, "revision")

    def to_dict(self) -> dict[str, str]:
        return cast(dict[str, str], asdict(self))


@dataclass(frozen=True)
class SFTAdapterBundleContract:
    target_modules: tuple[str, ...]
    rank: int
    alpha: int
    quantized_base: bool

    def __post_init__(self) -> None:
        _target_modules(list(self.target_modules))
        _positive_integer(self.rank, "rank")
        _positive_integer(self.alpha, "alpha")
        if not isinstance(self.quantized_base, bool):
            raise ValueError("quantized_base must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_directory": "adapter",
            "tokenizer_directory": "tokenizer",
            "adapter_type": "LORA",
            "task_type": "CAUSAL_LM",
            "target_modules": list(self.target_modules),
            "rank": self.rank,
            "alpha": self.alpha,
            "quantized_base": self.quantized_base,
        }


@dataclass(frozen=True)
class SFTAdapterBundleVerification:
    identity: SFTAdapterBundleIdentity
    contract: SFTAdapterBundleContract
    bundle_fingerprint: str
    file_count: int
    total_file_bytes: int
    adapter_tensor_count: int
    optimizer_step_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "contract": self.contract.to_dict(),
            "bundle_fingerprint": self.bundle_fingerprint,
            "file_count": self.file_count,
            "total_file_bytes": self.total_file_bytes,
            "adapter_tensor_count": self.adapter_tensor_count,
            "optimizer_step_count": self.optimizer_step_count,
        }


def bind_peft_adapter_identity(model: Any, *, model_id: str, revision: str) -> None:
    """Bind the saved PEFT config to the exact base used by this run."""

    exact_model_id = _nonempty_string(model_id, "model_id")
    exact_revision = _nonempty_string(revision, "revision")
    configurations = getattr(model, "peft_config", None)
    if not isinstance(configurations, dict) or set(configurations) != {"default"}:
        raise RuntimeError("expected exactly one PEFT adapter named 'default'")
    configuration = configurations["default"]
    configuration.base_model_name_or_path = exact_model_id
    configuration.revision = exact_revision


def publish_sft_adapter_bundle(
    training_output: Path, *, bundle_directory: Path | None = None
) -> SFTAdapterBundleVerification:
    """Copy one completed run into a strict, reloadable adapter bundle."""

    if training_output.is_symlink():
        raise ValueError("training_output must not be a symlink")
    source = training_output.resolve()
    if not source.is_dir():
        raise ValueError("training_output must be a regular directory")
    target = (
        source / "adapter-bundle"
        if bundle_directory is None
        else bundle_directory.resolve()
    )
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace adapter bundle: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    identity, contract, evidence = _training_contract(source)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.building-", dir=target.parent
    ) as temporary:
        building = Path(temporary)
        _copy_flat_directory(source / "adapter", building / "adapter")
        _copy_flat_directory(source / "tokenizer", building / "tokenizer")
        for filename in _REPORT_FILES:
            _copy_regular_file(source / filename, building / filename)
        files = _describe_files(building)
        manifest: dict[str, object] = {
            "schema_version": SFT_ADAPTER_BUNDLE_VERSION,
            "identity": identity.to_dict(),
            "contract": contract.to_dict(),
            "evidence": evidence,
            "files": files,
        }
        manifest["bundle_fingerprint"] = _manifest_fingerprint(manifest)
        write_strict_json(building / SFT_ADAPTER_BUNDLE_MANIFEST, manifest)
        verify_sft_adapter_bundle(
            building,
            expected_model_id=identity.model_id,
            expected_revision=identity.revision,
        )
        os.replace(building, target)
    return verify_sft_adapter_bundle(
        target,
        expected_model_id=identity.model_id,
        expected_revision=identity.revision,
    )


def verify_sft_adapter_bundle(
    root: Path,
    *,
    expected_model_id: str | None = None,
    expected_revision: str | None = None,
) -> SFTAdapterBundleVerification:
    """Verify every published byte and the identities needed for fresh reload."""

    if root.is_symlink():
        raise ValueError("adapter bundle must not be a symlink")
    bundle = root.resolve()
    if not bundle.is_dir():
        raise ValueError("adapter bundle must be a regular directory")
    manifest_path = bundle / SFT_ADAPTER_BUNDLE_MANIFEST
    manifest = _load_json_file(
        manifest_path, "adapter bundle manifest", max_bytes=_MAX_MANIFEST_BYTES
    )
    _exact_fields(manifest, _MANIFEST_FIELDS, "adapter bundle manifest")
    if manifest.get("schema_version") != SFT_ADAPTER_BUNDLE_VERSION:
        raise ValueError("unsupported adapter bundle schema_version")
    fingerprint = _digest(
        manifest.get("bundle_fingerprint"), "bundle_fingerprint"
    )
    unsigned = dict(manifest)
    del unsigned["bundle_fingerprint"]
    if fingerprint != _manifest_fingerprint(unsigned):
        raise ValueError("adapter bundle fingerprint mismatch")

    identity = _parse_identity(manifest.get("identity"))
    if expected_model_id is not None and identity.model_id != expected_model_id:
        raise ValueError("adapter bundle model_id differs from expected model")
    if expected_revision is not None and identity.revision != expected_revision:
        raise ValueError("adapter bundle revision differs from expected base revision")
    contract = _parse_contract(manifest.get("contract"))
    evidence = _parse_evidence(manifest.get("evidence"))
    files = _parse_file_descriptors(manifest.get("files"))
    actual_files = _describe_files(bundle)
    if files != actual_files:
        raise ValueError("adapter bundle file set, size, or digest mismatch")

    _verify_adapter_config(bundle, identity=identity, contract=contract)
    tensor_count = _verify_adapter_weights(bundle, contract=contract)
    _verify_tokenizer(bundle)
    _verify_training_reports(
        bundle, identity=identity, contract=contract, evidence=evidence
    )
    return SFTAdapterBundleVerification(
        identity=identity,
        contract=contract,
        bundle_fingerprint=fingerprint,
        file_count=len(files),
        total_file_bytes=sum(cast(int, item["bytes"]) for item in files),
        adapter_tensor_count=tensor_count,
        optimizer_step_count=cast(int, evidence["optimizer_step_count"]),
    )


def _training_contract(
    source: Path,
) -> tuple[SFTAdapterBundleIdentity, SFTAdapterBundleContract, dict[str, object]]:
    report = _load_json_file(source / "sft-training-run.json", "training report")
    model = _object(report.get("model"), "training report model")
    identity = SFTAdapterBundleIdentity(
        model_id=_nonempty_string(model.get("model_id"), "training report model_id"),
        revision=_nonempty_string(model.get("revision"), "training report revision"),
    )
    training = _object(report.get("training"), "training report training")
    report_version = _nonempty_string(
        report.get("report_version"), "training report version"
    )
    if report_version == "about-llm.sft-training-run.v1":
        quantized_base = False
    elif report_version == "about-llm.qlora-training-run.v1":
        quantized_base = True
    else:
        raise ValueError("unsupported training report version for adapter bundle")
    contract = SFTAdapterBundleContract(
        target_modules=_target_modules(training.get("target_modules")),
        rank=_positive_integer(training.get("rank"), "training rank"),
        alpha=_positive_integer(training.get("alpha"), "training alpha"),
        quantized_base=quantized_base,
    )
    outcome = _object(report.get("outcome"), "training report outcome")
    data = _object(report.get("data"), "training report data")
    evidence = {
        "training_report_version": report_version,
        "readiness_manifest_fingerprint": _digest(
            data.get("readiness_manifest_fingerprint"),
            "training readiness fingerprint",
        ),
        "assistant_mask_manifest_fingerprint": _digest(
            data.get("assistant_mask_manifest_fingerprint"),
            "assistant mask fingerprint",
        ),
        "final_labels_fingerprint": _digest(
            data.get("final_labels_fingerprint"), "final labels fingerprint"
        ),
        "optimizer_step_count": _positive_integer(
            outcome.get("optimizer_step_count"), "optimizer_step_count"
        ),
    }
    return identity, contract, evidence


def _verify_adapter_config(
    root: Path,
    *,
    identity: SFTAdapterBundleIdentity,
    contract: SFTAdapterBundleContract,
) -> None:
    config = _load_json_file(
        root / "adapter" / "adapter_config.json", "adapter config"
    )
    expected = {
        "base_model_name_or_path": identity.model_id,
        "revision": identity.revision,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": contract.rank,
        "lora_alpha": contract.alpha,
        "bias": "none",
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"adapter config {name} differs from bundle contract")
    if set(_target_modules(config.get("target_modules"))) != set(
        contract.target_modules
    ):
        raise ValueError("adapter config target_modules differ from bundle contract")
    if config.get("modules_to_save") not in (None, []):
        raise ValueError("adapter bundle does not support extra modules_to_save tensors")


def _verify_adapter_weights(
    root: Path, *, contract: SFTAdapterBundleContract
) -> int:
    path = root / "adapter" / "adapter_model.safetensors"
    if path.is_symlink() or not path.is_file():
        raise ValueError("adapter bundle is missing adapter_model.safetensors")
    try:
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("adapter bundle verification requires safetensors") from error
    try:
        tensors = load_file(path)
    except Exception as error:
        raise ValueError("adapter weights must be valid safetensors") from error
    if not tensors:
        raise ValueError("adapter weights must contain tensors")
    for name, tensor in tensors.items():
        if not any(
            marker in name
            for module in contract.target_modules
            for marker in (f".{module}.lora_A.", f".{module}.lora_B.")
        ):
            raise ValueError(f"adapter contains an unexpected tensor: {name!r}")
        if not bool(np.isfinite(tensor).all()):
            raise ValueError(f"adapter tensor {name!r} contains non-finite values")
    for module in contract.target_modules:
        a_marker = f".{module}.lora_A."
        b_marker = f".{module}.lora_B."
        a_tensors = [tensor for name, tensor in tensors.items() if a_marker in name]
        b_tensors = [tensor for name, tensor in tensors.items() if b_marker in name]
        if not a_tensors:
            raise ValueError(f"adapter weights do not contain LoRA A for {module!r}")
        if not b_tensors:
            raise ValueError(f"adapter weights do not contain LoRA B for {module!r}")
        if not any(bool(np.count_nonzero(tensor)) for tensor in a_tensors):
            raise ValueError(f"adapter LoRA A tensors are all zero for {module!r}")
        if not any(bool(np.count_nonzero(tensor)) for tensor in b_tensors):
            raise ValueError(f"adapter LoRA B tensors are all zero for {module!r}")
    return len(tensors)


def _verify_tokenizer(root: Path) -> None:
    tokenizer_root = root / "tokenizer"
    config = _load_json_file(tokenizer_root / "tokenizer_config.json", "tokenizer config")
    template_path = tokenizer_root / "chat_template.jinja"
    has_template = isinstance(config.get("chat_template"), str) and bool(
        cast(str, config["chat_template"]).strip()
    )
    if not has_template:
        has_template = (
            template_path.is_file()
            and not template_path.is_symlink()
            and bool(template_path.read_text(encoding="utf-8").strip())
        )
    if not has_template:
        raise ValueError("tokenizer bundle is missing a non-empty chat template")
    names = {item.name for item in tokenizer_root.iterdir() if item.is_file()}
    if not names.intersection(_TOKENIZER_PAYLOAD_NAMES):
        raise ValueError("tokenizer bundle is missing a vocabulary or tokenizer payload")


def _verify_training_reports(
    root: Path,
    *,
    identity: SFTAdapterBundleIdentity,
    contract: SFTAdapterBundleContract,
    evidence: dict[str, object],
) -> None:
    report = _load_json_file(root / "sft-training-run.json", "training report")
    if report.get("status") != "completed":
        raise ValueError("only a completed training run can be published")
    report_identity, report_contract, report_evidence = _training_contract(root)
    if report_identity != identity or report_contract != contract:
        raise ValueError("training report identity or contract differs from bundle")
    if report_evidence != evidence:
        raise ValueError("training report evidence differs from bundle")

    readiness = _load_json_file(
        root / "sft-training-readiness.json", "training readiness"
    )
    mask = _load_json_file(root / "sft-template-mask-audit.json", "mask audit")
    labels = _load_json_file(root / "sft-final-label-audit.json", "label audit")
    data_audit = _load_json_file(root / "sft-data-audit.json", "data audit")
    for label, record in (
        ("training readiness", readiness),
        ("mask audit", mask),
        ("label audit", labels),
        ("data audit", data_audit),
    ):
        if record.get("gate_passed") is not True:
            raise ValueError(f"{label} did not pass")
    if readiness.get("manifest_fingerprint") != evidence[
        "readiness_manifest_fingerprint"
    ]:
        raise ValueError("readiness fingerprint differs from training report")
    if mask.get("manifest_fingerprint") != evidence[
        "assistant_mask_manifest_fingerprint"
    ]:
        raise ValueError("assistant mask fingerprint differs from training report")
    if labels.get("labels_fingerprint") != evidence["final_labels_fingerprint"]:
        raise ValueError("final labels fingerprint differs from training report")


def _copy_flat_directory(source: Path, target: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"bundle source directory is missing or unsafe: {source}")
    target.mkdir()
    items = list(source.iterdir())
    if not items:
        raise ValueError(f"bundle source directory is empty: {source}")
    for item in items:
        if item.is_symlink() or not item.is_file():
            raise ValueError(f"bundle source contains non-regular entry: {item}")
        _copy_regular_file(item, target / item.name)


def _copy_regular_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"bundle source file is missing or unsafe: {source}")
    if source.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError(f"bundle source file is too large: {source}")
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _describe_files(root: Path) -> list[dict[str, object]]:
    descriptors: list[dict[str, object]] = []
    total = 0
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            path = current_path / directory
            if path.is_symlink():
                raise ValueError(f"adapter bundle contains a symlink: {path}")
            relative_directory = path.relative_to(root).as_posix()
            if relative_directory not in {"adapter", "tokenizer"}:
                raise ValueError(
                    "adapter bundle contains an unsupported directory: "
                    f"{relative_directory}"
                )
        for filename in filenames:
            path = current_path / filename
            relative = path.relative_to(root).as_posix()
            if relative == SFT_ADAPTER_BUNDLE_MANIFEST:
                continue
            _safe_relative_path(relative)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"adapter bundle contains a non-regular file: {path}")
            if path.suffix.lower() in {".bin", ".pt", ".pth", ".pkl", ".pickle"}:
                raise ValueError(f"adapter bundle contains a pickle-capable file: {relative}")
            size = path.stat().st_size
            if size > _MAX_FILE_BYTES:
                raise ValueError(f"adapter bundle file is too large: {relative}")
            total += size
            if total > _MAX_TOTAL_BYTES:
                raise ValueError("adapter bundle exceeds the total byte limit")
            descriptors.append(
                {"path": relative, "bytes": size, "sha256": _sha256_file(path)}
            )
            if len(descriptors) > _MAX_FILES:
                raise ValueError("adapter bundle contains too many files")
    descriptors.sort(key=lambda item: cast(str, item["path"]))
    return descriptors


def _parse_file_descriptors(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value or len(value) > _MAX_FILES:
        raise ValueError("adapter bundle files must be a bounded non-empty array")
    files: list[dict[str, object]] = []
    total = 0
    for index, item in enumerate(value):
        record = _object(item, f"files[{index}]")
        _exact_fields(record, _FILE_FIELDS, f"files[{index}]")
        path = _safe_relative_path(record.get("path"))
        size = _positive_integer(record.get("bytes"), f"files[{index}].bytes")
        if size > _MAX_FILE_BYTES:
            raise ValueError(f"files[{index}] exceeds the per-file byte limit")
        total += size
        if total > _MAX_TOTAL_BYTES:
            raise ValueError("adapter bundle manifest exceeds the total byte limit")
        files.append(
            {
                "path": path,
                "bytes": size,
                "sha256": _digest(record.get("sha256"), f"files[{index}].sha256"),
            }
        )
    paths = [cast(str, item["path"]) for item in files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("adapter bundle file descriptors must be unique and sorted")
    return files


def _parse_identity(value: object) -> SFTAdapterBundleIdentity:
    record = _object(value, "identity")
    _exact_fields(record, _IDENTITY_FIELDS, "identity")
    return SFTAdapterBundleIdentity(
        model_id=_nonempty_string(record.get("model_id"), "identity.model_id"),
        revision=_nonempty_string(record.get("revision"), "identity.revision"),
    )


def _parse_contract(value: object) -> SFTAdapterBundleContract:
    record = _object(value, "contract")
    _exact_fields(record, _CONTRACT_FIELDS, "contract")
    if (
        record.get("adapter_directory") != "adapter"
        or record.get("tokenizer_directory") != "tokenizer"
        or record.get("adapter_type") != "LORA"
        or record.get("task_type") != "CAUSAL_LM"
    ):
        raise ValueError("adapter bundle directory or PEFT contract is unsupported")
    quantized = record.get("quantized_base")
    if not isinstance(quantized, bool):
        raise ValueError("contract.quantized_base must be boolean")
    return SFTAdapterBundleContract(
        target_modules=_target_modules(record.get("target_modules")),
        rank=_positive_integer(record.get("rank"), "contract.rank"),
        alpha=_positive_integer(record.get("alpha"), "contract.alpha"),
        quantized_base=quantized,
    )


def _parse_evidence(value: object) -> dict[str, object]:
    record = _object(value, "evidence")
    _exact_fields(record, _EVIDENCE_FIELDS, "evidence")
    return {
        "training_report_version": _nonempty_string(
            record.get("training_report_version"), "evidence.training_report_version"
        ),
        "readiness_manifest_fingerprint": _digest(
            record.get("readiness_manifest_fingerprint"),
            "evidence.readiness_manifest_fingerprint",
        ),
        "assistant_mask_manifest_fingerprint": _digest(
            record.get("assistant_mask_manifest_fingerprint"),
            "evidence.assistant_mask_manifest_fingerprint",
        ),
        "final_labels_fingerprint": _digest(
            record.get("final_labels_fingerprint"),
            "evidence.final_labels_fingerprint",
        ),
        "optimizer_step_count": _positive_integer(
            record.get("optimizer_step_count"), "evidence.optimizer_step_count"
        ),
    }


def _load_json_file(
    path: Path, label: str, *, max_bytes: int = _MAX_JSON_BYTES
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{label} exceeds the configured byte limit")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 strict JSON") from error
    return _object(value, label)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _manifest_fingerprint(record: dict[str, object]) -> str:
    return "sha256:" + artifact_fingerprint(record)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_relative_path(value: object) -> str:
    path = _nonempty_string(value, "file path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in path:
        raise ValueError("adapter bundle file path must be safe and relative")
    if pure.parts[0] not in {"adapter", "tokenizer", *_REPORT_FILES}:
        raise ValueError(f"adapter bundle contains unsupported path: {path}")
    if pure.parts[0] in _REPORT_FILES and len(pure.parts) != 1:
        raise ValueError(f"adapter bundle report path must be top-level: {path}")
    return path


def _target_modules(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("target_modules must be a non-empty array")
    modules = tuple(_nonempty_string(item, "target module") for item in value)
    if len(modules) != len(set(modules)):
        raise ValueError("target_modules must be unique")
    return modules


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _digest(value: object, label: str) -> str:
    digest = _nonempty_string(value, label)
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(digest[7:], 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest") from error
    if digest != digest.lower():
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _exact_fields(record: dict[str, Any], expected: set[str], label: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{label} fields differ from the supported schema")
