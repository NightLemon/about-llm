"""Versioned JSON Schema contracts shared by model prompts and tool execution."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, cast

from about_llm.agents.model_planner import PlannerToolContract
from about_llm.agents.policy import ResourceRef
from about_llm.agents.runtime import SideEffect, Tool, freeze_json_value
from about_llm.llmops import artifact_fingerprint, canonical_json_bytes

DRAFT_2020_12_URI = "https://json-schema.org/draft/2020-12/schema"
JSON_SCHEMA_TOOL_PROFILE = "about-llm.closed-tool-json-schema.v1"


class ToolArgumentValidationError(ValueError):
    """Stable, value-redacted argument failure safe to expose to loop control."""

    def __init__(
        self,
        code: str,
        *,
        schema_revision: str,
        keyword: str | None = None,
        instance_pointer: str = "",
        schema_pointer: str = "",
    ) -> None:
        self.code = code
        self.schema_revision = schema_revision
        self.keyword = keyword
        self.instance_pointer = instance_pointer
        self.schema_pointer = schema_pointer
        fields = [f"code={code}", f"schema_revision={schema_revision}"]
        if keyword is not None:
            fields.append(f"keyword={keyword}")
        if instance_pointer:
            fields.append(f"instance_path={instance_pointer}")
        if schema_pointer:
            fields.append(f"schema_path={schema_pointer}")
        super().__init__("tool arguments rejected (" + ", ".join(fields) + ")")


@dataclass(frozen=True)
class JSONSchemaToolContract:
    """One immutable schema used for both planner disclosure and runtime validation.

    This profile requires an explicit Draft 2020-12 closed root object, rejects
    external references and identifiers, and makes format enforcement explicit.
    The jsonschema package performs the actual standard validation.
    """

    name: str
    description: str
    schema_revision: str
    arguments_schema: Mapping[str, Any]
    enforce_formats: bool = False
    max_schema_bytes: int = 64_000
    max_instance_bytes: int = 64_000
    _validator: Any = field(init=False, repr=False, compare=False)
    _schema_fingerprint: str = field(init=False, repr=False)
    _validator_revision: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("name", self.name),
            ("description", self.description),
            ("schema_revision", self.schema_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"tool schema {label} cannot be empty")
        if not isinstance(self.enforce_formats, bool):
            raise ValueError("enforce_formats must be boolean")
        for label, limit in (
            ("max_schema_bytes", self.max_schema_bytes),
            ("max_instance_bytes", self.max_instance_bytes),
        ):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError(f"{label} must be a positive integer")

        try:
            schema_bytes = canonical_json_bytes(self.arguments_schema)
        except (TypeError, ValueError) as error:
            raise ValueError(f"arguments_schema must be strict JSON: {error}") from error
        if len(schema_bytes) > self.max_schema_bytes:
            raise ValueError("arguments_schema exceeds max_schema_bytes")
        schema = json.loads(schema_bytes)
        if not isinstance(schema, dict):
            raise ValueError("arguments_schema must be a JSON object")
        if schema.get("$schema") != DRAFT_2020_12_URI:
            raise ValueError("arguments_schema must explicitly select Draft 2020-12")
        if schema.get("type") != "object":
            raise ValueError("tool arguments_schema root type must be object")
        if not (
            schema.get("additionalProperties") is False
            or schema.get("unevaluatedProperties") is False
        ):
            raise ValueError("tool arguments_schema root must reject unevaluated fields")
        format_names = _audit_schema_references_and_formats(schema)

        try:
            import jsonschema  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError(
                "install the 'agents' extra for runtime JSON Schema validation"
            ) from error
        validator_type = jsonschema.validators.validator_for(schema)
        if validator_type is not jsonschema.Draft202012Validator:
            raise ValueError("arguments_schema did not resolve to Draft 2020-12")
        try:
            validator_type.check_schema(schema)
        except jsonschema.SchemaError as error:
            raise ValueError("arguments_schema is invalid Draft 2020-12") from error

        checker = None
        if self.enforce_formats:
            checker = jsonschema.FormatChecker()
            unknown_formats = sorted(format_names - set(checker.checkers))
            if unknown_formats:
                raise ValueError(
                    f"arguments_schema uses unsupported format(s): {unknown_formats}"
                )
        jsonschema_version = version("jsonschema")
        validator_revision = (
            f"{JSON_SCHEMA_TOOL_PROFILE}+jsonschema-{jsonschema_version}"
            f"+formats-{'on' if self.enforce_formats else 'annotation'}"
        )
        frozen_schema = cast(Mapping[str, Any], freeze_json_value(schema))
        object.__setattr__(self, "arguments_schema", frozen_schema)
        object.__setattr__(
            self,
            "_validator",
            validator_type(schema, format_checker=checker),
        )
        object.__setattr__(self, "_validator_revision", validator_revision)
        object.__setattr__(
            self,
            "_schema_fingerprint",
            "sha256:"
            + artifact_fingerprint(
                {
                    "profile": JSON_SCHEMA_TOOL_PROFILE,
                    "schema_revision": self.schema_revision,
                    "validator_revision": validator_revision,
                    "schema": schema,
                    "max_instance_bytes": self.max_instance_bytes,
                }
            ),
        )

    @property
    def validator_revision(self) -> str:
        return self._validator_revision

    @property
    def schema_fingerprint(self) -> str:
        return self._schema_fingerprint

    def validate(self, arguments: Mapping[str, Any]) -> None:
        """Validate a detached JSON object without exposing rejected values."""
        try:
            instance_bytes = canonical_json_bytes(arguments)
        except (TypeError, ValueError) as error:
            raise ToolArgumentValidationError(
                "invalid_json_domain",
                schema_revision=self.schema_revision,
            ) from error
        if len(instance_bytes) > self.max_instance_bytes:
            raise ToolArgumentValidationError(
                "instance_too_large",
                schema_revision=self.schema_revision,
            )
        instance = json.loads(instance_bytes)
        import jsonschema

        try:
            self._validator.validate(instance)
        except jsonschema.ValidationError as error:
            raise ToolArgumentValidationError(
                "schema_violation",
                schema_revision=self.schema_revision,
                keyword=(
                    str(error.validator)
                    if getattr(error, "validator", None) is not None
                    else None
                ),
                instance_pointer=_json_pointer(tuple(error.absolute_path)),
                schema_pointer=_json_pointer(tuple(error.absolute_schema_path)),
            ) from error

    def planner_contract(self) -> PlannerToolContract:
        return PlannerToolContract(
            name=self.name,
            description=self.description,
            schema_revision=self.schema_revision,
            validator_revision=self.validator_revision,
            arguments_schema=self.arguments_schema,
        )

    def build_tool(
        self,
        *,
        tool_version: str,
        side_effect: SideEffect,
        handler: Callable[[Mapping[str, Any]], Any],
        required_capability: str,
        resolve_resource: Callable[[Mapping[str, Any]], ResourceRef],
    ) -> Tool:
        """Build a runtime Tool whose validator is this exact contract."""
        return Tool(
            name=self.name,
            version=tool_version,
            description=self.description,
            side_effect=side_effect,
            validate=self.validate,
            handler=handler,
            required_capability=required_capability,
            resolve_resource=resolve_resource,
        )


def _audit_schema_references_and_formats(value: Any) -> frozenset[str]:
    formats: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if "$id" in node:
                raise ValueError("tool arguments_schema must not contain $id")
            for keyword in ("$ref", "$dynamicRef"):
                reference = node.get(keyword)
                if reference is not None and (
                    not isinstance(reference, str) or not reference.startswith("#")
                ):
                    raise ValueError(
                        f"tool arguments_schema {keyword} must be a local fragment"
                    )
            format_name = node.get("format")
            if format_name is not None:
                if not isinstance(format_name, str) or not format_name:
                    raise ValueError("tool arguments_schema format must be a string")
                formats.add(format_name)
            for item in node.values():
                visit(item)
        elif isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            for item in node:
                visit(item)

    visit(value)
    return frozenset(formats)


def _json_pointer(parts: tuple[Any, ...]) -> str:
    if not parts:
        return ""
    escaped = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)
