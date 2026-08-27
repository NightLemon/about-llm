"""把 LangChain 与 LlamaIndex 当作 Safe Agent 工具提案的传输层。

两个框架负责公开 tool schema、接收参数并形成 proposal；canonical AgentRuntime 仍独占资源解析、
权限、幂等和 handler 执行。实验比较允许、重放、跨租户、类型错误和未知工具五条路径，
不调用模型、网络或远端工具。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Literal, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from llama_index.core.tools import FunctionTool, ToolSelection
from pydantic import BaseModel, ConfigDict

from about_llm.agents import (
    DRAFT_2020_12_URI,
    AgentRuntime,
    CapabilityPolicy,
    ExecutionContext,
    ExecutionOutcome,
    JSONSchemaToolContract,
    ResourceRef,
    SideEffect,
    ToolCall,
    ToolRegistry,
)
from about_llm.llmops import canonical_json_bytes

TOOL_NAME = "fixture_lookup"
TOOL_DESCRIPTION = "Read one tenant-scoped fixture value."
TOOL_SCHEMA_REVISION = "fixture-lookup-arguments@v1"
TOOL_VERSION = "fixture-lookup@v1"
POLICY_VERSION = "fixture-capability-policy@v1"


class LookupArguments(BaseModel):
    """两个框架与 Runtime 共同使用的唯一参数 schema 来源。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    key: Literal["public", "private"]


@dataclass
class Harness:
    """把一套独立 Runtime、可信上下文和 handler 调用记录放在一起。"""

    runtime: AgentRuntime
    context: ExecutionContext
    handler_calls: list[str]


def _strict_schema() -> dict[str, Any]:
    """从 Pydantic 模型生成 closed JSON Schema，并声明标准版本。"""

    schema = LookupArguments.model_json_schema(mode="validation")
    schema["$schema"] = DRAFT_2020_12_URI
    return schema


def _contract() -> JSONSchemaToolContract:
    """用共享 schema 构造 Runtime 的 canonical 工具 contract。"""

    return JSONSchemaToolContract(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        schema_revision=TOOL_SCHEMA_REVISION,
        arguments_schema=_strict_schema(),
    )


def _resource(arguments: Mapping[str, Any]) -> ResourceRef:
    """把 public/private key 解析到不同租户的可信资源。"""

    key = arguments["key"]
    tenant_id = "tenant-a" if key == "public" else "tenant-b"
    return ResourceRef(tenant_id, "fixture", str(key), "fixture-data@v1")


def _new_harness() -> Harness:
    """创建一套全新 Runtime，避免两个框架共享 cache 或调用计数。"""

    handler_calls: list[str] = []

    def handler(arguments: Mapping[str, Any]) -> dict[str, str]:
        """记录真正执行的 key，并返回固定本地值。"""

        key = cast(str, arguments["key"])
        handler_calls.append(key)
        return {"value": f"fixture:{key}"}

    tool = _contract().build_tool(
        tool_version=TOOL_VERSION,
        side_effect=SideEffect.READ_ONLY,
        handler=handler,
        required_capability="fixture:read",
        resolve_resource=_resource,
    )
    runtime = AgentRuntime(
        ToolRegistry([tool]),
        policy=CapabilityPolicy(POLICY_VERSION),
        max_tool_calls=5,
        clock=lambda: 1_786_723_200.0,
    )
    context = ExecutionContext(
        task_id="framework-adapter-control",
        subject_id="fixture-user",
        tenant_id="tenant-a",
        capabilities=frozenset({"fixture:read"}),
    )
    return Harness(runtime=runtime, context=context, handler_calls=handler_calls)


def _proposal(key: Any) -> dict[str, Any]:
    """生成两个适配器共同交给 Runtime 的 closed envelope。"""

    return {"tool_name": TOOL_NAME, "arguments": {"key": key}}


def _build_langchain_tool() -> StructuredTool:
    """构造只负责产出 proposal artifact 的 LangChain StructuredTool。"""

    def propose(key: Literal["public", "private"]) -> tuple[str, dict[str, Any]]:
        return "proposal_ready", _proposal(key)

    return StructuredTool.from_function(
        func=propose,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        args_schema=LookupArguments,
        response_format="content_and_artifact",
    )


def _build_llamaindex_tool() -> FunctionTool:
    """构造只负责产出 proposal 的 LlamaIndex FunctionTool。"""

    def propose(key: Literal["public", "private"]) -> dict[str, Any]:
        return _proposal(key)

    return FunctionTool.from_defaults(
        fn=propose,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        fn_schema=LookupArguments,
    )


def _langchain_proposal(
    tool: StructuredTool,
    *,
    call_id: str,
    selected_name: str,
    arguments: Mapping[str, Any],
) -> Mapping[str, Any]:
    """让 LangChain 解析一次 ToolCall，并取回 proposal artifact。"""

    # BaseTool.invoke 是执行 API，不是授权 API；调用前先对 allowlist 中的工具名。
    if selected_name != tool.name:
        raise ValueError("LangChain selection does not match the bound tool")
    result = tool.invoke(
        {
            "name": selected_name,
            "args": dict(arguments),
            "id": call_id,
            "type": "tool_call",
        }
    )
    if not isinstance(result, ToolMessage):
        raise TypeError("LangChain full ToolCall did not return ToolMessage")
    if result.tool_call_id != call_id or result.status != "success":
        raise ValueError("LangChain ToolMessage identity or status mismatch")
    if not isinstance(result.artifact, Mapping):
        raise TypeError("LangChain proposal artifact must be an object")
    return result.artifact


def _llamaindex_proposal(
    tool: FunctionTool,
    selection: ToolSelection,
) -> Mapping[str, Any]:
    """让 LlamaIndex 解析 ToolSelection，并取回 proposal。"""

    if selection.tool_name != tool.metadata.get_name():
        raise ValueError("LlamaIndex selection does not match the bound tool")
    result = tool.call(**selection.tool_kwargs)
    if result.tool_name != selection.tool_name:
        raise ValueError("LlamaIndex ToolOutput name mismatch")
    if not isinstance(result.raw_output, Mapping):
        raise TypeError("LlamaIndex proposal raw_output must be an object")
    return result.raw_output


def _execute_proposal(
    harness: Harness,
    *,
    call_id: str,
    proposal: Mapping[str, Any],
) -> ExecutionOutcome:
    """验证框架 envelope 后，将提案交给 canonical Runtime。"""

    # 框架输出也不可信，必须再次检查 closed envelope 与工具名。
    if set(proposal) != {"tool_name", "arguments"}:
        raise ValueError("framework proposal must use the closed canonical envelope")
    if proposal["tool_name"] != TOOL_NAME:
        raise ValueError("framework proposal tool_name mismatch")
    arguments = proposal["arguments"]
    if not isinstance(arguments, Mapping):
        raise TypeError("framework proposal arguments must be an object")
    return harness.runtime.execute(
        ToolCall(call_id, TOOL_NAME, arguments),
        context=harness.context,
    )


def _public_outcome(outcome: ExecutionOutcome) -> dict[str, Any]:
    """提取两个框架可公平比较的 Runtime 结果字段。"""

    value = json.loads(canonical_json_bytes(outcome.value)) if outcome.value is not None else None
    return {
        "status": outcome.status.value,
        "resource_tenant": outcome.resource.tenant_id,
        "value": value,
        "execution_fingerprint": outcome.execution_fingerprint,
    }


def _exception_type(operation: Any) -> str:
    """运行一个预期失败的负例，只公开异常类型。"""

    try:
        operation()
    except Exception as error:  # The report intentionally publishes only the type.
        return type(error).__name__
    raise AssertionError("negative control unexpectedly succeeded")


def run_control() -> dict[str, Any]:
    """让两个框架依次走正常、缓存、拒绝和非法输入路径。"""

    # 两个 harness 独立执行，避免一边的 call ID 缓存影响另一边。
    langchain_tool = _build_langchain_tool()
    llamaindex_tool = _build_llamaindex_tool()
    langchain = _new_harness()
    llamaindex = _new_harness()

    # 正常路径：框架只形成 proposal，Runtime 解析 public 资源后允许 handler。
    lc_allowed_proposal = _langchain_proposal(
        langchain_tool,
        call_id="lc-public-1",
        selected_name=TOOL_NAME,
        arguments={"key": "public"},
    )
    li_allowed_selection = ToolSelection(
        tool_id="li-public-1",
        tool_name=TOOL_NAME,
        tool_kwargs={"key": "public"},
    )
    li_allowed_proposal = _llamaindex_proposal(llamaindex_tool, li_allowed_selection)
    lc_allowed = _execute_proposal(
        langchain, call_id="lc-public-1", proposal=lc_allowed_proposal
    )
    li_allowed = _execute_proposal(
        llamaindex,
        call_id=li_allowed_selection.tool_id,
        proposal=li_allowed_proposal,
    )

    # 同一个 call ID 再执行应命中 Runtime cache，handler 不应第二次运行。
    lc_replayed = _execute_proposal(
        langchain, call_id="lc-public-1", proposal=lc_allowed_proposal
    )
    li_replayed = _execute_proposal(
        llamaindex,
        call_id=li_allowed_selection.tool_id,
        proposal=li_allowed_proposal,
    )

    # private key 被解析到 tenant-b，与可信 context 的 tenant-a 不符，应在 handler 前拒绝。
    lc_denied_proposal = _langchain_proposal(
        langchain_tool,
        call_id="lc-private-1",
        selected_name=TOOL_NAME,
        arguments={"key": "private"},
    )
    li_denied_selection = ToolSelection(
        tool_id="li-private-1",
        tool_name=TOOL_NAME,
        tool_kwargs={"key": "private"},
    )
    li_denied_proposal = _llamaindex_proposal(llamaindex_tool, li_denied_selection)
    lc_denied = _execute_proposal(
        langchain, call_id="lc-private-1", proposal=lc_denied_proposal
    )
    li_denied = _execute_proposal(
        llamaindex,
        call_id=li_denied_selection.tool_id,
        proposal=li_denied_proposal,
    )

    # key=7 类型错误：LangChain 先在框架 schema 拒绝，LlamaIndex direct call 后由 Runtime 拒绝。
    counts_before_invalid = (len(langchain.handler_calls), len(llamaindex.handler_calls))
    lc_invalid_type = _exception_type(
        lambda: _langchain_proposal(
            langchain_tool,
            call_id="lc-invalid-1",
            selected_name=TOOL_NAME,
            arguments={"key": 7},
        )
    )

    li_invalid_selection = ToolSelection(
        tool_id="li-invalid-1",
        tool_name=TOOL_NAME,
        tool_kwargs={"key": 7},
    )
    li_invalid_proposal = _llamaindex_proposal(llamaindex_tool, li_invalid_selection)
    li_invalid_type = _exception_type(
        lambda: _execute_proposal(
            llamaindex,
            call_id=li_invalid_selection.tool_id,
            proposal=li_invalid_proposal,
        )
    )
    counts_after_invalid = (len(langchain.handler_calls), len(llamaindex.handler_calls))

    # 即使拿到一个 tool instance，也必须核对模型选择的名字是否在该 adapter allowlist。
    lc_unknown = _exception_type(
        lambda: _langchain_proposal(
            langchain_tool,
            call_id="lc-unknown-1",
            selected_name="fixture_missing",
            arguments={"key": "public"},
        )
    )
    li_unknown = _exception_type(
        lambda: _llamaindex_proposal(
            llamaindex_tool,
            ToolSelection(
                tool_id="li-unknown-1",
                tool_name="fixture_missing",
                tool_kwargs={"key": "public"},
            ),
        )
    )

    # 最后比较实际暴露的 schema；记录 LlamaIndex 当前版本 projection 丢失 closed-root 的事实。
    model_schema = LookupArguments.model_json_schema(mode="validation")
    langchain_schema = langchain_tool.get_input_schema().model_json_schema()
    llamaindex_model_schema = llamaindex_tool.metadata.fn_schema.model_json_schema()
    llamaindex_projection = llamaindex_tool.metadata.get_parameters_dict()
    assertions = {
        "shared_pydantic_model_schema_matches_langchain": langchain_schema
        == model_schema,
        "shared_pydantic_model_schema_matches_llamaindex_model": (
            llamaindex_model_schema == model_schema
        ),
        "llamaindex_parameters_projection_omits_closed_root_in_this_version": (
            model_schema.get("additionalProperties") is False
            and "additionalProperties" not in llamaindex_projection
        ),
        "authorized_values_match": lc_allowed.value == li_allowed.value,
        "authorized_handlers_each_executed_once": (
            langchain.handler_calls == ["public"]
            and llamaindex.handler_calls == ["public"]
        ),
        "same_call_ids_replayed_from_canonical_cache": (
            lc_replayed.status.value == "cached"
            and li_replayed.status.value == "cached"
        ),
        "cross_tenant_proposals_denied_before_handler": (
            lc_denied.status.value == "policy_denied"
            and li_denied.status.value == "policy_denied"
        ),
        "invalid_types_never_reached_handler": counts_before_invalid
        == counts_after_invalid,
        "langchain_invalid_type_rejected_by_framework_schema": (
            lc_invalid_type == "ValidationError"
        ),
        "llamaindex_direct_call_required_canonical_schema_gate": (
            li_invalid_type == "ToolArgumentValidationError"
        ),
        "unknown_names_rejected_by_adapter_allowlist": (
            lc_unknown == "ValueError" and li_unknown == "ValueError"
        ),
    }
    if not all(assertions.values()):
        failed = [name for name, passed in assertions.items() if not passed]
        raise AssertionError(f"framework adapter control failed: {failed}")

    contract = _contract()
    return {
        "schema_version": "about-llm.agent-framework-tool-adapter-control.v1",
        "framework_versions": {
            "langchain": version("langchain"),
            "langchain_core": version("langchain-core"),
            "llama_index_core": version("llama-index-core"),
            "pydantic": version("pydantic"),
            "jsonschema": version("jsonschema"),
        },
        "contract": {
            "tool_name": TOOL_NAME,
            "schema_revision": TOOL_SCHEMA_REVISION,
            "validator_revision": contract.validator_revision,
            "schema_fingerprint": contract.schema_fingerprint,
            "model_visible_fields": sorted(model_schema["properties"]),
            "trusted_context_model_visible": False,
        },
        "cases": {
            "authorized": {
                "langchain": _public_outcome(lc_allowed),
                "llamaindex": _public_outcome(li_allowed),
            },
            "same_call_id_replay": {
                "langchain": _public_outcome(lc_replayed),
                "llamaindex": _public_outcome(li_replayed),
            },
            "cross_tenant": {
                "langchain": _public_outcome(lc_denied),
                "llamaindex": _public_outcome(li_denied),
            },
            "invalid_type": {
                "langchain_rejection": lc_invalid_type,
                "llamaindex_canonical_rejection": li_invalid_type,
            },
            "unknown_tool": {
                "langchain_adapter_rejection": lc_unknown,
                "llamaindex_adapter_rejection": li_unknown,
            },
        },
        "assertions": assertions,
        "scope": {
            "real_langchain_structured_tool_executed": True,
            "real_llamaindex_function_tool_executed": True,
            "canonical_schema_policy_resource_and_idempotency_executed": True,
            "model_or_provider_executed": False,
            "langgraph_or_llamaindex_agent_loop_executed": False,
            "network_remote_tool_or_external_side_effect_executed": False,
            "framework_default_authorization_or_production_safety_proved": False,
        },
    }


def main() -> int:
    """运行适配器对照并输出一行 JSON。"""

    print(json.dumps(run_control(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
