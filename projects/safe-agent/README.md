# Safe Agent Runtime

目标：把“模型建议动作”和“系统获权执行”分开。任何模型或 Agent 框架都只能产生 ToolCall；执行内核负责 schema、权限、审批、预算和幂等。

## 已实现的不变量

- 工具注册表拒绝重名和未知工具；
- 参数在审批前校验，避免让用户批准模糊或非法动作；
- 未配置 policy 默认拒绝；capability 缺失、policy indeterminate 和 server-resolved 跨 tenant 资源不进入 handler；
- cache replay 每次重新授权，撤权后不返回旧 payload；
- 可逆/不可逆工具必须提供绑定 execution identity 且未过期的 `ApprovalGrant`；
- proposal fingerprint 标识 tool + arguments；ledger execution fingerprint 还绑定主体、资源/tool/policy version；
- 相同 call_id 与相同 execution identity 只执行一次；换参数、主体或版本触发冲突，而不是覆盖；
- handler attempt 总数有硬预算；proposal/step 预算由上层 loop 另行限制；
- handler 错误转为失败结果，不伪装成功；
- handler 结果经严格 JSON round-trip 脱离原对象并递归冻结；非 JSON 结果失败且 claim 保持 pending；
- ToolCall 严格拒绝 NaN/Infinity、非字符串 object key 与非 JSON 对象，CLI artifact 还拒绝重复 key/未知字段；构造时生成脱离 caller 的递归只读快照；
- proposal fingerprint 是 tool name + 参数 canonical JSON 的 `sha256:` digest；ledger 只写 execution hash，不直接写参数明文；
- handler 一旦获得 claim 就消耗预算，超时/失败不能绕过硬上限；
- 可列出超时 pending，人工确认外部成功、标记放弃或已补偿；
- SQLite completion/reconciliation value 同样使用严格 canonical JSON，拒绝 NaN、非字符串 object key 和不透明 Python 对象；
- reconciliation 保留审计事件，放弃/补偿后的重试必须使用新审批的新 call_id。

## 可运行故障/恢复实验

`scenario.example.jsonl` 只注册三个内置离线工具，不执行网络、邮件或真实写操作。它依次演示只读调用、缺少审批、typed grant 后执行、相同 call id 缓存复用、撤权后 cache replay 拒绝、跨 tenant 拒绝，以及“handler 超时、外部状态未知”的 pending 状态。fixture 中的 context 代表可信控制面输入，不能在生产中让模型填写。

为每次实验使用一个新的 ledger 路径，保留旧数据库用于审计：

~~~powershell
python -m about_llm.agents.cli run `
  --scenario projects/safe-agent/scenario.example.jsonl `
  --ledger artifacts/agent/scenario-001.db `
  --max-tool-calls 5
~~~

输出明确包含 `simulated_offline: true`、每一步的 context/resource、policy reason、proposal/execution fingerprint、unsigned fixture approval、`handler_attempted`、status、pending 状态和期望不匹配。`simulated_effect_applied` 只描述这个确定性的离线 handler，不能冒充真实 provider 状态。样例中 `uncertain-1` 会保持 pending；查询待调查调用：

~~~powershell
python -m about_llm.agents.cli pending `
  --ledger artifacts/agent/scenario-001.db `
  --older-than-seconds 0
~~~

真实系统中，operator 必须先查 provider audit log、业务数据库或 outbox。只有确认结果后才能选择一种 resolution。下面命令仅适用于本离线 fixture，因为已知它没有真实外部副作用：

~~~powershell
python -m about_llm.agents.cli resolve `
  --ledger artifacts/agent/scenario-001.db `
  --call-id uncertain-1 `
  --resolution abandoned `
  --note "offline fixture verified: no external operation exists"

python -m about_llm.agents.cli inspect `
  --ledger artifacts/agent/scenario-001.db `
  --call-id uncertain-1
~~~

`external` resolution 必须提供 `--value-json`；`compensated` 表示副作用发生后已完成逆向操作。三种 resolution 都保留原 call id，旧调用不能重新执行。安装仓库后也可使用 `about-llm-agent`。

## Typed planner loop

`loop.example.jsonl` 用无网络 `ScriptedPlanner` 回归五种控制流：tool observation 后由 exact verifier 确认完成、连续重复 cached action、`A/B/A/B` cycle、不同 action 的连续同类 policy error，以及不可逆工具的 approval pause。

~~~powershell
python -m about_llm.agents.cli loop `
  --cases projects/safe-agent/loop.example.jsonl
~~~

loop 同时限制 decision step、模型 token、cost unit、monotonic wall time、重复 action 和重复 error；只有 verifier `PASSED` 才输出 `completed=true`。token/cost 是 JSONL 中 supplied fixture usage，clock 是固定本地值，exact verifier 只核对答案与已完成/cached evidence call id：三者都不冒充 provider usage、真实账单、线上延迟或开放任务语义判断。动作检测只覆盖连续相同 fingerprint 和最近四步 `A/B/A/B`，不能证明发现任意长周期或语义等价循环。

`needs_approval` 会给出 call id、execution fingerprint 和严格 JSON checkpoint，并保证 handler 尚未调用。checkpoint 保存原预算、累计 usage/handler counter、历史 event/action 和 pending decision；resume 先重新授权并执行原 decision，不重复 planner token/cost。

使用全新路径运行跨进程式离线恢复；checkpoint 以 exclusive create 写入，已有文件不会覆盖：

~~~powershell
python -m about_llm.agents.cli pause-loop `
  --cases projects/safe-agent/loop.example.jsonl `
  --case-id approval-pause `
  --ledger artifacts/agent/loop-001.db `
  --checkpoint artifacts/agent/loop-001.checkpoint.json

python -m about_llm.agents.cli resume-loop `
  --cases projects/safe-agent/loop.example.jsonl `
  --case-id approval-pause `
  --ledger artifacts/agent/loop-001.db `
  --checkpoint artifacts/agent/loop-001.checkpoint.json
~~~

resume CLI 构造的是 `simulated_unsigned_approval`，工具也是本地 simulated send。checkpoint fingerprint 不是签名；文件与 SQLite ledger 不原子；approval 等待 downtime 不计入 active wall time；没有并发 lease、一次性 grant store、加密/retention 或 provider session 恢复。因此这里证明的是确定性 restart 控制流，不是生产 durable workflow。

## Strict JSON model planner boundary

`StrictJSONModelPlanner` 补上“模型文本到 typed proposal”的边界。发送给 transport 的 request fingerprint 绑定 system prompt、prompt revision、task id、剩余 step/token/cost/time、tool catalog/schema revision/validator revision、最近事件的完整 identity/value、输出上限和预期 model revision。工具 observation 会进入最近事件，但 system prompt 明确把它标为 untrusted data；这只是 defense in depth，真正授权仍由可信 `ExecutionContext`、server-resolved resource、policy、approval 和 runtime validator 决定。

响应必须同时提供 exact model revision、provider request id、input/output token usage、cost unit 和允许的 finish reason。parser 只接受单个 closed-schema JSON object；拒绝 Markdown fence、duplicate key、`NaN/Infinity`、溢出为无穷的 float、未知字段、未知工具、空/重复 evidence id。Provider 报告的 output usage 不能超过 request cap。通过后的 request fingerprint、完整 normalized response fingerprint、raw response 和 typed action 一起生成 decision id；这些无密钥 SHA-256 只做 canonical identity，不认证 provider、来源、安全性或语义正确性。Raw response 可能含敏感数据，生产审计必须另设加密、访问控制和 retention。

`JSONSchemaToolContract` 让 Planner 展示与 runtime validation 从同一份 immutable schema 生成。安装对应 extra：

~~~powershell
python -m pip install -c constraints/ci.txt -e ".[agents]"
~~~

当前安全 profile 明确要求 Draft 2020-12、root `type: object`，并以 `additionalProperties: false` 或 `unevaluatedProperties: false` 闭合根参数；`$ref/$dynamicRef` 只允许 local fragment，拒绝 `$id` 与外部 retrieval。Schema 本身和 instance 都有 UTF-8 canonical byte cap。`format` 默认按标准作为 annotation；只有 `enforce_formats=True` 才用当前 `jsonschema` FormatChecker 执行，且未知 format 在构造 contract 时拒绝。Schema/validator revision、`jsonschema` 精确版本、format mode、schema bytes 和 instance cap 进入 contract identity。校验不做字符串转数字等 coercion、不插入 `default`、不执行资源授权；失败只暴露 keyword 和 JSON Pointer，不回显 rejected value。

下面的 control 使用两条代码内冻结的 recorded provider response，不联网也不调用模型。第一条 JSON 提议只读 `fixture_tool`，标准 JSON Schema、tenant resource resolver 和 exact-capability policy 允许后，handler 返回一段恶意指令文本；第二次 request 把它按不可信 observation 纳入状态，模型 fixture 再提议 finish，最后由独立 exact verifier 核对本地 event 才完成：

~~~powershell
python projects/safe-agent/model_planner_control.py
~~~

报告锁定两次 request/response fingerprint、两个 decision id、Draft/schema/validator identity、62 个 authored fixture tokens、0.03 authored cost units、一次 handler attempt 和最终 `verified answer`。四条 negative control 证明 request/state 漂移不能 replay、Markdown-fenced JSON 被拒绝、模型参数虽通过 JSON parser 仍会被 runtime `const` schema 在 resolver/policy/handler 前拒绝、缺 capability 时合法 proposal 也在 handler 前被 policy 拒绝。这里的 token/cost/provider request id 都是 authored metadata；control 不证明真实 API schema、目标模型能遵循协议、provider usage/账单、网络重试、生产 IAM 或开放任务 verifier。接真实 provider 时，adapter 仍需从原始响应提取精确 revision/usage/finish reason，并保存受保护的原始 receipt；缺字段不能猜测。

手写 `PlannerToolContract` 仍可能与任意 callback validator 漂移；需要强一致时应由 `JSONSchemaToolContract.planner_contract()` 与 `.build_tool()` 同源生成。JSON Schema 只验证声明的 JSON 结构和值约束，不知道 resource 是否存在/归属当前 tenant、调用是否获权、effect 是否安全或 handler 返回是否真实，这些边界不能挪进模型 schema。

## LangChain / LlamaIndex framework tool adapter control

Agent framework 适合承载消息、tool selection、callback 和 orchestration，但不应成为资源归属、权限或副作用状态的事实源。本项目新增一条 **proposal-only adapter** control：框架工具函数只返回 closed proposal envelope，随后由 framework-independent `AgentRuntime` 重新执行 schema、可信资源解析、exact-capability policy、call-id 幂等和 handler gate。

~~~powershell
python -m pip install -c constraints/ci.txt -e ".[agents,langchain,llamaindex]"
python projects/safe-agent/framework_tool_adapter_control.py
python -m pytest tests/test_agent_framework_tool_adapters.py -q
~~~

当前固定环境实际执行：

- `langchain==1.3.14`、`langchain-core==1.5.3` 的 `StructuredTool.from_function()` 与 full `ToolCall` `invoke()`；
- `llama-index-core==0.14.23` 的 `FunctionTool.from_defaults()`、`ToolSelection` 与 `FunctionTool.call()`；
- `pydantic==2.13.4` 的 strict/forbid `LookupArguments`；
- `jsonschema==4.26.0` 的 canonical Draft 2020-12 runtime validator。

两框架共用一份 Pydantic model schema。LangChain 返回带原 `tool_call_id` 的 `ToolMessage.artifact`；LlamaIndex 的 `ToolSelection.tool_id` 则由 adapter 显式带入 canonical `ToolCall`。可信 `subject_id`、`tenant_id`、capabilities、resource resolver 与 policy revision 从未放进 model-visible `key` 参数。

### 固定 case 与预期结果

| Case | LangChain | LlamaIndex | handler |
|---|---|---|---|
| `key=public` | `completed` | `completed` | 各 1 次，value 都是 `fixture:public` |
| 同 call id 重放 | `cached` | `cached` | 不增加 |
| `key=private` 解析到 tenant-b | `policy_denied` | `policy_denied` | 不进入 |
| `key=7` | framework Pydantic `ValidationError` | 先形成 proposal，再被 canonical `ToolArgumentValidationError` 拒绝 | 不进入 |
| 选择 `fixture_missing` | adapter allowlist 拒绝 | adapter allowlist 拒绝 | 不进入 |

这里最值得保留的反例是 `key=7`。在当前 LlamaIndex 版本中，直接调用 `FunctionTool.call()` 会进入 Python function，并不自动按 `fn_schema` 做 Pydantic runtime validation；canonical schema gate 才在 resolver/policy/handler 前拒绝它。当前 `fn_schema.model_json_schema()` 保留 `additionalProperties: false`，但 `ToolMetadata.get_parameters_dict()` 的 planner projection 又省略该 closed-root 关键字。LangChain 当前 direct tool path 的行为不同，但这也只是固定版本的局部观察，不能推成所有 entry point、历史/未来版本或所有 schema feature 都相同。

因此 adapter 必须把三件事分开：

1. **disclosure**：模型看见什么工具名、描述和参数 schema；
2. **proposal transport**：框架怎样携带 tool name、arguments 与 call/tool id；
3. **effect authorization**：可信控制面怎样解析资源、授权、审批、claim、执行并核验 effect。

Schema projection equality 只能发现已比较字段的漂移。它不证明模型会给出正确参数，也不证明 Pydantic/框架和标准 JSON Schema 在 coercion、`default`、`format`、union、local `$ref` 或错误路径上语义完全相同。生产 adapter 应把固定 framework version、原始 selection identity、canonicalized arguments、schema/validator revision 和 execution receipt 放入受保护 trace，并为升级运行同一 negative matrix。

### 证据边界

这条 control 真实执行两个框架的 tool API 和本仓库 canonical runtime，但只使用一个本地只读 fixture。它没有执行 LangGraph 或 LlamaIndex Agent loop、模型/provider、callback/tracing backend、async/cancel、streaming、memory、网络、remote tool、审批或真实副作用；也没有证明框架默认 ACL、schema enforcement、幂等、生产安全、性能或跨版本兼容。两框架得到相同 value/fingerprint 是 authored fixture 下 canonical core 的预期不变量，不是质量榜或框架等价证明。

## LangChain / LlamaIndex framework Agent-loop control

上面的 proposal-only control 故意停在 direct tool API。下面的独立 control 才把相同的 canonical `AgentRuntime` 放进真实 framework orchestration：LangChain 通过 `create_agent()` 执行 LangGraph 的 model→tool→model loop，LlamaIndex 通过 `FunctionAgent.run()` 执行 Workflow。两边的 chat model 都是进程内、确定性的 scripted fixture；“真实 loop”描述的是框架控制流，不是 provider 或目标模型。

~~~powershell
python -m pip install -c constraints/ci.txt -e ".[agents,langchain,llamaindex]"
python projects/safe-agent/framework_agent_loop_control.py
python -m pytest tests/test_framework_agent_loop_control.py -q
~~~

固定环境为 `langchain==1.3.14`、`langchain-core==1.5.3`、`langgraph==1.2.10`、`llama-index-core==0.14.23`。模型只看见 `key` 和 allowlisted `fixture_lookup`；subject、tenant、capability、resource resolver、policy、canonical call identity 与 verifier 都在模型外。四组 case 在两个 loop 中执行同一断言：

| Case | canonical runtime | handler | 独立 verifier |
|---|---|---|---|
| authorized | `completed` | 1 次 | 通过 |
| same-id replay | `completed → cached` | 仍为 1 次 | 通过 |
| cross-tenant | `policy_denied` | 0 次 | 拒绝模型声称的成功 |
| unknown-tool | framework error，无 canonical receipt | 0 次 | 拒绝模型声称的成功 |

Verifier 不相信最终 assistant 文本或 framework 的 finished 状态；它只接受本地 canonical receipt 中的 `completed/cached` 状态与预期 value。因此后两组即使 scripted model 最终都输出 `fixture:public`，任务仍不能通过。这个 verifier 只是固定 exact-value fixture，不是开放语义 judge，也没有验证外部 effect。

### 两种 call identity 不能假装相同

LangChain 路径使用 `InjectedToolCallId`，将 framework tool-call ID 直接作为 canonical call ID。模块启用了 postponed annotations；在当前固定版本中，局部工具若只留下字符串 annotation，ID injection 会丢失，所以 control 显式恢复运行时 `Annotated[str, InjectedToolCallId]` 对象并用回归测试锁定。

当前 LlamaIndex `FunctionTool` handler 在这条 `FunctionAgent` 路径中只收到 tool kwargs，不收到 `ToolSelection.tool_id`。Control 因而从可信 fixture 的 case/action 内容派生 canonical ID，而不把幂等键塞进 model-visible 参数；same-id case 的两次相同 action 得到同一 ID。这个 hash 是局部控制实验的确定性 identity，不是通用生产方案：若业务允许“同参但有意执行两次”，必须由可信 orchestrator/ledger 分配不同 action identity，并把 task、subject、resource、tool/schema/policy revision 等纳入冲突检查。

当前 LlamaIndex Workflow 每个 case 还会触发 73 次 Pydantic deprecated-field warnings。Control 将它们捕获到 `dependency_warnings`，并断言没有其他 warning；这是一条升级风险证据，不是“忽略后即可生产”的许可。升级任一框架后应重新执行 call-ID injection、catalog binding、unknown-tool、warning 和 verifier matrix。

### 证据边界

这条 control 确实执行 `create_agent()`/LangGraph 与 `FunctionAgent.run()` 的本地 model→tool→model control flow，以及 canonical schema、资源解析、policy、cache 和独立 verifier。它没有执行真实模型或 provider、网络、remote tool、外部副作用、持久化 checkpointer/resume、streaming、parallel tools、interrupt、cancel/deadline、callback/tracing backend、性能、费用或质量评测；也不证明 framework 默认 authorization、幂等、生产安全或跨版本兼容。CPU/offline scripted 结果只能写成框架控制流回归。

## MCP 2025-11-25 official-SDK memory control

~~~powershell
python -m pip install -c constraints/ci.txt -e ".[agents]"
python projects/safe-agent/mcp_sdk_memory_control.py
python -m pytest tests/test_mcp_sdk_memory.py -q
~~~

这个 control 固定官方 `mcp==1.29.0`，用 AnyIO in-memory object streams 连接官方 `ClientSession` 与 low-level `Server`，真实执行 generated types、initialize/ping、tools capability、`tools/list` 和三次 `tools/call`。`fixture.add` 发布显式 closed input/output schema；成功返回 `structuredContent={"sum":5}`。多余字段由 SDK JSON Schema validation 在应用 handler 前拒绝，handler delta 为 0。

对未列出的 `fixture.missing`，client 没有 cached input schema 可先校验，low-level server SDK 会进入应用 `call_tool` handler；应用 allowlist 再返回 error，handler delta 为 1。这说明官方 SDK 不替代应用的工具名、资源和授权 gate。SDK error content 可能含 validation detail，因此公开 closed report 不发布 raw error，只记录布尔结果、调用计数、scope 与无密钥 fingerprint。

准确证据边界：它没有启动 subprocess，没有执行 OS stdio、TCP/HTTP、SSE、session resumption、TLS/OAuth、远程或跨厂商 server、官方 conformance suite、授权/审批、外部副作用或生产日志。下一节会把相同 SDK fixture 接到真实 stdio；memory control 仍只证明 in-process 路径。

## MCP 2025-11-25 official-SDK stdio control

~~~powershell
python -m pip install -c constraints/ci.txt -e ".[agents]"
python projects/safe-agent/mcp_sdk_stdio_control.py
python -m pytest tests/test_mcp_sdk_stdio.py -q
~~~

官方 `mcp.client.stdio.stdio_client` 启动独立 Python subprocess；子进程以官方 `mcp.server.stdio.stdio_server` 把 low-level `Server` 接到真实 OS stdin/stdout pipe。control 固定 `mcp==1.29.0` 与协议 2025-11-25，执行 initialize/ping、tools discovery，以及成功、schema-invalid、unknown-tool 三次调用。client 配置 UTF-8 strict；当前官方 server 的 stdin 则使用 UTF-8 replacement error handling，不能合写成“两端 strict”。官方 client/session 与 server/generated types 都真实参与，不再只是 memory stream。

server 在协议输入 EOF、`Server.run()` 返回后，以 exclusive create 写临时 canonical receipt。receipt 的 handler 序列精确为 `fixture.add, fixture.missing`，证明 schema-invalid 没进入应用 handler，而 unknown tool 进入应用 allowlist；内部 PID 只用于验证子进程不同于父进程。公开报告不含 PID、receipt path、raw transcript、raw 参数/result content 或 SDK error content，只保留 allowlisted `successful_sum=5`；临时目录退出即删除。

准确证据边界：本 control 同时执行官方 SDK 与真实本地 stdio/subprocess，但没有独立注入 missing LF、duplicate key、invalid UTF-8、byte cap、stdout 污染，没有触发 forced terminate/kill、取消或 deadline；不能把 SDK 源码中存在的分支写成已测。它也没有 HTTP/SSE、TLS/OAuth、远程/跨厂商 server、官方 conformance suite、授权/审批、副作用或生产 supervisor。最小 receipt 与无密钥 fingerprint 不认证进程、来源或真实执行。

## MCP 2025-11-25 official-SDK Streamable HTTP control

~~~powershell
python -m pip install -c constraints/ci.txt -e ".[agents]"
python projects/safe-agent/mcp_sdk_streamable_http_control.py
python -m pytest tests/test_mcp_sdk_streamable_http.py -q
~~~

官方 `streamable_http_client`/`ClientSession` 连接独立 server subprocess；子进程以官方 low-level `Server`、`StreamableHTTPSessionManager` 和 SDK ASGI adapter 运行真实 IPv4 loopback TCP/HTTP。stateful control 固定执行 7 次 POST、1 次 GET 与 1 次 DELETE，其中 initialized notification 是 202，其余 profile 精确为 8 个 200、7 个 SSE response 与 2 个 JSON response。client 观察到 opaque session id，但公开报告不保存它。

临时 receipt 的 handler 序列仍为 `fixture.add, fixture.missing`，并要求 session manager 正常退出、私有 shutdown control 已收到、server exit code 为 0 且 stdout/stderr 为空。报告不发布 PID、session id、token、header、raw HTTP/protocol payload、参数、result 或 SDK error；只保存 method/status/media-type 计数、allowlisted `successful_sum=5` 和无密钥 fingerprints。

准确证据边界：随机 token 只保护测试编排使用的私有 control endpoint，缺失 token 的真实负例为 401；它不是 MCP auth、OAuth、subject/tenant/scope 或业务授权。这个 control 没有执行 MCP endpoint 的 malformed body、Host/Origin failure、resumption、TLS 或 OAuth，也没有网络故障、取消/deadline、远程/跨厂商 server、conformance、审批、副作用、多 worker 或生产 supervisor 证据。先选 loopback port 再由子进程 bind 仍有竞争窗口；receipt/hash 不认证进程、来源或真实执行。

## MCP 2025-11-25 authored strict stdio control

~~~powershell
python projects/safe-agent/mcp_stdio_control.py
python -m pytest tests/test_mcp_stdio.py -q
~~~

这个局部集成不是静态 manifest：client 用当前 Python 启动一个 server 子进程，通过 OS stdin/stdout pipe 交换 LF-delimited UTF-8 JSON-RPC。它依次完成 `initialize` 版本/`tools` capability 协商、无 response 的 `notifications/initialized`、`tools/list`，以及成功、schema-invalid、unknown-tool 三次 `tools/call`。`fixture.add` 发布显式 Draft 2020-12 closed `inputSchema`/`outputSchema`；成功结果同时给 text 和 `structuredContent`，参数 schema 错误返回 `isError: true`，未知工具返回 JSON-RPC `-32602`。

Framing/parser 拒绝 missing LF、embedded raw newline、duplicate key、非有限数、非法 UTF-8、非 object 和 byte cap 外输入。公开 report 不回显原始消息，只对 direction、JSON-RPC version、request id、method、response kind、tool-error flag 与 error code 的 allowlist projection 做 canonical fingerprint；参数与 result content 不在投影内。这不代表生产日志已自动安全，真实 request/result 仍要按 secret/PII 做访问控制、加密和 retention。投影 fingerprint 既不绑定被省略字段，也不认证进程或消息来源。

准确证据边界：它真实执行本地 subprocess 与 stdio，但只实现仓库定义的严格 MCP 2025-11-25 子集；本节 control 没有使用官方 MCP SDK、官方完整 schema/conformance suite、Streamable HTTP、远程网络、认证、授权/人工审批、A2A client/server 或跨厂商互操作。前面的 official-SDK controls 证明的是其他实现/transport，不能把 SDK 身份借给本节自写 parser/server；反过来，本节的畸形 framing 负例也不能借给官方 SDK control。因此不能写成“通过 MCP conformance”“接通任意 MCP server”或“协议连接已安全”。

## MCP 2025-11-25 Streamable HTTP control

~~~powershell
python -m pip install -c constraints/ci.txt -e ".[agents]"
python projects/safe-agent/mcp_streamable_http_control.py
python -m pytest tests/test_mcp_streamable_http.py -q
~~~

父进程启动一个只绑定 `127.0.0.1` 的 server subprocess，client 在真实 TCP/HTTP 上只使用 `/mcp` endpoint。control 验证 POST 的 `Accept: application/json, text/event-stream`、JSON initialize/tools-list、SSE tools-call、GET SSE、空 202 notification 与空 204 DELETE；每个 SSE 都先发带 id 的空-data priming event。初始化返回 opaque visible-ASCII session，后续请求必须同时携带 session 和 `MCP-Protocol-Version: 2025-11-25`，缺失/错版本为 400，DELETE 后重用为 404。

每次 endpoint 请求先检查 Origin allowlist，再检查随机 fixture Bearer header；负例固定拒绝错误 Origin、缺/错 token。另一个并发控制让 `fixture.wait` 保持 in-flight，client 收到 priming event 后显式 POST `notifications/cancelled`；notification 返回空 202，被取消 stream 随即关闭且不发送 JSON-RPC response。公开报告只包含无内容的 transport/status/verifier 投影，不发布 token、session、event id、raw HTTP、参数或 result。

准确证据边界：随机 Bearer 只证明本机 shared-secret header gate，不是 MCP Authorization/OAuth、用户身份、tenant/scope 或业务授权。本节 control 没有官方 MCP SDK、完整 Schema/conformance suite、TLS、远程 server、event store/resumption/redelivery、server-to-client request、跨 stream non-broadcast、审批或跨厂商证据；独立 SDK memory/stdio/HTTP controls 不改变这份自写 HTTP 实现的边界。无密钥投影 fingerprint 既不覆盖省略字段，也不认证执行来源。

## A2A 1.0 official-SDK loopback control

~~~powershell
python -m pip install -c constraints/ci.txt -e ".[agents]"
python projects/safe-agent/a2a_loopback_control.py
python projects/safe-agent/a2a_loopback_control.py --verify-official-schema
python -m pytest tests/test_a2a_loopback.py -q
~~~

父进程启动一个只监听 `127.0.0.1` 的 official-SDK server subprocess；official-SDK client 通过真实 TCP/HTTP 解析 `/.well-known/agent-card.json`，选择 `JSONRPC`/`1.0` interface，依次执行 `SendMessage` 与 `GetTask`。固定 executor 产生 completed task 与结构化 artifact，本地 verifier 再独立验收。raw HTTP 负例要求旧 `kind` 字段返回 `-32602`，`A2A-Version: 9.9` 返回 `-32009`。

默认路径完全本地，只执行 SDK 1.0 generated-proto 与 required-field gate。`--verify-official-schema` 另从冻结的 v1.0.0 URL 下载 `a2a.json`，要求 SHA-256 `6b6560c7…b8d62` 后以 Draft 2020-12 验证 Agent Card、Send Message Request 和 Task；不会隐式跟随 schema 内相对 `$ref` 发起更多请求。公开报告只 fingerprint 协议元数据投影，不发布 raw message、参数、task/context id 或 artifact value。

准确证据边界：本 control 确实使用官方 `a2a-sdk==1.1.2` 的 client、server、Agent Card resolver 与生成类型，并执行真实 IPv4 loopback TCP/HTTP；它不是 A2A TCK 或完整 conformance suite，没有 SSE、HTTP+JSON/REST、gRPC、TLS、认证、签名 Agent Card、授权/人工审批、远程 Agent、跨语言或跨厂商互操作。远端 completed、schema-valid 和无密钥 fingerprint 都不能证明业务正确、身份真实、安全或生产可用。

## Recorded trajectory gate

`trajectory.example.jsonl` 展示独立的 recorded trace artifact。每个 case 固定 environment、policy 与 verifier 版本，并同时保存 proposal/execution fingerprint，把 task verifier、策略判定、handler attempt、外部 effect verifier 和 unresolved pending 分开记录：

~~~powershell
python -m about_llm.agents.cli evaluate `
  --traces projects/safe-agent/trajectory.example.jsonl
~~~

输出为每个比例保留 numerator/denominator，并带逐 case findings；分母为零时 `value` 是 `null`，不会伪报 0%。`max_steps` 限全部 recorded tool proposal，`max_handler_attempts` 只限真正进入 handler 的次数。task success 与安全 guardrail 分开：即使 task verifier 全通过，只要出现 policy-denied handler、policy over-refusal、未审批副作用 attempt、重复 applied effect、未解决 pending、任一种预算超限或 unjudged case，gate 仍失败。

这里的 `handler_attempted` 表示进入 handler，不表示远端动作成功；`effect_applied` 必须来自模拟环境状态、provider audit 或业务状态 verifier，不能从 `completed` 字符串猜测。`policy_allowed` 也必须由独立 policy engine/标注器给出。样例 trace 是手工冻结的离线契约 fixture，并不证明 demo runtime 已实现生产 policy engine、真实 effect observer 或防篡改 trace recorder。

## Transactional outbox crash demo

为副作用投递使用一个全新的数据库路径；脚本拒绝复用已有文件，避免把旧状态误当本轮证据：

~~~powershell
python projects/safe-agent/outbox_demo.py `
  --database artifacts/agent/outbox-demo-001.db
~~~

实验在同一事务写 local task state 与 `pending` effect。worker A 领取 lease 并让 in-memory provider 成功，但故意不 ack；worker B 用新的 `SQLiteTransactionalOutbox` 实例在 lease 到期后重领，仍以同一 `effect_id` 作为 provider idempotency key。输出应为 `attempts=2`、`provider_calls=2`、`provider_effect_count=1`、最终 `delivered`，并保留 `enqueued → claimed → lease_expired → claimed → delivered` timeline。

这是无网络、local SQLite + simulated idempotent provider 的 at-least-once 教学证据。Transactional outbox 只让本地业务状态与待投递记录原子，不能和远端 provider 构成一个事务；lease 是并发所有权，不是 exactly-once。只有 provider honor idempotency key 时重投才可能折叠；receipt 是 supplied artifact，不自动证明真实 effect。错误仅保存脱敏 machine token，dead letter 必须由 operator/runbook 处理。实验不覆盖真实网络/provider、broker、跨库/跨区域恢复、断电 durability 或生产 retention。

完整回归：

~~~powershell
python -m pytest tests/test_agent_runtime.py tests/test_agent_policy.py tests/test_agent_schema.py tests/test_agent_loop.py tests/test_model_planner.py tests/test_model_planner_control.py tests/test_agent_framework_tool_adapters.py tests/test_framework_agent_loop_control.py tests/test_mcp_sdk_memory.py tests/test_mcp_sdk_stdio.py tests/test_mcp_sdk_streamable_http.py tests/test_mcp_stdio.py tests/test_mcp_streamable_http.py tests/test_sqlite_agent_ledger.py tests/test_agent_cli.py tests/test_agent_evaluation.py
~~~

## 生产替换点

默认教学运行可使用内存 ledger；SQLiteLedger 提供跨进程持久化、原子 call-id claim 和 pending/completed 状态。若 handler 超时或崩溃，记录保持 pending，后续实例不会盲目重放。

SQLite 只能保护 claim，不能与远程副作用构成一个原子事务。`list_stale_pending` 找出待调查调用；外部审计确认成功后用 `resolve_external_completion` 写入结果，确认未发生或已经逆向操作后用 `resolve_without_completion` 标记 abandoned/compensated。旧 call id 永不重新执行。大规模服务可替换为带唯一约束的数据库，并用业务事务与 outbox 协调。

CLI scenario 中的 `approved` 布尔值只要求 runner 构造一个明确标记的 unsigned fixture grant；核心 runtime 不接受这个布尔值。生产审批服务还必须验 approver 权限、签名、一次性消费、会话与 retention。权限由认证 context、可信 resource resolver 和 policy 决定，不由模型在参数中自报。

## 后续里程碑

1. Schema migration/compatibility registry、受控 remote reference store、每个业务 tool 的 semantic cross-field validator；当前 profile 只支持 local reference，且 schema 不代替业务校验；
2. checkpoint/ledger 原子持久化、分布式 lease/broker adapter、绝对 deadline 与状态进展/长周期检测；
3. 签名/一次性 approval service、可信 trace recorder 与真实 simulator state verifier；
4. 集中 IAM/deny override、resource resolver side-channel 测试；
5. 外部文档提示注入生成器、benign over-refusal 对照、真实 provider adapter 与受保护 response receipt。
