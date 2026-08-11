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
python -m pip install -e ".[agents]"
~~~

当前安全 profile 明确要求 Draft 2020-12、root `type: object`，并以 `additionalProperties: false` 或 `unevaluatedProperties: false` 闭合根参数；`$ref/$dynamicRef` 只允许 local fragment，拒绝 `$id` 与外部 retrieval。Schema 本身和 instance 都有 UTF-8 canonical byte cap。`format` 默认按标准作为 annotation；只有 `enforce_formats=True` 才用当前 `jsonschema` FormatChecker 执行，且未知 format 在构造 contract 时拒绝。Schema/validator revision、`jsonschema` 精确版本、format mode、schema bytes 和 instance cap 进入 contract identity。校验不做字符串转数字等 coercion、不插入 `default`、不执行资源授权；失败只暴露 keyword 和 JSON Pointer，不回显 rejected value。

下面的 control 使用两条代码内冻结的 recorded provider response，不联网也不调用模型。第一条 JSON 提议只读 `fixture_tool`，标准 JSON Schema、tenant resource resolver 和 exact-capability policy 允许后，handler 返回一段恶意指令文本；第二次 request 把它按不可信 observation 纳入状态，模型 fixture 再提议 finish，最后由独立 exact verifier 核对本地 event 才完成：

~~~powershell
python projects/safe-agent/model_planner_control.py
~~~

报告锁定两次 request/response fingerprint、两个 decision id、Draft/schema/validator identity、62 个 authored fixture tokens、0.03 authored cost units、一次 handler attempt 和最终 `verified answer`。四条 negative control 证明 request/state 漂移不能 replay、Markdown-fenced JSON 被拒绝、模型参数虽通过 JSON parser 仍会被 runtime `const` schema 在 resolver/policy/handler 前拒绝、缺 capability 时合法 proposal 也在 handler 前被 policy 拒绝。这里的 token/cost/provider request id 都是 authored metadata；control 不证明真实 API schema、目标模型能遵循协议、provider usage/账单、网络重试、生产 IAM 或开放任务 verifier。接真实 provider 时，adapter 仍需从原始响应提取精确 revision/usage/finish reason，并保存受保护的原始 receipt；缺字段不能猜测。

手写 `PlannerToolContract` 仍可能与任意 callback validator 漂移；需要强一致时应由 `JSONSchemaToolContract.planner_contract()` 与 `.build_tool()` 同源生成。JSON Schema 只验证声明的 JSON 结构和值约束，不知道 resource 是否存在/归属当前 tenant、调用是否获权、effect 是否安全或 handler 返回是否真实，这些边界不能挪进模型 schema。

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
python -m pytest tests/test_agent_runtime.py tests/test_agent_policy.py tests/test_agent_schema.py tests/test_agent_loop.py tests/test_model_planner.py tests/test_model_planner_control.py tests/test_sqlite_agent_ledger.py tests/test_agent_cli.py tests/test_agent_evaluation.py
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
