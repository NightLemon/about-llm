# Cloud API Contracts：安全重试、计费和收尾

这个项目不教你“怎样发出一条 HTTP 请求”，而是追踪一次云模型调用从请求构造、发送、流式接收，到重试、预算结算
和未知结果对账的完整生命周期。目标是在网络异常、进程重启或 provider 协议变化时，仍然知道发生了什么。

第一次学习请从[项目教学页](../../docs/practice/projects/cloud-api-contracts.md)开始。那里沿一次逻辑调用解释每个状态；
本页只保留快速运行、脚本索引和排错信息。

## 第一次运行 { #run }

先运行一次包含 HTTP 500、自动重试和逐次预算的完整离线调用。每次使用新的数据库路径：

```powershell
New-Item -ItemType Directory -Force artifacts/cloud-api | Out-Null

python projects/cloud-api-contracts/budgeted_retry_demo.py `
  --database artifacts/cloud-api/first-budgeted-retry.sqlite
```

输出会显示第一次 attempt 保守记为 `uncertain`，第二次按实际 usage 结算。完整解释见
[教学页](../../docs/practice/projects/cloud-api-contracts.md#run)。随后可运行几个较小的组件实验：

```powershell
python projects/cloud-api-contracts/prompt_contract_walkthrough.py

python -m about_llm.integrations.cloud_api_cli verify `
  --contracts projects/cloud-api-contracts/contracts.example.jsonl `
  --output artifacts/cloud-api/contracts.json

python -m about_llm.integrations.cloud_api_cli retry-matrix `
  --output artifacts/cloud-api/retry-matrix.json

python projects/cloud-api-contracts/gemini_interactions_replay.py

python projects/cloud-api-contracts/usage_budget_toy.py
```

这些小实验分别回答：

- 模型输出怎样经过 JSON、结构、原文位置和字段证据检查；
- 三种 Provider 文本协议怎样映射为共同业务对象；
- Gemini Interactions 的函数参数怎样跨步骤重建，为什么流结束后仍可能等待客户端动作；
- 哪些失败允许安全重放；
- 一次调用怎样先预留最大费用，再根据实际用量结算或进入结果未知状态。
Prompt Contract 实验的逐步解释见 [Prompt Engineering 与输出契约](../../docs/applications/prompting.md#contract-walkthrough)。

## 一次逻辑调用的状态

```text
本地 preflight
→ reserve 最大费用
→ attempt 1 send boundary
→ response / error / partial stream
→ settle / cancel / uncertain
→ 必要时等待
→ attempt 2 单独 reserve
→ 最终结果或人工对账
```

每个 attempt 都有自己的 reservation。Attempt 1 必须先进入终态，才能为 attempt 2 占用新的预算。进程崩溃留下的
active reservation 继续占额度，因为“进程不在了”不能证明 provider 没收到请求。

## Provider 可以统一到什么程度

| Provider family | 请求差异 | Response/usage/terminal 的主要位置 |
|---|---|---|
| OpenAI-compatible Chat | System 位于 `messages` | `choices/message`、usage、finish reason |
| Anthropic Messages | System 是顶层字段 | Content blocks、usage、stop reason |
| Gemini `generateContent` | `user/model` roles 与 `systemInstruction` | Parts、`usageMetadata`、finish reason |

共同对象只保留文本对话真正共享的部分。工具调用、思考内容、媒体、拒答或没有文本的结果不会被悄悄转成字符串。
Gemini Interactions 与 `generateContent` 也属于不同协议，不能因为品牌相同就混用字段。

## 什么时候可以重试

| 已观察到的情况 | 默认处理 |
|---|---|
| Connect/pool failure，能够证明尚未发送 | 在 deadline 与 attempt cap 内重试 |
| 408、429、部分 5xx，且请求可重放 | 遵守 `Retry-After` 后重试 |
| Write/read/protocol timeout | 结果可能已发生，停止自动重放并对账 |
| 已交付 partial SSE | 保留 partial output，不自动重放 |
| 工具副作用或重放语义未知 | 不自动重放 |
| Schema、认证或业务拒绝 | 修复请求，而不是重复同一调用 |

即使纯文本生成没有业务副作用，重复请求也可能重复计算和计费。Provider 的 idempotency key 是否可用、覆盖什么范围，
必须按具体接口确认。

`Retry-After` 只接受合法的非负秒数或 HTTP date。若等待会越过 overall deadline，本次逻辑调用应直接结束。

## HTTP 与 SSE 边界

JSON 请求执行器依次检查：

- origin 是否精确匹配允许列表，是否使用 HTTPS；
- 重定向是否关闭，URL 是否带有 query、userinfo 或 fragment；
- Content-Type、响应大小和 JSON 结构是否符合约定。

JSON 中的重复字段和非法数值会失败。单次请求超时与整个逻辑调用的 deadline 分开记录。

SSE 解析器接收任意字节片段：一次网络读取可能只有半个 UTF-8 字符，也可能包含多个事件。因此，网络片段、
SSE 事件、文本增量和模型 token 是四种不同对象。

流式执行遵守以下规则：

- 只有空行结束一个 SSE event；截断的行或 event 会失败。
- Tool、thinking、media 和未知 block 不会被静默丢弃。
- 一旦 2xx stream 开始，后续失败不自动重放。
- 已经交给回调的 partial text 无法撤回。
- Client close 只证明本地连接结束，不证明 provider 停止 GPU 工作或计费。

## 预算怎样跨进程重启

每次 demo 使用新的数据库路径：

```powershell
python projects/cloud-api-contracts/sqlite_usage_budget_demo.py `
  --database artifacts/cloud-api/durable-budget.sqlite
```

SQLite 账本在事务中保存价格与配置身份、费用预留和事件。重开数据库后，未终结的预留仍会占用额度。
操作人员需要结合 request ID、provider 用量或账单导出和业务状态，才能将它结算、取消或标为结果未知。

逐 attempt 的 HTTP 与 retry 接线：

```powershell
python projects/cloud-api-contracts/budgeted_http_demo.py `
  --database artifacts/cloud-api/budgeted-http.sqlite

python projects/cloud-api-contracts/budgeted_retry_demo.py `
  --database artifacts/cloud-api/budgeted-retry.sqlite
```

这套流程避免“内部重试了两次，预算却只记一次”。SQLite 能保证同一数据库内的本地配额原子性，不能和远端请求形成
一个事务，也不能替代跨区域 quota service。

## 根据当前问题选择入口

| 你想理解什么 | 入口 |
|---|---|
| 结构化抽取怎样从错误输出走到可接受结果 | `prompt_contract_walkthrough.py` |
| 三种 text API 怎样映射共同对象 | `cloud_api_cli verify` |
| Interactions event 与 resource status 为什么不同 | `gemini_interactions_replay.py` |
| Replay-safe、deadline 与 `Retry-After` | `cloud_api_cli retry-matrix` |
| JSON HTTP 的 origin、redirect、timeout 与解析边界 | `execute_json_request` 及 `tests/test_cloud_http.py` |
| SSE framing 与三种 provider stream | `SSEDecoder`、`cloud_stream` 相关测试 |
| 费用 reservation、settle 与 uncertain | `usage_budget_toy.py` |
| 崩溃后预算怎样恢复 | `sqlite_usage_budget_demo.py` |
| 单 attempt 怎样连接预算与 HTTP | `budgeted_http_demo.py` |
| 每次 retry 怎样单独预留预算 | `budgeted_retry_demo.py` |
| Responses event graph 与 Chat delta 的差别 | `openai_responses_replay.py` |
| Opaque reasoning artifact 怎样绑定上下文 | `reasoning-replay-matrix` |
| 哪些 trajectory blocks 可以发布 | `trajectory-release-gate` |

每条路径的解释和完成信号见[项目教学页](../../docs/practice/projects/cloud-api-contracts.md)。固定结果和适用范围见
[Cloud API 证据页](../../docs/evidence/cloud-api-controls.md)。

## Responses、Opaque artifact 与发布投影

OpenAI Responses 按“响应 → 输出项 → 内容片段”的事件层次组织数据，不是 Chat Completions 的文本增量：

```powershell
python projects/cloud-api-contracts/openai_responses_replay.py `
  --events projects/cloud-api-contracts/openai-responses-events.example.jsonl
```

本地回放会跟踪消息、拒答、函数参数、输出项完成、最终输出和用量。函数参数即使无法解析为严格 JSON，
也会保留原字符串并标记为不可执行。

Opaque reasoning artifact 即使是密文，也要绑定租户、主体、会话、分支、模型、策略、密钥和过期时间：

```powershell
python -m about_llm.integrations.cloud_api_cli reasoning-replay-matrix `
  --output artifacts/cloud-api/reasoning-replay-matrix.json
```

准备对外发布 trajectory 时，只允许明确审过的 `text/tool_call/tool_result/citation` blocks：

```powershell
python -m about_llm.integrations.cloud_api_cli trajectory-release-gate `
  --input projects/cloud-api-contracts/trajectory-release.example.json `
  --output artifacts/cloud-api/trajectory-release-report.json
```

Shape allowlist 通过不代表文本已经完成 secret、PII、版权或用途审查；这些是独立发布门禁。

## 主要输入与输出

| 文件 | 用途 |
|---|---|
| `contracts.example.jsonl` | 三类 adapter 的 request、固定 response 与预期规范化结果 |
| `gemini-interactions-function-call.example.sse` | Interactions 函数调用与 `requires_action` 固定事件 |
| `openai-responses-events.example.jsonl` | Responses typed-event 本地 replay |
| `trajectory-release.example.json` | 允许与拒绝发布的 block 样例 |
| `artifacts/cloud-api/contracts.json` | Provider 映射验证结果 |
| `artifacts/cloud-api/retry-matrix.json` | 重试决策表 |
| `artifacts/cloud-api/*.sqlite` | Durable budget、reservation 与 attempt 事件 |

Pricing snapshot 是人工核对的本地估值，不是 provider 发票。真实成本还可能受 reasoning/cache tokens、最低计费单位、
套餐、税费和 tokenizer 差异影响。

## 常见故障

| 现象 | 先检查 |
|---|---|
| 相同消息在不同 provider 返回不同角色/字段 | Adapter 是否保留 provider-specific block 与 terminal 语义 |
| 429 后立即再次失败 | `Retry-After`、deadline、attempt cap 和 jitter |
| Read timeout 后重复计费 | Outcome 已不确定，是否停止自动重放并进入对账 |
| Client 已取消，reservation 仍 active | 是否越过发送边界；未知结果应保守记账 |
| 流式文本重复或断裂 | 是否把 network chunk 当成 event/token，或在 partial output 后自动重试 |
| Usage 缺失却记成 0 | 已发送请求缺 usage 应进入 uncertain，而不是免费 |
| Actual cost 超过 hard limit | 先持久化已发生费用，再报告 post-call breach |
| 进程重启后额度越来越少 | 查找 stale active reservation，并用 provider 记录对账 |
| API key 出现在 trace | Request identity 和日志是否对 credential header 做了脱敏 |
| Opaque artifact 跨会话可重放 | Associated data 是否绑定完整上下文，是否有 single-use ledger |

## 运行检查

```powershell
python -m pytest `
  tests/test_cloud_api.py `
  tests/test_gemini_interactions_replay.py `
  tests/test_cloud_api_cli.py `
  tests/test_openai_responses_replay.py `
  tests/test_reasoning_artifact.py `
  tests/test_trajectory_release.py `
  tests/test_cloud_api_retry.py `
  tests/test_cloud_http.py `
  tests/test_sse.py `
  tests/test_cloud_stream.py `
  tests/test_usage_budget.py `
  tests/test_sqlite_usage_budget.py `
  tests/test_budgeted_cloud.py `
  tests/test_prompt_contract.py -q

python scripts/check_docs.py
python scripts/check_content_accuracy.py
```

默认测试使用固定 JSON、内存字节输入和 `httpx.MockTransport`，不会访问真实 provider。网络 smoke 必须显式设置
允许的服务地址、请求数、token/费用上限和超时，并单独保存计费与失败证据。
