# Cloud API Contracts 运行手册

本目录提供云模型协议、SSE、重试、预算和轨迹发布的离线可执行样例。学习顺序与结论解释见
[项目教学页](../../docs/practice/projects/cloud-api-contracts.md)；通用机制见
[云 API 契约](../../docs/models/cloud-api-contracts.md)和
[可靠性、重试与预算](../../docs/models/cloud-api-reliability.md)。本页只维护安装、命令、输入输出和故障定位。

## 安装

从仓库根目录运行：

```powershell
python -m pip install -c constraints/ci.txt -e ".[dev,api]"
python scripts/doctor.py
New-Item -ItemType Directory -Force artifacts/cloud-api | Out-Null
```

默认样例使用 `.invalid` 地址、固定 JSON/SSE 和 `httpx.MockTransport`，不会访问真实供应商或产生费用。

## 第一次运行 {#run}

每次使用一个尚不存在的数据库路径：

```powershell
python projects/cloud-api-contracts/budgeted_retry_demo.py `
  --database artifacts/cloud-api/first-budgeted-retry.sqlite
```

检查输出中有两个不同的 reservation ID，第一条为 `uncertain`，第二条为 `settled`。固定样例的 80、66 与 146
micro-USD 只用于核对本地状态机，不是供应商价格或发票。

## 按问题选择命令

### Prompt 与 provider 对象

```powershell
python projects/cloud-api-contracts/prompt_contract_walkthrough.py

python -m about_llm.integrations.cloud_api_cli verify `
  --contracts projects/cloud-api-contracts/contracts.example.jsonl `
  --output artifacts/cloud-api/contracts.json

python projects/cloud-api-contracts/gemini_interactions_replay.py

python projects/cloud-api-contracts/openai_responses_replay.py `
  --events projects/cloud-api-contracts/openai-responses-events.example.jsonl
```

### 重试、流式与预算

```powershell
python -m about_llm.integrations.cloud_api_cli retry-matrix `
  --output artifacts/cloud-api/retry-matrix.json

python projects/cloud-api-contracts/usage_budget_toy.py

python projects/cloud-api-contracts/sqlite_usage_budget_demo.py `
  --database artifacts/cloud-api/durable-budget.sqlite

python projects/cloud-api-contracts/budgeted_http_demo.py `
  --database artifacts/cloud-api/budgeted-http.sqlite

python projects/cloud-api-contracts/budgeted_retry_demo.py `
  --database artifacts/cloud-api/budgeted-retry.sqlite
```

### Reasoning 工件与发布投影

```powershell
python -m about_llm.integrations.cloud_api_cli reasoning-replay-matrix `
  --output artifacts/cloud-api/reasoning-replay-matrix.json

python -m about_llm.integrations.cloud_api_cli trajectory-release-gate `
  --input projects/cloud-api-contracts/trajectory-release.example.json `
  --output artifacts/cloud-api/trajectory-release-report.json
```

完整协议解释分别见[不透明推理工件安全](../../docs/quality/reasoning-artifact-security.md)与
[实验 0D](../../docs/practice/labs/lab-0d-reasoning-artifact-security.md)。

## 输入与输出

| 路径 | 用途 |
|---|---|
| `contracts.example.jsonl` | 三类 adapter 的请求、固定响应与预期规范化结果 |
| `gemini-interactions-function-call.example.sse` | Interactions 函数调用与 `requires_action` 事件 |
| `openai-responses-events.example.jsonl` | Responses typed-event 本地回放 |
| `trajectory-release.example.json` | 允许与拒绝发布的 block 样例 |
| `artifacts/cloud-api/contracts.json` | Provider 映射报告 |
| `artifacts/cloud-api/retry-matrix.json` | 重试决策报告 |
| `artifacts/cloud-api/*.sqlite` | Durable budget、reservation 与 attempt 事件 |

不要提交 `artifacts/`。普通日志不得保存 API key、认证 header、原始 reasoning 或未经审查的敏感正文。

## 故障速查

| 现象 | 先检查 |
|---|---|
| 输出数据库已存在 | 换一个新路径，避免把两次实验混成一条账本 |
| 429 后立即再次失败 | `Retry-After`、overall deadline、attempt cap 与 jitter |
| Read timeout 后重复费用 | 是否把 outcome unknown 错当成未发送 |
| SSE 文本重复或断裂 | 是否把 byte chunk 当成 event/token，或在 partial output 后重试 |
| Usage 缺失却记成 0 | 已发送请求应进入 `uncertain` |
| 重启后额度减少 | 找 stale active reservation，并用 request ID/账单对账 |
| 高级字段无法解析 | 当前脚本是否选对 Chat、Responses 或 Interactions 协议 |
| Opaque artifact 可跨上下文使用 | AAD、可信上下文比较和 single-use ledger 是否同时启用 |

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

这些检查不访问网络。真实端点测试必须显式限制 origin、模型、请求数、token、费用和时限，并把结果与本地固定样例
分开保存。
