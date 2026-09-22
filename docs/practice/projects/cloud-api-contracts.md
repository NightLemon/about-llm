# Cloud API Contracts：一次调用怎样留下可对账结果

**项目导航**：[项目索引](../project-index.md) · [云 API 契约](../../models/cloud-api-contracts.md) ·
[可靠性、重试与预算](../../models/cloud-api-reliability.md) · [实验 0C](../labs/lab-0c-cloud-budget.md) ·
[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/cloud-api-contracts/README.md) ·
[证据台账](../../evidence/cloud-api-controls.md)
{ .doc-nav }

这个项目把解析、流式、重试和预算接成一条可审计调用。教材解释各机制，本页只决定按什么顺序组装；完整命令、文件与
故障处理由项目 README 维护，固定结果由证据页维护。

## 先跟完一次有重试的调用 {#run}

第一次只运行一个离线场景：

```powershell
python projects/cloud-api-contracts/budgeted_retry_demo.py `
  --database artifacts/cloud-api/first-budgeted-retry.sqlite
```

运行前预测三件事：第一次 HTTP 500 后是否释放 80 micro-USD，第二次发送是否复用 reservation ID，最终本地估值
是多少。固定样例应形成两条独立 attempt：第一条费用待确认，第二条按 usage 结算。它使用 `httpx.MockTransport`
与 `.invalid` 目标，不访问真实供应商。

## 项目要连接的四层

| 层 | 交付物 | 通过后仍不知道什么 |
|---|---|---|
| 协议 | 三类文本 adapter、Responses/Interactions typed events、SSE 终态 | 真实端点当前是否兼容 |
| 可靠性 | 每个 attempt 的 retry/replay/outcome 决策与总 deadline | 供应商是否已经生成或计费 |
| 预算 | `reserve → settle/cancel/uncertain` 账本和重启恢复 | 本地估值是否等于发票 |
| 发布 | 对外投影只含允许 block，拒绝不透明或未知类型 | 可见文本是否已完成 PII/版权审核 |

这四层不能合成一个 `success`。HTTP 500 可以允许重试，同时让第一次费用保持 `uncertain`；流式终态完成也可能仍有
待客户端执行的工具动作；结构允许发布也不代表内容安全。

## 推荐组装顺序

1. **固定文本契约**：先让 OpenAI-compatible、Anthropic 和 Gemini 样例映射到共同对象，同时保留各自终态。
2. **加入流式状态机**：区分网络 byte、SSE event、文本增量与模型 token；截断必须成为失败终态。
3. **加入重试判断**：分别回答错误是否可重试、请求能否重放、上一次结果是否明确。
4. **把预算放到发送边界前**：每次 attempt 独立预留，随后进入 `settled`、`cancelled` 或 `uncertain`。
5. **让账本跨重启**：旧的 active reservation 不能因进程退出自动释放，必须结合供应商记录对账。
6. **限制可发布对象**：从内部轨迹重建窄投影，再执行 block allowlist 与独立内容审查。

每一步的最小命令在[运行手册](https://github.com/NightLemon/about-llm/blob/main/projects/cloud-api-contracts/README.md)
按问题列出。遇到差异时，从最早改变状态的一层排查，不要只比较最后的回答文本。

## 两条必须保留的反例

第一条是“已经发送但 usage 缺失”。它必须占用费用待确认额度，不能被记作 0。第二条是“已经向用户交付部分 SSE 后
断流”。它必须停止透明重放，否则用户可能看到重复文本并产生第二笔费用。

对工具调用还要增加副作用边界：模型输出参数不等于已经执行，结果未知时先查询效果账本。完整执行原语见
[Agent Runtime](../../applications/agent-runtime.md)。

## 完成标准

- 能从 logical call 下钻到每个 attempt、request fingerprint、terminal 与 reservation。
- 能解释连接前失败、HTTP 5xx、读取超时和 partial stream 为什么有不同重试结论。
- 能在进程重启后找到未终结 reservation，并说明对账证据来自哪里。
- 能区分共同文本对象与供应商专有 tool/reasoning/media 事件。
- 能给公开轨迹构造窄投影，并说明结构门禁没有执行 secret/PII/版权审核。
- 能把离线 MockTransport/SQLite 结果与真实端点、账单和生产 SLO 分开陈述。

最终交付不是一张“调用成功”截图，而是一条能复算的 attempt 时间线、一份预算账本和一组失败终态。精确固定结果、
版本和未覆盖项只在[云 API 证据台账](../../evidence/cloud-api-controls.md)维护。
