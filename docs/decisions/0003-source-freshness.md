# ADR 0003：来源失效时显式降级

- 状态：接受
- 日期：2026-08-28

## 背景

API、模型目录、价格和政策会变化。只在 PR 中检查 URL 与最近日期，会让已部署页面在来源失效后继续显示为“已核验”；反过来，把网络可用性设为构建门禁又会使临时故障阻止发布状态更新。

## 决策

来源注册表使用稳定 ID、语义审核状态、易变性、下次复核日期、使用页面和可选 revision/fingerprint。PR 仅执行离线结构检查。定时任务执行有界网络探测，并把不可访问标为 `unknown`、fingerprint 变化标为 `pending-review`。MkDocs 继续构建和部署，同时在引用处及页首显示降级状态。

逐项语义复核确认影响后才更新正文、状态、核对日期和 fingerprint。自动化不得把网络失败或内容变化解释成“原结论已错”或“仍然正确”。

## 结果

读者能看到来源状态变化，维护者能获得可审计报告，临时网络故障不会冻结旧的“已核验”站点。代价是高易变来源需要持续语义复核。

## 当前实施说明（2026-09-13）

注册表中的原始审核状态只可为 `verified`、`pending-review` 或 `rejected`，并由语义复核维护。页面 badge 显示的是有效状态：原始状态为 `verified` 时，超过 `next_review_at` 显示 `stale`；若存在定时任务生成的 probe 报告，其 `unknown` 或 `pending-review` 观测会覆盖该有效状态。原始状态不是 `verified` 时，页面直接显示该原始状态。

`stale` 是日期计算结果，`unknown` 是本次探测不可访问的观测；二者都不应写回注册表来伪装成语义审核。probe 只写入 `artifacts/source-health.json` 这类报告，供渲染时读取，不修改登记项。当前工作流只在 schedule 事件运行 `scripts/probe_sources.py`；PR 的离线检查与手工执行不运行它。

人工或 LLM 复核都应记录实际核对日期、范围、版本或 fingerprint，并判断变化是否影响每一条 claim。LLM 复核还必须按 [ADR 0004](0004-llm-source-review.md) 登记模型身份与逐项记录；没有这份记录的 AI 批处理仍只能提出待复核项，不能刷新 `checked_at` 或把来源改为 `verified`。
