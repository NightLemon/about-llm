# 实验 0C：云 API 预算、重试与对账

**定位**：工程选修，预计 90–120 分钟；默认全部离线，不发送真实请求、不需要 API key。

**实验导航**：[返回总览](../labs.md#lab-0) · [云 API 契约](../../models/cloud-api-contracts.md) · [项目入口](../projects/cloud-api-contracts.md#run)
{ .doc-nav }

## 开始前

**先修知识**：能够区分 logical call、attempt、request id、token estimate、provider usage 和最终 invoice。

**本页完成后**：你应该能解释为什么“超时”不等于“确定未发送”，以及每个可能计费的 retry attempt 为什么需要独立 reservation 和 reconciliation。

## 第一部分：内存预算账本

~~~powershell
python projects/cloud-api-contracts/usage_budget_toy.py
~~~

手算 60 input + 10 max output 在 `$1/M + $2/M` authored price 下为什么预留 80 micro-USD，以及 58+4 为什么结算 66 micro-USD。

依次替换 API key、Prompt、cap 与 billing scope，确认只有同 scope 的 key rotation 保持 fingerprint；再构造缺 cap、双 cap 与 bool cap，确认 transport 前失败。

**最低通过**：模拟两个线程争抢只够一次调用的 token cap，并区分“确定未发送并 cancel”“可能已发送但 usage 缺失，按 reservation 记 uncertain”“实际 usage 超预留，先记账再阻断”。

## 第二部分：持久化与恢复

~~~powershell
python projects/cloud-api-contracts/sqlite_usage_budget_demo.py --database artifacts/cloud-api/durable-budget.sqlite
~~~

使用一个全新数据库路径。核对进程重开后 active reservation 仍占容量，event 顺序为 `reserved → uncertain`，配置/请求 fingerprint 不包含假密钥。

**最低通过**：解释 worker crash 或 lease/TTL 到期为什么不能自动 cancel，以及 `BEGIN IMMEDIATE` 为什么不能把 SQLite commit 与远程 HTTP、provider usage 或 invoice 变成原子事务。

## 第三部分：HTTP 结果对账

~~~powershell
python projects/cloud-api-contracts/budgeted_http_demo.py --database artifacts/cloud-api/budgeted-http.sqlite
~~~

手算 settled 58+4=66 与 HTTP 500 uncertain 60+10=80，确认最终 committed 为 146 micro-USD。把 HTTP 500 改成 ConnectError、2xx 缺 usage、malformed JSON 和 cancellation，解释为什么只有明确的连接前失败可以 cancel。

## 第四部分：逐 attempt 预算重试

~~~powershell
python projects/cloud-api-contracts/budgeted_retry_demo.py --database artifacts/cloud-api/budgeted-retry.sqlite
~~~

核对 `logical-call:attempt:1` 与 `logical-call:attempt:2` 是两个独立 tombstones，event 顺序为：

```text
reserved → uncertain → reserved → settled
```

手算 HTTP 500 的 80 与最终成功的 66 为什么合计 146。成功 usage 只属于 attempt 2，不能替 attempt 1
证明零计费。

把 hard cost limit 从 200 改为 140。第一次 500 已按 uncertain 提交 80；第二次再 reserve 80 会让 projected total
变成 160，所以必须在 transport 前失败，MockTransport call count 仍为 1。

再做两个反例：把 500 改为 ConnectError，确认首个 attempt cancelled、第二个 settled；把响应改为
`429 + Retry-After: 2`，用注入的 sleep recorder 检查等待值。

**最低通过**：能画出 `reserve → send → terminalize → retry sleep`，并解释 cancellation 若发生在 reserve 后、
trace 前为什么仍按 uncertain。还要指出当前 orchestrator 只覆盖 JSON，不支持 streaming partial-output replay，
也不解析 Provider-specific error usage。

## 常见失败

- 用 request hash 代替 provider request id 或 invoice 对账。
- 超时后一律释放预算并自动重试，造成潜在双重计费。
- 把 API key 写入 fingerprint、日志、异常或测试快照。
- 认为 SQLite 事务能覆盖远程 provider 副作用。

## 交付与结论边界

最低交付物包括四份机器输出、一份 attempt ledger、一份 uncertain reconciliation 方案，以及一个被 hard gate
挡在 transport 前的故意失败案例。对账输入至少保存 stable call ID、request fingerprint、attempt/request-ID trace、
Provider usage/billing export 与人工处置结果。

离线 HTTP/SQLite control 不执行真实 DNS、TLS、provider 请求或计费；authored usage 和价格只验证本地协议，不能证明供应商 usage、取消、账单或 exactly-once billing。
