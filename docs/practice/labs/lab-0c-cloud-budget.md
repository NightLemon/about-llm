# 实验 0C：云 API 预算、重试与对账

**定位**：工程选修，预计 90–120 分钟；全部实验默认离线运行，不需要 API key，也不会向模型供应商发送请求。

**实验导航**：[返回总览](../labs.md#lab-0) · [云 API 契约](../../models/cloud-api-contracts.md) · [项目入口](../projects/cloud-api-contracts.md#run)
{ .doc-nav }

## 开始前

假设客服系统调用云模型生成一条回复。第一次请求收到 HTTP 500，第二次重试成功。用户只看到最终答案，
但预算系统必须保留两次发送的记录：第一次可能已经生成并计费，不能因为响应失败就当作零成本。

先认识本实验反复使用的四个对象：

| 对象 | 在这个例子里是什么 |
|---|---|
| 逻辑调用（logical call） | 用户点击一次“生成回复” |
| 发送尝试（attempt） | 程序实际向远端发起的一次 HTTP 请求 |
| 预算预留（reservation） | 发送前暂时占用本次调用最多可能花掉的额度 |
| 对账（reconciliation） | 根据已知结果，把预留记录结算、取消或标成费用待确认 |

本实验使用一份固定的示例价格：输入 `$1 / 1M tokens`，输出 `$2 / 1M tokens`。换算后，每个输入 token 是
1 micro-USD，每个输出 token 是 2 micro-USD。因此，输入估算为 60 tokens、最多输出 10 tokens 时，需要先预留：

\[
60\times 1 + 10\times 2 = 80\ \text{micro-USD}.
\]

一次带重试的调用会走过下面这条主线：

```text
逻辑调用开始
├── attempt 1：预留 80 → 发送 → HTTP 500、usage 缺失 → 暂记 uncertain 80
└── attempt 2：重新预留 80 → 发送 → HTTP 200、usage=58/4 → 结算 66
最终本地账本：80 + 66 = 146 micro-USD
```

接下来的四部分不是四套互不相关的概念。第一部分先理解内存中的状态变化，第二部分让状态跨进程保存，
第三部分接上 HTTP 结果，第四部分再加入重试。

## 第一部分：内存预算账本

先运行最小例子：

~~~powershell
python projects/cloud-api-contracts/usage_budget_toy.py
~~~

第一条调用先预留 80 micro-USD。响应报告实际使用 58 个输入 token 和 4 个输出 token，因此最终结算为：

\[
58\times 1 + 4\times 2 = 66\ \text{micro-USD}.
\]

预留与结算的差额会释放给后续调用。脚本中的第二条调用最多占用 30 micro-USD；它模拟已经越过发送边界、
但没有可信 usage 的情况，所以把整笔预留记为 `uncertain`，而不是记成免费。

接着运行三个针对性测试：

~~~powershell
python -m pytest tests/test_usage_budget.py `
  -k "supported_requests_bind_exact_output_cap_and_hide_credentials or request_semantic_or_cap_drift_changes_budget_fingerprint or two_concurrent_reservations" `
  -q
~~~

结合测试代码确认三件事：

1. 在同一计费账户或项目内轮换 API key，不会改变请求指纹；密钥值不会写进指纹。
2. Prompt、输出上限或计费范围发生变化，请求指纹也会变化。
3. 两个线程同时争抢最后一份额度时，只能有一个预留成功。

完成这一部分后，你应该能区分三个终态：能证明尚未发送时才 `cancelled`；有完整 usage 时 `settled`；
可能已经发送但费用未知时 `uncertain`。

## 第二部分：持久化与恢复

为每次运行选择一个尚不存在的数据库文件：

~~~powershell
python projects/cloud-api-contracts/sqlite_usage_budget_demo.py `
  --database artifacts/cloud-api/durable-budget.sqlite
~~~

输出中的 `active_after_reopen` 应保留一条记录。随后程序把它对账为 `uncertain`，事件顺序应为：

```text
reserved → uncertain
```

这里模拟的是“进程在发送前后某处退出，但来不及写终态”。重启后，请求是否到达供应商仍属未知，所以旧预留继续占用额度。

记录年龄或 lease 到期只说明本地状态很久没有更新。要取消预留，还需要能够证明请求停在发送边界之前。

SQLite 的 `BEGIN IMMEDIATE` 可以防止两个本地写入者同时花掉最后一份额度。它无法把本地事务、远端 HTTP、
供应商 usage 和最终账单合并成一个原子操作。真正对账时还需要本地 attempt、供应商 request ID、usage 导出和业务结果。

## 第三部分：HTTP 结果对账

下面的脚本用 `httpx.MockTransport` 运行两条独立调用：一条成功，一条收到 HTTP 500。

~~~powershell
python projects/cloud-api-contracts/budgeted_http_demo.py `
  --database artifacts/cloud-api/budgeted-http.sqlite
~~~

| 调用 | 已观察到的结果 | 账本终态 | 本地计入费用 |
|---|---|---|---:|
| `call-success` | HTTP 200，usage 为 58 input / 4 output | `settled` | 66 micro-USD |
| `call-http-500` | HTTP 500，没有可信 usage | `uncertain` | 80 micro-USD |

最终 `committed_estimated_microusd` 应为 146。HTTP 500 说明响应失败，不说明供应商一定没有处理请求。

再运行失败路径测试：

~~~powershell
python -m pytest tests/test_budgeted_cloud.py `
  -k "connect_failure_is_the_only_transport_path_that_cancels or sent_success_without_trustworthy_usage_is_uncertain or cancellation_after_reservation_never_fabricates_zero_usage" `
  -q
~~~

在这套离线执行器中，只有被分类为“连接前失败”的 `ConnectError` 路径会释放预留。HTTP 错误、成功响应却缺少 usage、
JSON 解析失败，以及预留后发生的取消，都已经越过了无法证明未发送的边界，因此保守记为 `uncertain`。

## 第四部分：逐 attempt 预算重试

现在把前面的状态连接成一次真正的逻辑调用：

~~~powershell
python projects/cloud-api-contracts/budgeted_retry_demo.py `
  --database artifacts/cloud-api/budgeted-retry.sqlite
~~~

输出中应出现两条不可复用的 reservation ID：`logical-call:attempt:1` 和 `logical-call:attempt:2`。两次 attempt 的事件合在一起是：

```text
reserved → uncertain → reserved → settled
```

第一次的 80 micro-USD 不会被第二次成功响应中的 usage 覆盖。第二次只结算自己的
`58×1 + 4×2 = 66`，所以逻辑调用的本地累计值仍是 146。

下面两个测试展示预算门禁和 `Retry-After`：

~~~powershell
python -m pytest tests/test_budgeted_cloud.py `
  -k "retry_budget_gate_blocks_second_network_attempt or retry_after_delay_is_preserved_by_budget_orchestrator or retry_connect_failure_then_success_cancels_first_attempt" `
  -q
~~~

预算门禁测试把费用上限设为 140。第一次 attempt 已经以 `uncertain` 提交 80；第二次若再预留 80，预计总额会变成
160，因此程序必须在第二次网络发送前停止。测试中的 transport 调用次数保持为 1。

`Retry-After: 2` 测试则记录一次 2 秒等待。等待规则只决定何时重试；请求是否适合重放、是否还有预算和总 deadline，
仍然需要分别判断。另一个测试把第一次结果改成明确的连接前失败，因此首笔预留可以取消，第二次成功后只结算 66。

完成这一部分后，你应能按顺序解释：

```text
预留预算 → 跨过发送边界 → 写入本次 attempt 的终态 → 判断能否重试 → 等待 → 为下一次 attempt 重新预留
```

## 常见失败

- 用本地 request hash 代替供应商 request ID 或账单记录做最终对账。
- 一看到超时就释放预算并自动重试，忽略上一次请求可能已经到达远端。
- 多次网络发送共用一条 reservation，使预算只计算一次。
- 把 API key 写入指纹、日志、异常或测试快照。
- 认为 SQLite 事务能够覆盖远端生成和计费。

## 交付与结论边界

完成实验时，保留以下材料：

- 四次主命令的输出；
- 一张写有 reservation ID、request ID、状态和金额的 attempt 账本；
- 一份说明 `uncertain` 记录需要哪些外部数据才能结清的方案；
- 一个被 140 micro-USD 门禁挡在第二次网络发送前的测试结果。

这些实验只验证本地 JSON 请求、SQLite 账本和固定的 HTTP 响应。运行过程不访问真实供应商，也不产生费用。

流式输出中断后的重放不在本实验范围内。示例价格与用量只用于检查账本算术；真实 token 分类、缓存、取消语义和发票，
仍需根据目标 API 与账户数据核对。
