# 云模型 API 可靠性：一次重试为什么要记两笔账

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已能解析 typed response，准备处理重试、流式故障、费用和生产治理的工程师。
- **先修**：[云 API 契约基础](cloud-api-contracts.md)、deadline、幂等与基本数据库事务。
- **首次阅读**：客服调用的两次 attempt → 两种“不确定” → 重试三问 → 预算与对账。
- **完成信号**：能解释第一次 HTTP 500 为什么可以进入重试决策，却仍要把它的预算记为费用待确认。
- **卡住时**：画出一次 logical call 的 attempt timeline，不要先写自动重试循环。

</div>

**学习入口**：[契约基础](cloud-api-contracts.md) · [实验 0C](../practice/labs/lab-0c-cloud-budget.md) · [生产检查表](../practice/production-checklist.md) · [证据台账](../evidence/cloud-api-controls.md)
{ .doc-nav }

云模型调用最棘手的失败，不是明确的 400 或成功的 200，而是“客户端不知道远端做了什么”。

请求可能已经被接收并生成、计费或触发工具，只是响应在返回途中丢失。可靠系统必须把这种不确定性保留下来，而不是用一个 retry 按钮把它覆盖。

## 先跟一次重试走完 {#worked-retry}

假设客服系统要生成一条回复。用户只点击一次，这叫一次**逻辑调用（logical call）**。程序每向供应商发送一次
HTTP 请求，就产生一次**发送尝试（attempt）**。

[实验 0C](../practice/labs/lab-0c-cloud-budget.md)为这次调用准备了一个离线场景。示例价格不代表任何真实供应商：
每个输入 token 计 1 micro-USD，每个输出 token 计 2 micro-USD。程序估计输入 60 个 token，最多输出 10 个，
所以每次发送前先占用：

\[
60\times 1+10\times 2=80\ \text{micro-USD}.
\]

第一次 attempt 收到 HTTP 500，没有可信 usage；第二次 attempt 收到 HTTP 200，并报告输入 58、输出 4 个 token：

~~~text
客服回复这一次 logical call
├── attempt 1：预留 80 → HTTP 500、usage 缺失 → 费用待确认 80
├── 重试决策：状态允许？请求可重放？还有时间和预算？
└── attempt 2：重新预留 80 → HTTP 200、usage=58/4 → 按实际值结算 66

本地账本合计：80 + 66 = 146 micro-USD
~~~

用户最终只看到第二次返回的答案。账本仍然保留第一次 attempt，因为 HTTP 500 留下的生成和计费情况尚未确认。
因此，本地先保守记录 146 micro-USD；这个数字需要外部对账，不能当作供应商发票。

### 两种“不确定”不要混在一起

这条主线里有两套不同判断：

| 判断 | 它回答什么 | 它影响什么 |
|---|---|---|
| 网络执行结果是否已知 | 上一次发送有没有得到可分类的响应 | 能否进入自动重试决策 |
| 用量和费用是否已知 | 这次发送究竟应结算多少 | 预算预留怎样进入终态 |

第一次 attempt 收到了明确的 HTTP 500，所以网络层知道响应状态，可以继续检查它是否属于允许重试的状态。
但响应没有可信 usage，预算层无法把费用算成 0，只能把整笔 80 记为 `uncertain（费用待确认）`。

如果第一次失败改成写入后的 read timeout，客户端连响应状态都不知道。本仓库默认停止自动重放，并保留费用待确认。
如果能证明连接建立前就失败，请求没有发送，预算才可以取消。

### 每次发送都要有自己的身份

日志和预算 identity 至少使用：

~~~text
logical-call-id
attempt number
canonical request fingerprint
provider/model/API revision
reservation id
provider request id if observed
terminal classification
~~~

因此，attempt 2 不能复用 attempt 1 的 reservation ID。它要重新检查预算，再建立一笔新的预留。

## 重试前回答三个独立问题

### 1. 这个错误允许重试吗？ {#retryable}

目标供应商是否把当前错误或状态码定义为可重试的瞬时失败？这个结论要绑定具体接口与 API 版本。

只看网络库异常名或“属于 5xx”还不够。允许重试的状态、`Retry-After` 和配额规则都可能随接口变化。

### 2. 这次请求适合重新发送吗？ {#replay-safe}

再次发送是否会造成不可接受的重复生成、费用或业务副作用？

即使 Prompt 只要求生成文本，没有写数据库，第二次发送仍可能得到不同答案并产生第二笔费用。如果工具循环已经
执行写操作，必须依靠业务幂等键和效果账本，而不能只看模型调用 ID。

### 3. 上一次网络结果已经明确吗？ {#outcome-known}

客户端能否证明上一次未被 provider 接收，或已经得到明确 terminal？

连接建立前的失败有机会证明请求尚未发送。写入或读取阶段超时通常无法做到这一点。收到 HTTP 响应时，
网络状态可以分类；但是否产生了用量，仍要由响应契约和后续账单回答。

只有重试策略允许、请求适合重放、上一次网络结果足够明确，而且仍有时间和预算时，程序才自动发送下一次 attempt。
预算账本仍要为每次 attempt 独立结算；它不会因为允许重试就把上一笔费用清零。

## 用失败矩阵替代一个 `if retry`

把两种“不确定”放进同一张表，决策会清楚很多：

| 已观察到什么 | 网络结果 | 预算预留 | 默认下一步 |
|---|---|---|---|
| 本地结构或预检失败 | 已知未发送 | 尚未建立 | 修正请求，不重试 |
| 能证明连接前失败 | 已知未发送 | `cancelled` | 在时限内有界重试 |
| 收到非 2xx 响应，没有可信 usage | 响应状态已知 | `uncertain` | 按状态、重放安全和预算决策 |
| 写入、读取或整次 attempt 超时 | 是否处理未知 | `uncertain` | 停止自动重放，进入对账 |
| 2xx 流式响应中途截断 | 收到部分内容，终态未知 | 通常 `uncertain` | 返回不完整终态，不透明重放 |
| 工具成功但回执丢失 | 工具效果未知 | 模型费用另行判断 | 查询业务效果账本 |
| 调用方取消等待 | 只知道本地停止 | 不能自动取消 | 保留记录并查供应商状态 |

程序应根据明确的失败阶段分类，而不是搜索任意异常字符串。远端返回的原始错误正文也可能含有敏感数据，
不能未经处理就写入普通日志。

## 两次 attempt 共用一条总时限

至少区分：

- 等待连接池名额的超时；
- 建立连接、写入和读取超时；
- 流式响应长时间没有新数据的空闲超时；
- 单次 attempt 的上限；
- 整个 logical call 的总时限；
- 调用方取消。

使用单调时钟，让每次 attempt、退避等待和 `Retry-After` 共享同一个总时限。

~~~text
t0 reserve
→ connect/write/read
→ failure classification
→ retry decision
→ backoff
→ next reserve
→ next attempt
→ overall deadline
~~~

每一步都要重新计算剩余时间。业务如果承诺 60 秒，就不能执行三个 30 秒 attempt，再额外等待两次 20 秒。

### Retry-After 怎样处理

如果目标接口支持，`Retry-After` 可以是非负秒数，也可以是 HTTP 日期：

- 没有该字段：使用本地退避规则；
- 格式有效：按服务端要求等待，但仍受本地策略和总时限约束；
- 格式错误：记录这一事实，再回退到本地规则；
- 等待后会越过总时限：停止，而不是提前发送。

指数退避和随机抖动属于调用方策略，不是供应商事实。测试时注入时钟与随机数来源，避免真的等待。

## Cancellation 不证明远端停止

取消本地协程、关闭响应对象或断开 socket，只能证明客户端停止等待。

它不能单独证明：

- 供应商没有继续生成；
- 服务端已经释放资源；
- 用量为零；
- 工具没有执行；
- 计费已经取消。

取消后仍要让本地预留进入明确终态。可以使用供应商请求 ID、用量记录或账单导出来对账；没有足够证据时，
这笔费用继续标记为 `uncertain`。

## 2xx stream 开始后默认不重放

一旦向用户发布了 partial output，透明重试可能导致：

- 文本重复或分叉；
- tool proposal 重复；
- 两次 usage；
- 用户已经消费但本地没有 commit；
- 新一次随机生成不再延续旧文本。

可选策略是：

1. 以 incomplete terminal 结束；
2. 发起新的 logical call，并明确新 identity；
3. 仅使用 provider 正式支持且已验证的 resume contract。

不要让通用 SSE decoder 自行 reconnect。Framing 层不知道业务、usage 和工具 effect。

## 先核对请求要发往哪里，再占预算

发送前完成：

- 精确匹配目标 origin 的允许列表；
- 只允许 HTTPS；
- 禁止 URL 携带用户信息、fragment 或非预期 query；
- 默认关闭重定向；
- 明确代理、证书、DNS 和出站网络规则；
- 认证信息来自 secret manager 或受控环境注入；
- 请求正文和 header 使用确定的序列化规则，并拒绝无法表示的值。

如果 URL 或本地预检已经失败，请求根本不应发送，此时也没有必要先占用预算。密钥不能放进 URL 或错误消息。

收到成功状态后仍要限制内容类型和响应字节数。JSON 解析要拒绝重复字段、NaN、Infinity、错误的顶层类型和
不符合 Schema 的对象。如果先把完整正文读入内存、再检查大小，下载过程本身仍可能占用过多内存。

## 为什么发送前就要预留预算

只在响应后累加 usage 会产生并发超支。若十个请求同时看到剩余预算，每个都可能各自发送到上限。

对 attempt \(i\)，发送前可保守预留：

\[
R_i=\widehat T_{\mathrm{in}}^{(i)}
+T_{\mathrm{out,max}}^{(i)}.
\]

费用估计为：

\[
\widehat C_i
=C_{\mathrm{in}}(\widehat T_{\mathrm{in}}^{(i)})
+C_{\mathrm{out}}(T_{\mathrm{out,max}}^{(i)})
+C_{\mathrm{other,max}}^{(i)}.
\]

发送前的输入 token 数通常只是根据目标 tokenizer 和模板得到的估算。真实接口还可能分别统计缓存、推理或工具用量。
价格也可能受服务层级、币种和税费影响，所以计价规则必须带上核对日期。

预留金额只是本地风险上限，不是实际用量，更不是发票。

## 每个 reservation 只有一个终态

~~~mermaid
stateDiagram-v2
    [*] --> reserved
    reserved --> cancelled: proven never sent
    reserved --> settled: complete trusted usage
    reserved --> uncertain: sent / usage unknown
    settled --> [*]
    cancelled --> [*]
    uncertain --> [*]
~~~

- **Cancelled**：有结构化证据证明 attempt 未发送。
- **Settled**：严格解析到可信 usage。
- **Uncertain**：可能已发送，但 outcome 或 usage 不足。

若 actual usage 超过 reservation，必须先记录已发生的真实值，再触发 post-call breach。为了让 cap 看起来没超而截断 ledger，会让审计失真。

### Retry 要重新 reserve

每个 attempt 都建立独立 reservation：

~~~text
call-42:attempt:1 → uncertain
call-42:attempt:2 → settled
logical call total  → sum of both terminal amounts
~~~

第一次的费用若无法证明为零，就不能被第二次成功覆盖。Hard limit 也必须在第二次真正发送前重新判断。

注意，这里的 `uncertain` 是**预算终态**。示例中的 HTTP 500 已经给出可分类的网络响应，所以它仍能进入
重试决策；未知的是这次发送最终产生了多少用量。

## 本地账本无法和远端组成一个事务

SQLite 事务可以让同一数据库文件的写入者依次修改账本，却无法与供应商的生成和计费一起原子提交。

进程崩溃后，仍在活动状态的预留继续占用额度，是更保守的选择。只根据 TTL 自动释放，会把“本地进程消失”
错误解释成“供应商没有收到请求”。

跨机器或跨区域部署需要共享、可持久化的配额服务。不确定窗口仍要依靠供应商请求 ID、用量或账单导出，
必要时由人工对账。

同理，业务 outbox 和幂等键可以降低工具重复执行的风险，却不能单独证明远端效果只发生一次。

## 成本指标使用 task 分母

只报平均请求费用会奖励廉价失败。把“每个验证成功任务的费用”记作 CPT，并令 (V_i=1) 表示任务 (i)
通过业务验证：

\[
\mathrm{CPT}=\frac{\sum_i C_i}{\sum_i \mathbb 1[V_i=1]}.
\]

在客服示例中，一个任务最终通过验证，本地保守费用是 146 micro-USD。只报告第二次成功 attempt 的 66，
会把第一次可能产生的费用藏起来。

同时报告：

- 所有 attempt 的费用；
- 只看成功 attempt 时的费用；
- 重试把调用和费用放大了多少；
- 费用待确认的预留占比；
- 缓存、推理和工具用量；
- 价格快照、币种、税费、服务层级和抵扣；
- 与账单导出之间的差额。

没有账单导出对账时称为估算，不称为发票成本。

## 生产代码怎样分层

不要把重试、供应商 SDK 和业务工具都塞进一个 `call_model()`。可以按下面的职责拆开：

| 层 | 只负责什么 |
|---|---|
| 业务策略与工具运行时 | 发起 logical call，判断结果能否用于业务 |
| 重试与逐 attempt 预算 | 决定是否再发送，并管理每笔预留 |
| HTTP/SSE 传输 | 目标地址、时限、字节与取消 |
| 供应商解析器 | 验证 JSON 或事件顺序，保留原始协议语义 |
| 供应商 adapter | 把已验证对象映射成共同业务类型 |
| 共同类型 | 表达稳定的请求、输出项、终态、用量和身份 |

共同类型不应包含供应商 SDK 的响应类，业务代码也不应直接读取这些类。这样，解析器可以离线回放；SDK 或
API 升级的差异会留在 adapter 内，真实联网测试则可以单独开启并限制费用。

## 安全和发布工件

普通日志只保存稳定的错误类别、状态码、相对时间、脱敏请求 ID、重试决策和工件版本。

不要默认保存：

- API key 或认证 header；
- 原始 Prompt 与响应；
- 密钥、个人信息与完整工具结果；
- 供应商返回的不透明推理状态；
- 任意远端异常文本。

内部轨迹与公开工件要分开保存。公开输出只允许预先审过的字段，再单独检查密钥、个人信息、版权、同意范围和用途。

## 从离线验证走向真实小流量测试

真实测试会访问供应商并可能产生费用，因此必须由操作者显式开启。第一轮先限制：

1. 精确允许的 origin；
2. 模型和 API 版本；
3. 请求数、并发和输出上限；
4. 单次 attempt 与总费用上限；
5. logical call、单次 attempt 和流式空闲时限；
6. 禁止会产生真实副作用的工具；
7. 工件脱敏规则与保存期限；
8. 供应商用量和账单对账方式。

分层记录：

~~~text
DNS、TLS 与 HTTP 可以连接
→ 认证成功
→ 非流式响应通过结构校验
→ 流式响应出现完整终态和用量
→ 主动取消后的远端状态得到观察
→ 错误与限流行为符合目标接口
→ 本地记录与账单导出完成对账
~~~

每一层都回答不同问题。一次成功调用只证明当时那个账号、模型和输入可以运行，不能证明生产 SLO。

## 故障定位顺序

### 解析错误

先保存原始响应及其内容摘要，再检查 Content-Type，并按 JSON 规范处理重复字段、非法数值、schema revision
和未知字段。只有解析成功后才能进入业务逻辑，解析失败的响应应单独留存和排查。

### 流式重复或缺字

先检查 UTF-8 字节边界和 SSE 分帧，再看事件身份、内容块编号和增量顺序。最后核对完成事件，以及客户端是否
擅自重新连接。不要一开始就把问题归因于模型重复生成。

### 重试风暴

检查哪些状态被允许重试、`Retry-After` 是否生效、逻辑调用是否共用总时限，以及网络结果未知时有没有停止。
还要看所有调用方合计的并发，不能只把退避时间调大。

### 预算与账单不一致

从逻辑调用开始，逐次找到 attempt、预算预留、供应商请求 ID，最后对到用量与账单导出。检查差额究竟来自
发送前估算、额外用量分类、失败 attempt，还是进程崩溃留下的不确定窗口。

### 工具重复执行

查询业务效果账本。逐项核对调用者、目标资源、工具、规范化参数、策略版本、幂等键和执行回执。

## 一个可运行的故障实验

先运行本章开头的两次 attempt。数据库路径必须是一个尚不存在的新文件：

```powershell
New-Item -ItemType Directory -Force artifacts/cloud-api | Out-Null

python projects/cloud-api-contracts/budgeted_retry_demo.py `
  --database artifacts/cloud-api/reliability-walkthrough.sqlite
```

程序使用 `httpx.MockTransport`，不会访问供应商。输出中应该出现两条不同的 reservation ID，事件顺序为：

```text
logical-call:attempt:1  reserved → uncertain
logical-call:attempt:2  reserved → settled
```

确认这条主线后，再用 MockTransport 或本地模拟服务构造：

1. 能证明发生在连接前的失败；
2. 写入后的读取超时；
3. 非 2xx 响应携带 `Retry-After`；
4. 2xx 流式响应发布部分文本后截断；
5. 第一次费用待确认、第二次成功；
6. 工具效果成功但回执丢失。

每个场景都先预测五件事：错误是否允许重试、请求能否重放、网络结果是否明确、预算进入什么终态，以及用户
最终会看到什么。预测完成后再运行对账。

仓库用于检查解析规则的固定样例、逐 attempt 记录、命令与当前适用范围见
[云 API 证据台账](../evidence/cloud-api-controls.md)。

## 常见错误

- 对所有 429/5xx 或网络异常自动重试。
- 把读取超时当成“已经证明请求没有发送”。
- 一次 logical call 只预留一次预算，内部却发送多个 attempts。
- 客户端取消后立即释放费用，或认定工具没有执行。
- 2xx 流式响应截断后悄悄重新连接。
- 用 SQLite 事务声称远端计费只会发生一次。
- 只报告成功请求的平均成本，忽略失败和费用待确认的 attempts。
- 用离线 MockTransport 结果声称真实供应商的计费和取消已经验证。

## 面试时怎样回答

面对“怎样设计 LLM API 重试”，先回答三问：错误是否允许重试（retryable）、请求是否适合重放（replay safe），
以及上一次网络结果是否明确（outcome known）。

然后说明每次 attempt 都有独立的预算预留和终态。已经交付部分文本的流式响应不透明重放；工具副作用通过
outbox 和效果账本对账。最后再把本地记录与供应商账单核对。

这个回答比“指数退避加随机抖动”更完整，因为它覆盖了真正的重复生成、费用和副作用风险。

## 自测

1. 为什么同一个 logical call 的第二次发送需要新的预算预留？
2. 哪些失败阶段可能证明请求尚未发送？哪些通常不能？
3. 客户端取消后，为什么仍在活动状态的预留不能自动释放？
4. 为什么已经交付部分文本的 2xx 流式响应默认不应透明重放？
5. 每个验证成功任务的费用，比平均请求费用多揭示了什么？

## 继续学习

- [实验 0C](../practice/labs/lab-0c-cloud-budget.md)：逐 attempt 预算实验。
- [Cloud API 项目](../practice/projects/cloud-api-contracts.md)：会检查输入结构的 adapters、SSE 与 retry 验证程序。
- [Agent Runtime](../applications/agent-runtime.md)：tool effect、outbox 和 reconciliation。
- [生产检查表](../practice/production-checklist.md)：发布、观测和回滚。
- [云 API 证据台账](../evidence/cloud-api-controls.md)：具体策略、固定样例和未验证范围。
