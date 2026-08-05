# LLM 系统设计题

## 通用回答框架

1. 澄清用户、流量、数据、风险和成功指标。
2. 给最小基线，不默认一定需要 Agent 或微调。
3. 画在线链路、离线数据链路与信任边界。
4. 定义接口、数据模型、权限和版本。
5. 做容量估算：QPS、token、并发、显存、延迟和成本。
6. 定义离线/在线评测、观测和发布门禁。
7. 分析故障、降级、恢复、安全和回滚。
8. 最后才讨论高级优化和未来扩展。

## 题一：企业知识库问答

### 需求澄清

文档类型与规模、更新频率、用户/租户、ACL、语言、是否需引用、允许延迟、无答案行为和数据驻留。成功指标至少包含检索覆盖、答案忠实度、引用、权限、p95 延迟与每成功回答成本。

### 设计

离线：对象存储 → 解析/OCR → 结构切分 → 内容哈希/去重 → metadata/ACL → sparse+dense index → 版本发布。

在线：身份 → query 改写 → ACL filter → BM25+dense 召回 → RRF/rerank → 去重与 token budget → 带 source id 生成 → claim/citation 验证 → 响应。

### 关键取舍

- 小 chunk 检索、父 chunk 返回；
- BM25 保留型号/错误码；
- 索引更新用蓝绿版本，cache key 包含 index version 与 ACL；
- 无证据就 abstain，不用参数记忆补齐；
- 权限在检索前执行，生成器永远看不到无权文档。

### 容量估算

以 QPS、平均/峰值输入 token、输出 token、检索/重排/模型延迟拆分。Little's Law 粗估并发等于到达率乘平均服务时间；GPU 容量再用目标 workload 压测，不从参数量直接猜。

### 故障

解析错、索引延迟、召回为空、证据冲突、模型超时、引用错误和跨租户缓存。每类有指标、降级和回滚；删除请求传播到原文、索引、cache 与日志。

## 题二：能发邮件和建工单的 Agent

### 最小方案

优先固定 workflow：理解请求 → 收集必填字段 → 生成草稿 → 用户确认 → 执行 → 查询结果。模型不直接持有邮件/工单凭证。

### 执行契约

ToolCall 包含 call_id、工具、结构化参数；执行层校验身份、ACL、schema、速率和业务状态。副作用审批绑定参数指纹和过期时间。幂等 ledger 与业务事务/Outbox 防止重放。

### 安全

邮件、网页和工单正文是不可信数据；不把其中指令提升权限。秘密在 credential broker；工具最小权限；附件扫描；出站域名和收件人策略；高风险动作二次确认。

### 恢复

每步状态持久化。超时后先按 call_id 查询外部系统，不盲目重试。达到步数、费用、时间或无进展阈值转人工。审计记录 proposed、approved、executed 和 result。

### 评测

任务成功、字段正确、步骤数、恢复、重复副作用、越权、注入抵抗、延迟和成本。真实副作用在模拟环境回归。

## 题三：单卡多租户 LLM 服务

### 需求

模型/量化、显存、上下文、平均/峰值 QPS、交互或批处理、TTFT/TPOT SLO、租户优先级和数据隔离。

### 服务

API gateway 做认证、配额、最大 token 与取消；scheduler 做 continuous batching、长度感知和公平；engine 管理权重、Paged KV 和 prefix cache；observer 记录队列、TTFT、TPOT、tokens/s、KV usage、preemption 与错误。

### 容量

显存 = 权重 + KV + workspace。KV 按层数、长度、KV 头、head dim、dtype 和并发估算，再保留碎片/峰值余量。分别扫描 prefill-heavy 和 decode-heavy workload。

### 隔离

prefix/response cache 包含 tenant、model、adapter、token ids 与权限版本。敏感 prompt 不跨租户共享。adapter 动态加载设显存与切换限制。

### 过载

有限队列、最大排队时间、按 token 估工作量。超载时快速 429、路由小模型或降低非必要生成预算；不悄悄跳过安全检查。

### 发布

固定模型 revision、runtime 与 kernel；shadow/canary；质量/延迟联合门禁；保留上一权重、template、generation config 与 engine image 回滚。

## 面试官常见追问

- 为什么不用微调或为什么不用 RAG？
- 数据更新和删除如何传播？
- p99 超时、GPU OOM、provider 429 怎么降级？
- 如何证明没有跨租户泄漏？
- 指标变好是否统计显著、是否过拟合测试集？
- 当模型升级但 Prompt/索引没变，为什么仍可能回归？
- 成本估算包含哪些被忽略的重试、工具和人工费用？
