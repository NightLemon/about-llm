# 服务与可观测性

## 先定义 SLO

服务目标应按流量类型分层，例如聊天重视流式 TPOT，批量抽取重视吞吐，Agent 重视工具链总成功率。常见 SLI：

- 可用性、请求成功率、超时率；
- p50/p95/p99 TTFT、TPOT 和端到端延迟；
- 输入/输出 tokens、并发、队列长度、batch 大小；
- tokens/s/GPU、GPU 显存和利用率；
- 每成功任务成本，而非只看每 token 价格；
- 格式合法率、引用正确率、安全拦截与业务质量。

成功请求的 latency percentile 是条件统计，不能替代 availability。每轮同时记录 attempted、success、timeout、429、5xx、cancelled；无成功请求时 p95 不应伪报为 0。Offered QPS、attempted RPS 与 successful RPS 口径不同，容量报告必须标明分子、分母与时间窗口。

负载生成器的 concurrency semaphore 也是队列。只从取得槽位后开始 TTFT/E2E 计时，会隐藏 client queue；同时记录 workload `offered_at`、HTTP dispatch、首 token 与 terminal。Client queue 不等于服务端 queue，快速 429 的 terminal latency 很低也不等于体验达标，因此 queue/latency 必须与 success rate 联合门禁。

Closed-loop worker 通常在上一请求完成后才发下一条，服务变慢会自动降低 offered load；constant/Poisson open-loop 则按外生 schedule 继续到达，更适合找饱和 knee。open-loop 也必须报告 generator lag：把 scheduled timestamp 写成 `offered_at` 可让迟到进入 client queue，却不证明发生器实际按时执行。有限预生成任务、无限流量服务和生产 arrival distribution 是三种不同证据。

## API 设计

采用稳定请求 id、幂等键、超时、取消、速率限制和明确错误码。流式连接中断应停止后端生成，避免“用户已走、GPU 仍算”。记录模型版本、tokenizer/chat template、采样参数、adapter、检索和工具版本。

Stop string 是独立的增量文本协议：它可能跨 token/event/UTF-8 byte chunk，客户端必须暂存仍可能成为 stop 的最长 suffix，不能先展示后撤回。明确是否返回 stop、overlap/priority、大小写/Unicode normalization、usage 与 finish reason。客户端本地截断只改变展示，不证明服务端停止 decode、释放 KV 或停止计费；需要 cancellation/terminal trace 关联验证。

重试只用于瞬时失败，并使用指数退避与抖动。对非幂等工具调用（付款、发邮件、创建资源），不能盲目重试；需执行 id、去重和状态查询。

## 路由与降级

可按任务难度、语言、上下文长度、合规要求和负载选择模型。小模型处理分类/抽取，大模型处理开放推理。降级策略包括减少候选、缩短上下文、关闭非必要工具、切备用模型或转人工；不能悄悄降低安全检查。

## 缓存

- 响应缓存：输入完全相同且结果允许复用。
- 语义缓存：相似问题复用，风险更高，需租户/权限/时效隔离。
- 前缀/KV 缓存：复用计算。
- 检索缓存：需结合索引版本和 ACL。

缓存键必须包含所有影响结果的版本与权限上下文；这些字段由认证/策略层从可信状态生成，不能接受模型或请求体自报。Prefix/KV cache 至少绑定 tenant、安全可见域、authorization/policy、model/tokenizer/template/adapter、position/RoPE、KV dtype 和 exact token prefix。Fingerprint 只做 bucket index，命中仍比较完整字段；否则 hash collision、过期 ACL 或同文本不同 tokenization 都可能错误复用。敏感数据设置 TTL、加密和删除机制，评估 hit/miss timing side channel。随机生成或个性化答案通常不适合直接响应缓存。

## 背压与过载保护

队列无限增长只会把失败变成超时。设置最大排队时间、并发与 token 预算；按输入长度估计工作量；超载时快速拒绝或降级。长上下文请求可单独队列，避免阻塞短请求。

429 可以是正确的过载保护，但仍是调用方未成功完成的 attempt。是否满足 SLO 取决于流量是否在合同配额内；不能把 429 从 availability 分母删除，也不能把恶意/超配额流量无条件混入正常租户 SLO。

## 追踪

一次请求的 trace 应串联：网关 → Prompt 构造 → 检索 → 重排 → 模型 → 工具 → 验证 → 响应。保存必要元数据和哈希，敏感原文按最小化原则处理。指标用于趋势，trace 用于单次诊断，日志用于事件细节，三者不能互相替代。

## 质量监控

线上缺少即时标签，可用代理指标和抽样人工审查，但要防止代理被优化歪。持续回放版本化评测集，监控输入分布漂移、语言/长度变化、拒答率、用户纠错与升级人工率。模型或 Prompt 更新采用 shadow/canary，保留快速回滚。

## 成本

总成本包含模型推理、Embedding/重排、检索存储、工具、网络、可观测、人工审核和失败重试。优化“每成功任务成本”：更便宜但错误率高的模型可能因重试和人工处理更贵。

## 事件响应

预先定义安全泄露、错误工具操作、模型不可用和质量大幅回归的负责人、分级、止损、证据保留、通知和复盘。回滚不仅是模型权重，也包括 Prompt、索引、工具 schema 和策略配置。

## 自测

1. 为什么 p50 延迟优秀仍可能意味着体验很差？
2. 语义缓存为什么必须带权限与时效边界？
3. 如何防止客户端断流后继续浪费 GPU？
