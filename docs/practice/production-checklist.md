# 生产检查表

这是一份发布前核对表，不代替按业务风险设计的评审。

## 目标与责任

- [ ] 用户问题、成功指标和不可接受失败已书面定义。
- [ ] 已证明 LLM 相比规则、搜索或小模型有实际价值。
- [ ] 产品、模型、数据、安全、隐私和运维责任人明确。
- [ ] 高风险决策有合格人工复核和申诉路径。

## 数据与隐私

- [ ] 数据来源、许可、用途、保留期和删除流程有记录。
- [ ] 敏感数据最小化，传输/存储加密，访问按最小权限。
- [ ] 日志、缓存、向量库和第三方供应商纳入数据流图。
- [ ] 跨租户 ACL 在检索、缓存、工具和观测各层测试。

## 模型与 Prompt

- [ ] 模型、tokenizer、chat template、adapter、Prompt 均版本化。
- [ ] Adapter 发布绑定 immutable base revision/content digest；路径、模型家族名或 identity string 不足以认证 exact base，明确是框架自身校验还是调用方在加载前强制仓库 manifest。
- [ ] 已比较 trained adapter、独立 base+reloaded adapter、in-memory merged 与 disk-reloaded merged 的 logits/生成；量化前后另测，不能用 CPU FP32 merge 容差替代 QLoRA 导出验证。
- [ ] 发布 manifest 覆盖完整目录而非只列 weight；额外/缺失文件、symlink、路径穿越、duplicate/non-canonical manifest、资源上限、size/hash 与 config/tokenizer/template 语义漂移均 fail closed。
- [ ] Verify 与实际 load 使用同一不可变发布快照；仅按路径先验 hash 再重新打开存在 TOCTOU，需 ACL、原子目录切换、lease/content-addressed handle 或等价控制。
- [ ] 导出 artifact 显式覆盖或引用 config、tokenizer payload、未量化 state、shard index 与 runtime layout；只有 identity/manifest 的教学 bundle 不冒充完整 checkpoint。
- [ ] 明确是 inference checkpoint 还是可恢复训练的 checkpoint；后者还需 optimizer、scheduler、scaler、RNG、sampler/data cursor 与分布式状态的一致快照。
- [ ] 恢复范围列出实际消费的 Python/NumPy/Torch CPU/CUDA RNG、DataLoader worker/prefetch、gradient accumulation position/未提交梯度和 sharded optimizer；未使用可以省略，使用却未保存则不得声称 exact resume。
- [ ] 数据 snapshot 不嵌入 checkpoint 时，恢复前校验 shape、ordered content fingerprint、tokenizer/template/chunker 与 sampler identity；仅有路径或行数不足以绑定数据。
- [ ] 已做“不中断到 N”对“在 K 写盘、进程退出、重载到 N”的对照，逐步比 sample/LR/loss，终点比参数、optimizer、scheduler、stream 与 RNG；loss 接近不替代状态等价。
- [ ] Loader 拒绝 duplicate/unknown 字段、非 canonical manifest、truncation/trailing、tensor name/order/offset/length/digest 漂移，并在分配前限制总文件、manifest、tensor 数和单 tensor 大小。
- [ ] Artifact 完整性 hash 与来源认证分开；需要可信来源时使用签名/受控发布链，不能把 unkeyed SHA-256 当签名。
- [ ] Architecture revision 对应经过测试的 loader/forward 实现；字符串 revision 或 config JSON 本身不嵌入代码，也不证明未来实现兼容。
- [ ] 上下文与输出 token 有硬预算，截断策略可解释。
- [ ] 结构输出使用 schema 并在应用层校验。
- [ ] 模型更新有兼容性、质量、安全和成本回归。

## RAG

- [ ] 解析/OCR 质量抽检；来源、页码、时间和 ACL 保留。
- [ ] 删除和更新能传播到所有索引与缓存。
- [ ] 召回与生成分别评测，引用蕴含主张。
- [ ] 无资料、证据冲突、过期数据有明确行为。
- [ ] Tenant/principals 从已验证 transport identity 派生；request body、Prompt 或模型输出不能自报权限，ACL 在正文进入 scorer/cache/callback 前执行。
- [ ] Readiness 真正检查所需 store/schema/index snapshot；liveness 不冒充可接流量，数据库路径缺失不能静默创建空库。
- [ ] Async deadline 明确底层 work 是否支持 cooperative cancellation；`wait_for(to_thread(...))` 的 504 不会杀死 thread，permit/lease 必须持有到真实 work 终止或由隔离 worker 回收。
- [ ] In-process semaphore 按 worker/replica 分账；需要服务总并发时使用可验证的全局 admission，不把单 worker 测试外推到多进程/多副本。
- [ ] Answer artifact/trace 含授权正文、身份和 query 时进入受控存储，并与普通 metrics label、公开日志和错误响应隔离。

## Agent 与工具

- [ ] 工具单一职责、严格 schema、超时和稳定错误码。
- [ ] 身份、权限和参数由执行层验证，不信任模型决定。
- [ ] 不可逆/外部副作用有具体人工确认、幂等和恢复策略。
- [ ] 步数、时间、token、费用和重复动作均有限制。
- [ ] 外部内容按不可信处理；秘密不进入模型上下文。

## 评测与安全

- [ ] 评测集含真实、边界、对抗、无答案和多语言样本。
- [ ] 报告置信区间与群体/场景切片，不只报平均分。
- [ ] LLM judge 已用人工标注校准。
- [ ] 提示注入、越狱、数据外传、权限越界和供应链已红队。
- [ ] 安全误拦和漏拦同时评测；结果进入回归集。

## 可靠性与运维

- [ ] 定义可用性、TTFT/TPOT、端到端延迟和质量 SLO。
- [ ] 超时、退避重试、背压、限流、取消和降级经过压测。
- [ ] 容量压测固定 arrival process/seed/长度分布，区分 scheduled offered、dispatch 与 server queue，并监控负载生成器 lag。
- [ ] 非幂等动作不会被自动重试。
- [ ] 云 API egress 使用 exact origin allowlist、HTTPS、redirect/query policy；URL、异常链和 trace 不泄露密钥。
- [ ] 区分 connect 前失败与 write/read 后 outcome-uncertain；取消/超时后有查询或 reconciliation，不盲目重放。
- [ ] 非流式 body acceptance cap 不冒充下载内存上限；流式读取逐 chunk 限制并能提前关闭。
- [ ] Stop string 覆盖跨 token/event/UTF-8 byte、partial-prefix withholding、overlap priority、include/exclude、截断 EOF；客户端截断不冒充服务端 finish reason、usage 或取消完成。
- [ ] trace 串联检索、模型、工具和验证，且日志已脱敏。
- [ ] 模型导出采用临时文件校验后原子发布，并验证文件与 parent-directory durability；exclusive create + file fsync 不能自动排除 partial target 或证明断电原子性。
- [ ] canary、自动门禁、快速回滚和事件响应已演练。

## 成本与可持续性

- [ ] 估算峰值流量、长上下文、失败重试与第三方工具成本。
- [ ] 云调用在发送前按 request identity、目标 tokenizer estimate 与 maximum output 原子 reserve；成功按 reported usage settle，确定未发送才 cancel，unknown usage 保守占用并 reconciliation。
- [ ] Pricing snapshot 绑定 provider/model/revision/checked_at；多 worker 使用 durable atomic quota，并用 billing export 核对 cache/reasoning/tier/税费等本地估值未覆盖项。
- [ ] Crash 后 active reservation 不按 TTL 自动释放；reconciliation 绑定 stable call id、request fingerprint、attempt/request id、usage 与处置审计。
- [ ] 明确本地 quota 与远程调用的非原子窗口；SQLite 只覆盖同一数据库的 writer，不冒充跨区域配额、provider invoice 或 exactly-once billing。
- [ ] Retry 按每个可能计费的 attempt 单独 reserve/terminalize；不能用一次逻辑-call reservation 覆盖多次 replay，也不能用最终成功 usage 抹掉早期未知 attempt。
- [ ] 预算 wrapper 只在有结构化证据证明未发送时 cancel；HTTP error、2xx 缺 usage、partial/cancel/timeout outcome-uncertain 均进入对账。
- [ ] 监控每成功任务成本，不只监控 token 单价。
- [ ] 有模型路由、缓存和批处理策略，并评估质量影响。
- [ ] 记录硬件/能源假设，避免无依据的环境声明。
