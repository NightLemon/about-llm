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
- [ ] Provider response 中 visible text、tool、reasoning summary、opaque reasoning/signature 与 unknown block 分别建模；未知 block 不静默丢弃或无条件回传。
- [ ] 公开 trajectory/fixture 由字段 allowlist 重新生成，`opaque_reasoning_block_count == 0`；只清理 visible text 不算脱敏完成。
- [ ] Opaque state 绑定 authenticated subject、tenant、session/branch、predecessor、model audience、expiry、key status 与 replay identity；这些值不接受 Prompt/body 自报。
- [ ] 外部下载或共享的 Agent trajectory 视为不可信序列化状态，不直接恢复模型上下文、工具权限或副作用执行。
- [ ] Reasoning artifact 泄露有隔离、credential/session/artifact/key 撤销、旧格式退役、下游副本删除、通知和无真实秘密回归 fixture。

## 模型与 Prompt

- [ ] 模型、tokenizer、chat template、adapter、Prompt 均版本化。
- [ ] Adapter 发布绑定 immutable base revision/content digest；路径、模型家族名或 identity string 不足以认证 exact base，明确是框架自身校验还是调用方在加载前强制仓库 manifest。
- [ ] 区分“目标 checkpoint backward/adapter reload 已执行”与“训练改善”：保存 assistant label boundary、trainable/frozen 参数账本、gradient/nonzero adapter、前后 loss 与 held-out 指标；单步 loss 上升或无 held-out 时不得写成收敛/质量提升。
- [ ] 明确训练 estimand 是 token mean、sequence mean 还是 task-weighted mean；按整个 optimizer-update window 与所有 data-parallel ranks 记录 loss numerator/effective-token count。默认 DDP gradient mean 下的 `D/N`、漏 world size 的 `1/N` 和 rank-local mean 必须与单设备 full-batch 对照，不能用 `loss/accumulation_steps`、batch size 或 max length 猜分母。
- [ ] 分别验证 reducer sum/mean、count collective、accumulation + `no_sync`、AMP scale/unscale、完整归一化后 clipping、optimizer/scheduler step 边界；`no_sync` 必须包住非末批的 forward+backward。overflow 时整个 window 的 optimizer/scheduler 都按同一 global decision 跳过，不能因 optimizer 成功也常返回 `None` 而错误推进。共识必须在所有可能产生 non-finite 的 gradient transform 之后、任何 optimizer mutation 之前；默认 reduction 前 Inf 可能已传播，但 custom hook、条件参数、post-reduction transform/故障仍需目标路径验证。极小 CPU/Gloo DDP+AMP control 也只有单参数/单 bucket、人为 Inf 和显式 scale policy，不能外推为随机层、native distributed scaler、FSDP/ZeRO、GPU、多节点或目标 Trainer 正确。
- [ ] 已比较 trained adapter、独立 base+reloaded adapter、in-memory merged 与 disk-reloaded merged 的 logits/生成；量化前后另测，不能用 CPU FP32 merge 容差替代 QLoRA 导出验证。
- [ ] 发布 manifest 覆盖完整目录而非只列 weight；额外/缺失文件、symlink、路径穿越、duplicate/non-canonical manifest、资源上限、size/hash 与 config/tokenizer/template 语义漂移均 fail closed。
- [ ] Verify 与实际 load 使用同一不可变发布快照；仅按路径先验 hash 再重新打开存在 TOCTOU，需 ACL、原子目录切换、lease/content-addressed handle 或等价控制。
- [ ] 导出 artifact 显式覆盖或引用 config、tokenizer payload、未量化 state、shard index 与 runtime layout；只有 identity/manifest 的教学 bundle 不冒充完整 checkpoint。
- [ ] 明确是 inference checkpoint 还是可恢复训练的 checkpoint；后者还需 optimizer、scheduler、scaler、RNG、sampler/data cursor 与分布式状态的一致快照。
- [ ] 用 scale-sensitive overflow/finite 边界验证 GradScaler state 真的被保存和恢复；只比较参数或普通 finite loss 可能看不出漏状态。进程内 `state_dict` replay 不等于文件 checkpoint、进程退出/重启或 crash recovery。
- [ ] overflow 跳过 optimizer update 时同步决定 scheduler 是否前进，并以 optimizer 的真实 update receipt 验证；`GradScaler.step()` 对常见 optimizer 返回 `None`，不能把返回值直接当通用 `did_step`。仓库 tiny AdamW control 用所有 per-parameter `step` 一致递增作观测，只对该 fixture 成立。
- [ ] 恢复范围列出实际消费的 Python/NumPy/Torch CPU/CUDA RNG、DataLoader worker/prefetch、gradient accumulation position/未提交梯度和 sharded optimizer；未使用可以省略，使用却未保存则不得声称 exact resume。对多 worker loader 分开记录 sampler-emitted、main-loop-consumed 与 optimizer-committed cursor：prefetch queue 中“已发出但未消费”的 index 不能按已训练跳过；已 consumed/backward 但未 committed 的窗口若没保存 gradients，必须从 commit boundary 重放；若保存，则需绑定 pending IDs、position、loss divisor、逐参数 gradients、RNG 和 base identity。多 artifact checkpoint 还要定义 manifest-last completeness gate：closed manifest 绑定文件名/schema/size/hash 与发布顺序，加载时先验 complete，再对实际反序列化 bytes 重查 identity；base-only 是否仍可 replay 必须单独定义。fresh worker RNG 需恢复公开状态或改为绑定 dataset/transform revision、epoch/visit、sample ID 的完整 stateless key。仓库 2-worker stochastic optimizer control 已证明 commit-boundary RNG replay 与完整 gradients+crash-RNG sidecar resume 都可 bit-exact；一个隔离负例在正确 RNG、相同 5 steps/LR 下漏 gradients/sample 并漂移，另一个在 gradients/ledger/steps/LR 完整时只因 RNG 错位而漂移。四种 publication fault snapshots 会 fail closed。但它仍未保存 queue/worker/Python/NumPy/CUDA RNG state，也未证明 directory `fsync`/断电、来源认证/不可变快照或 sample—optimizer—base+sidecar+manifest 原子事务。
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
- [ ] No-evidence/unauthorized-evidence 分支有 **pre-generation 0-call** 不变量；监控分别记录 policy callback、framework `generate()`、provider attempt 与 outcome/usage，不能用本地 method count 推断远端取消或零计费。
- [ ] Publication decision 分离 audit/public projection；被 reject 的 raw output、未知引用和 finding text 只进受限审计面，HTTP/stream/error/APM 不得因直接序列化内部 decision 而重新泄露。
- [ ] Answer artifact/trace 含授权正文、身份和 query 时进入受控存储，并与普通 metrics label、公开日志和错误响应隔离。

## Agent 与工具

- [ ] 工具单一职责、严格 schema、超时和稳定错误码。
- [ ] 身份、权限和参数由执行层验证，不信任模型决定。
- [ ] 不可逆/外部副作用有具体人工确认、幂等和恢复策略。
- [ ] 步数、时间、token、费用和重复动作均有限制。
- [ ] 外部内容按不可信处理；秘密不进入模型上下文。
- [ ] MCP/A2A adapter 固定协议版本、transport、官方 schema/SDK revision 与 negotiated capability；discovery/card/schema-valid/remote completed 都不能直接变成本地授权或 verifier success。
- [ ] 分开验收 SDK、transport 与 negative-control matrix：in-memory SDK control 不冒充 stdio/HTTP；official-SDK stdio/HTTP 即使同时执行 SDK+具体 transport，也不冒充另一 transport、未触发的畸形 framing/body/header、forced shutdown/cancel、resumption 或 conformance；手写 stdio/HTTP control 不借用官方 SDK 身份。SDK schema-invalid 在 handler 前拒绝也不意味着 unknown tool、资源归属或授权无需应用层 gate。
- [ ] MCP stdio 的 stdout 只传协议消息，日志走 stderr；每个 request 有 deadline/cancel/max timeout 与 EOF→terminate→kill shutdown。Streamable HTTP 独立验证 Origin、认证、session/version、JSON/SSE、显式 cancel、DELETE 与 DNS rebinding；测试编排的 readiness/shutdown token 和本机 Bearer gate 都不能冒充 MCP auth、OAuth 或业务授权，loopback control 也不能代替 TLS、代理和远程测试。
- [ ] Tool result、resource、prompt、remote artifact 和原始 transcript 进入不可信数据/敏感日志路径；对外报告采用 allowlist projection，hash 只做 identity，不冒充认证或脱敏。

## 评测与安全

- [ ] 评测集含真实、边界、对抗、无答案和多语言样本。
- [ ] 报告置信区间与群体/场景切片，不只报平均分。
- [ ] LLM judge 已用人工标注校准。
- [ ] 提示注入、越狱、数据外传、权限越界和供应链已红队。
- [ ] 安全误拦和漏拦同时评测；结果进入回归集。

## 可靠性与运维

- [ ] 定义可用性、TTFT/TPOT、端到端延迟和质量 SLO。
- [ ] 服务验收把 checkpoint bytes、实际 loader class/dtype/parameter count、独立进程/socket、request/response contract 与 server-side generation audit 串起来；model-id 字符串、mock transport 或 client 200 不冒充目标权重执行。
- [ ] 区分 incremental decode streaming 与 generation-complete 后分块发送；SSE `[DONE]`/usage 通过不证明断流取消、KV 释放或停止计费，client/server trace 要关联同一 request id。
- [ ] 断连验收分开记录 transport disconnect、ASGI/task cancellation、backend token 停止、blocking thread/process/kernel 终止、scheduler removal、KV/GPU release 与 billing reconciliation；`CancelledError` 传播不能替代后四项证据。
- [ ] Threaded generation 的 cooperative token/event 与 `StoppingCriteria` 有明确检查频率、join timeout 和 worker-recycle fallback；人为 pause/control 通过不外推为未修改调用、不可中断 kernel 或 allocator release 已验证。
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
