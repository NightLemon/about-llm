# 生产检查表：这次发布能不能放量

这张表不是让团队证明“系统绝对安全”，而是帮助发布负责人作出一个可追溯的决定：现在发布、限制范围发布，
还是暂缓发布。

假设你正在上线一个客服助手。它会检索租户文档，调用云模型，必要时发起退款。检查时不要只勾“已经测试”，
而要为每个相关项目写下四样东西：**负责人、证据链接、失败后的动作、当前结论**。其中任何一项说不清，
就不能把复选框当作完成。

```text
变更：客服助手 v1.4 / model-route@7 / index-2026-08-20
范围：5% 内部客服流量，仅 tenant-shop-a
负责人：产品、模型、数据、安全、SRE、值班经理
结论：go / limited-go / no-go
证据：评测报告、压测、红队、回滚演练、数据与权限审查
```

## 先写清这次究竟改了什么

- [ ] 变更单固定了模型、tokenizer、Prompt、adapter、索引、工具、policy 和运行时版本。
- [ ] 目标用户、使用场景、流量范围与成功指标已经写明。
- [ ] 不可接受的失败有具体例子，例如跨租户泄露、重复退款或错误医疗建议。
- [ ] 每类风险有能够停止发布的负责人，而不只是“知会”名单。
- [ ] 回滚对象明确；团队知道回滚模型是否也要回滚索引、Schema、Prompt 或数据库迁移。

## 数据与隐私

- [ ] 数据的来源、许可、用途、保留期和删除路径均有记录。
- [ ] 数据流图包含 Prompt、日志、缓存、向量库、工具结果、第三方 Provider 和离线评测工件。
- [ ] 敏感字段按最小需要进入模型；传输、存储与访问权限符合数据等级。
- [ ] Tenant 与 principal 来自可信身份层，不能由 request body、Prompt 或模型输出自报。
- [ ] 公开日志、trajectory 和 fixture 由字段 allowlist 生成，而不是只对可见文本做字符串脱敏。
- [ ] Provider 返回的 opaque reasoning、signature 和未知 block 有单独策略：不静默丢弃，也不无条件回传。
- [ ] 数据或 reasoning artifact 泄露后，团队能够隔离工件、撤销凭据并清理下游副本。

涉及 reasoning artifact、跨会话 replay 或第三方 trajectory 时，继续完成
[Reasoning Artifact 安全审查](../quality/reasoning-artifact-security.md)。

## 模型、训练与发布工件

- [ ] 评测加载的正是准备发布的 checkpoint，而不是同名模型或训练进程里的临时对象。
- [ ] Adapter 绑定不可变 base revision；merge、量化、写盘和重新加载后均做了目标任务回归。
- [ ] 发布 manifest 覆盖 config、tokenizer、template、权重与 shard index，并拒绝额外、缺失或漂移文件。
- [ ] 完整性 hash 与来源认证分别处理；需要可信来源时有签名或受控发布链。
- [ ] Loader 在分配大块内存前限制文件数、manifest 大小、tensor 数量和单 tensor 大小。
- [ ] 训练改进由 held-out 结果支持；“成功 backward”或“adapter 可以 reload”不写成质量提升。
- [ ] 上下文和输出有硬上限；结构化输出在应用层再次做 Schema 与语义校验。

若本次发布包含训练、恢复或 adapter 导出，继续完成
[单卡微调项目](projects/single-gpu-finetuning.md)和
[PEFT/QLoRA 工程检查](../training/peft-qlora-engineering.md)。

## 训练恢复与分布式专项

这一节只在训练作业、可恢复 checkpoint 或分布式训练发生变化时勾选。

- [ ] Loss 的 estimand 已定义，并在 accumulation 与所有 data-parallel rank 上使用正确分子和分母。
- [ ] `no_sync`、AMP、gradient clipping、optimizer 与 scheduler 的先后顺序经过单卡 full-batch 对照。
- [ ] Overflow 是否跳过 optimizer/scheduler 由全局一致的 update receipt 判断，而不是猜 `step()` 返回值。
- [ ] Checkpoint 明确保存实际消费的 optimizer、scheduler、scaler、RNG、sampler 和数据游标状态。
- [ ] 多 worker 数据加载区分 emitted、consumed 与 optimizer-committed；崩溃后从可证明的边界恢复。
- [ ] “不中断跑到 N”与“在 K 退出并重载到 N”逐步比较 sample、LR、loss 和最终状态。
- [ ] 多文件 checkpoint 只有在 closed manifest 最后发布并验证后，才被视为完整快照。

这里最容易出现“测试通过但结论过大”。CPU/Gloo 的单参数实验不能证明 FSDP、ZeRO、GPU 或多节点路径正确。
完整推导与实验入口见[分布式训练](../systems/distributed-training.md)。

## RAG

- [ ] 解析与 OCR 做过抽检；chunk 保留来源、页码、时间、版本和 ACL。
- [ ] 删除或更新能够传播到索引、cache 和已生成的派生工件。
- [ ] ACL 在正文进入 scorer、cache、callback 或 Prompt 之前执行，并有跨租户负例。
- [ ] 召回、重排、生成、引用和拒答分别评测；“无授权证据”在生成前保持零模型调用。
- [ ] 无资料、证据冲突、资料过期和引用不支持主张时，产品行为已经定义。
- [ ] 内部审计对象与对外答案分开；被拒绝的 raw output 不进入 HTTP 错误、stream 或 APM。
- [ ] Readiness 检查真实 store、Schema 与 index snapshot；缺失数据库不会被静默创建成空库。

用[一次 RAG 请求](../applications/rag-request-lifecycle.md)复查完整链路；部署细节见
[RAG 生产化](../applications/rag-production.md)。

## Agent 与工具

- [ ] 模型只提交 proposal；Schema、资源解析、ACL、审批和预算由可信执行层完成。
- [ ] 不可逆动作向用户展示具体对象、金额与后果，approval 绑定当前 execution identity。
- [ ] Handler 调用前先 claim；timeout 后保留 `pending`，不会因为本地异常就盲目重试。
- [ ] Provider idempotency key、查询接口或补偿路径已经实测；没有恢复协议时转人工处理。
- [ ] Completion 由业务 verifier 建立，不接受模型、远端 Agent 或自然语言状态自报成功。
- [ ] Cache replay 会重新授权；步数、时间、token、费用、重复动作和循环都有硬上限。
- [ ] 秘密由工具代理注入；网页、工具结果和远端 artifact 始终按不可信输入处理。

先用[一次 Agent 退款](../applications/agent-task-lifecycle.md)逐阶段复核，再进入
[Agent Runtime](../applications/agent-runtime.md)检查 pending、outbox 与 recovery。

## MCP、A2A 与远程工具专项

这一节只在新增 transport、SDK 或远程 Agent 时勾选。

- [ ] 固定协议、SDK、Schema 与 negotiated capability；discovery 成功不会自动变成业务授权。
- [ ] 分别测试实际使用的 stdio 或 HTTP 路径；in-memory SDK 测试不冒充 transport 测试。
- [ ] Stdio 的 stdout 只承载协议消息，deadline、cancel 和 EOF shutdown 都有明确状态。
- [ ] HTTP 独立验证 Origin、认证、session/version、JSON/SSE、取消与 session 删除。
- [ ] Tool result、resource、prompt 和 transcript 进入不可信数据与敏感日志路径。
- [ ] Remote `completed` 仍须通过本地 policy 和 verifier；loopback 测试不冒充 TLS、OAuth 或远程互操作。

协议层的证据边界见[Agent 互操作](../applications/agent-interoperability.md)。

## 评测与安全

- [ ] 评测集覆盖真实、边界、对抗、无答案和相关语言，并避免按单条样本随机切分近重复数据。
- [ ] 主指标、失败切片、置信区间和样本分母同时报告；平均分不会掩盖高风险失败。
- [ ] LLM judge 用人工样本校准，并报告与人工分歧最大的切片。
- [ ] Prompt injection、越权、数据外传、滥用和供应链风险已经红队。
- [ ] 安全漏拦和误拦一起进入回归，且每个门禁都有明确阻断阈值与 owner。
- [ ] 报告区分真实线上观测、离线 fixture、模拟故障和作者声明，避免把局部实验外推到生产。

评测设计见[评测方法论](../quality/evaluation-methodology.md)，风险分级与处置见
[安全](../quality/safety.md)。

## Serving、可靠性与恢复

- [ ] SLO 分开定义可用性、TTFT、TPOT、端到端延迟、质量与错误预算。
- [ ] 服务验收串起 checkpoint bytes、实际 loader/dtype、独立进程/socket 和 server-side generation trace。
- [ ] 压测固定到达过程、seed 与长度分布，并区分 load generator lag、dispatch queue 和 server queue。
- [ ] Timeout、退避、限流、背压、降级和非幂等重试策略经过故障演练。
- [ ] Trace 能用同一 request ID 串起检索、模型、工具、verifier 和最终用户投影。
- [ ] Canary、自动门禁、回滚和事件响应已经由当班人员演练。
- [ ] 模型工件先写临时位置、验证后原子发布；文件与 parent directory 的 durability 都有定义。

### 断连和取消不能只看一个异常

若服务支持 streaming 或后台生成，还要分别观察：

- [ ] Client disconnect 与 transport/ASGI task cancellation 已记录。
- [ ] Backend 确实停止产出 token，而不只是前端停止读取。
- [ ] Blocking thread、process 或 kernel 有 cooperative stop 或隔离回收路径。
- [ ] Scheduler 移除请求，KV/GPU 资源得到释放。
- [ ] Provider usage 与账单得到 reconciliation；本地 `[DONE]` 或 `CancelledError` 不冒充停止计费。

完整请求生命周期见[推理请求](../systems/inference-request-lifecycle.md)和
[Serving](../systems/serving.md)。

## 云 API、成本与配额

- [ ] Egress 使用精确 origin allowlist、HTTPS 和 redirect policy；URL、异常链与 trace 不含密钥。
- [ ] 每个可能计费的 attempt 在发送前原子 reserve，成功后按 Provider usage settle。
- [ ] 只有结构化证据证明“没有发送”时才释放 reservation；timeout 和部分响应进入对账。
- [ ] Crash 后的 active reservation 不按 TTL 猜测释放，而是绑定 call、attempt 与 request ID 核对。
- [ ] Pricing snapshot 固定 Provider、模型、revision 与检查时间，并用账单导出校正本地估值。
- [ ] 峰值流量、长上下文、失败重试、缓存、批处理和第三方工具均进入容量与成本模型。
- [ ] 团队监控每个成功任务的成本和质量，而不只看 token 单价。

可运行预算协议见[云 API 契约项目](projects/cloud-api-contracts.md)。

## 最后作出发布决定

发布会议结束时只保留一份结论，不要留下“大家应该都同意”的口头状态：

| 结论 | 何时使用 | 必须记录 |
|---|---|---|
| `go` | 所有适用门禁通过 | 版本、证据、负责人、放量与回滚条件 |
| `limited-go` | 风险可通过范围或流量隔离 | 租户/用户 allowlist、上限、观察窗口、自动停止条件 |
| `no-go` | 高风险证据缺失或门禁失败 | 阻断项、负责人、下一次复核所需证据 |

专项测试的精确命令、fixture 与 evidence boundary 不适合继续堆在发布清单里。需要复核某个实现结论时，
转到[项目控制台账](../evidence/project-controls.md)查对应 control；清单只负责确保风险没有被遗漏，
以及发布决定确实由证据支撑。
