# 作品集 claim 证据台账

本页保留仓库项目可以如何精确表述、必须同时披露什么，以及哪些说法会越过现有证据。第一次准备作品集请先读[简历项目与作品集](../career/resume-projects.md)；只有需要核对具体 control、版本和数字时再查本页。

## 项目不是技术名词清单

面试官需要确认你能定义问题、建立基线、做公平实验、处理故障并解释数字。作品集至少提交：

- 架构与信任边界；
- 数据说明和可运行配置；
- 原始评测结果与错误分类；
- 性能曲线和容量假设；
- 测试、CI 和复现命令；
- 已知限制与下一步。

## 每个数字都要有证据账本

把简历句子当作需要复核的实验结论。建议在仓库中保留 `evidence.md` 或机器可读 manifest：

| 字段 | 示例口径 |
|---|---|
| claim | “Recall@5 从 0.62 提升到 0.78” |
| task/data | 数据来源、许可、样本量、时间切分、标签定义 |
| baseline/candidate | model/revision、Prompt、索引、代码 commit、配置 fingerprint |
| metric | 公式、聚合单位、slice、置信区间或重复次数 |
| environment | CPU/GPU 型号、runtime、并发、输入/输出长度分布 |
| artifact | 原始 JSONL、报告、日志、复现命令和生成时间 |
| boundary | 离线/模拟/目标硬件/线上；已知未覆盖失败模式 |

数字至少满足“别人按固定 artifact 能重算”。配置 fingerprint 只证明显式序列化字段的 canonical bytes 身份；若漏掉模型 revision、数据、模板、索引或外部状态，仍不能重放，更不证明语义、质量或安全。

证据强度不要混写：

- 单元测试证明局部契约；
- 固定离线集证明该数据分布上的指标；
- CPU replay 不证明 GPU 吞吐；
- 合成/模拟故障不证明真实 provider 行为；
- shadow 观察不证明副作用链路；
- canary/线上实验必须给时间窗、样本量、流量占比与 guardrail。

如果当前只有离线 L2 证据，就写“在 N 条离线回放上”，不要写“生产可用”“高并发”“SLA 99.9%”或“线上提升”。

## 项目一：可诊断企业 RAG

**问题**：多租户文档问答，要求引用、更新和 ACL。

**工作**：结构切分、版本化 store/备份恢复、BM25+dense、RRF、rerank、引用验证、tenant/principal ACL、目标 tokenizer/chat-template context packing、packing→raw-output→answer trace binding、ASGI 服务、蓝绿索引和 tenant-aware cache。

**可写指标**：Recall@5、nDCG@10、引用 precision/recall、无答案准确、ACL 负例、p95、每成功答案成本。

**简历句式**：

> 面向 X 类文档和 Y 条评测查询，构建 BM25+dense+reranker 的多租户 RAG；相对 BM25 基线将 Recall@5 从 A 提升到 B，同时通过 N 条跨租户 canary 测试；在输入 P50/P95 为 M/N token 时将端到端 p95 控制在 T。

所有字母必须替换为可复现数字。不要只写“提升检索准确率”。

若项目使用 LangChain/LlamaIndex，不要只写“基于某框架搭建 RAG”。更有区分度的表述是：“保留框架无关的 canonical Document/SearchResult 与 authorization-first retriever，把同一次结果接入 LangChain/LlamaIndex Retriever API，并以 strict round trip 对账 ID、正文、保护 metadata、rank/score、Prompt bytes 与 answer artifact；框架路径不得改写 ACL 或评测口径。”本仓库当前可补充 exact evidence：`langchain-core==1.5.3`、`llama-index-core==0.14.23`，engineering/anonymous 分别返回 2/1 条授权证据，16 个测试覆盖保护键、rank/duplicate ID、NaN/±Inf/bool score、mutation、metadata exclusion 与 security context。若证据只来自这组四文档 authored fixture，还必须注明未执行 native vector index、learned embedding/reranker、LLM generation、目标数据库或性能负载；满分 Recall@4/nDCG@4 只能写成协议回归，不能写成框架质量榜。

**最低证据包**：版本化 corpus/query/qrels、BM25 baseline、候选检索结果、指标脚本、ACL 负例、引用 claim-source 对、长度分布和原始 latency 样本。若声称支持增量更新与恢复，还要展示 expected-version 冲突、显式 delete、事务中途失败回滚、一致 backup、strict manifest、物理/逻辑篡改拒绝和恢复后授权 query；单机 SQLite fixture 不能写成远端向量索引原子切换、完整灾备或已达到 RPO/RTO。先用 exact-span answer/abstain 基线贯通检索、packing 与离线 gate，保存 span offset/content hash 并证明 qrels 不进入生成；简历只能写“授权原文逐字支持”，不能写成语义正确率或 LLM 忠实度。Context packing 要绑定 tokenizer/revision、chat/system/user-template hash、完整 prompt token IDs、输出预留和每个 selected/dropped reason；generation trace 还要 exact join query/security context、逐 chunk version/content hash、canonical context、raw output 与 parsed answer fingerprint，并展示篡改失败 case。至少保留一个真实模型的失败样本：例如固定 Qwen control 的漏引与空 context 幻觉；简历可写“定位并建立门禁”，不能把 0/2 gate 改写成质量提升。若展示 publication-policy replay，可写“固定 attempt 上重建出 1 次 post-generation reject 与 1 次零调用 pre-generation abstain”，但必须标注它是 counterfactual replay，不得写成真实线上调用下降、忠实度提升或 guard 已与模型同跑。若展示后续的**真实 guarded runtime**，可写“固定 Qwen CPU control 中，有证据时 `GenerationMixin.generate` API 进入 1 次并在漏引后 reject；空授权证据时 callback/framework method 为 0 次并 pre-generation abstain”；不能省略“API method 计数不等于内部 forward/kernel/provider 请求或计费”，也不能把两个共享 authored corpus/checkpoint 的 query 写成质量提升。还要展示 audit/public allowlist projection 的泄露负例；仅写“拦截答案”却把含 raw output 的 decision JSON 返回前端，不算发布门禁。仅有 UTF-8 bytes 不能写成模型 token 窗口，自定义 fixture token IDs 也不能写成目标模型 token；unsigned hash 不能写成真实调用/生产 provenance，成功 tokenize 也不证明 tokenizer/权重匹配、生成忠实或 context 上限有效。若只在本机 CPU 跑过，延迟只注明该环境，不能外推生产 p95。

一般生成答案还应把 `claim → source ID` 收紧为可重放的 exact span artifact。可以写“以 strict `citation_evidence_span` 对账 supplied source、end-exclusive offset 与 quote，并让 unknown source/offset mismatch/duplicate JSON fail closed”；同时必须展示“无关 claim + exact span 仍通过”的正对照，明确该指标只建立 span identity，不建立 entailment、真实 ACL provenance、source truth 或答案完整性。

## 项目二：可恢复工具 Agent

**问题**：Agent 创建工单/发送消息，不能重复执行或越权。

**工作**：typed planner decision、独立 completion verifier、可信 subject/resource policy、cache 重新授权、proposal/execution 双 identity、审批 grant、幂等 ledger、step/token/cost/time 预算、循环/重复错误停止、严格结果快照、故障恢复、MCP adapter、注入测试和 trace。

**指标**：带 verifier 分母的任务成功率、平均步骤、参数合法率、policy-denied handler 数、未审批副作用 attempt 数、重复 applied effect 数、unresolved pending case 数、恢复成功率和 p95。安全 guardrail 不与任务成功取平均。

**面试证据**：演示 approval pause 后把 checkpoint 严格 JSON 化，新 runtime 恢复 handler counter，重新授权并执行原 pending decision，且 planner token/cost 不重复；再演示进程在工具调用后崩溃，恢复后查询 ledger 而不重复执行。展示身份漂移、过期 grant、未恢复 counter、撤权后 cache/resume 被拒绝，以及同 call_id 换参数/主体/资源/tool/policy version 被拒绝。

**最低证据包**：确定性 fake tool、verifier pass/fail、repeated action/短周期/repeated error、approval pause/resume、checkpoint tamper/identity/counter/revocation、执行前/执行后响应丢失、handler 非 JSON 结果、ledger 快照、同键异参冲突、审批过期和各预算耗尽 case；trace 记录 environment/policy/verifier version，并区分 proposal、handler attempt 与 verified effect。`0/N` 次重复副作用必须附分母、effect verifier、测试次数和场景覆盖，不能解释成所有生产故障下的绝对保证。若 planner、token/cost 和 verifier 都是本地 supplied fixture，应明确写“控制流回归”，不能包装成真实模型任务成功、账单或开放语义评测。无密钥 checkpoint hash、文件与 ledger 分开写入也不能写成“防篡改、原子持久化”。

若展示 LangChain/LlamaIndex 适配，可如实写：“以同一 strict Pydantic schema 构建 LangChain `StructuredTool` 和 LlamaIndex `FunctionTool` proposal adapters，把 framework call/tool id 映射到独立 canonical runtime；固定授权 case 两边各执行一次、同 id 重放 cache、跨 tenant 与未知 tool 在 handler 前拒绝，并以类型错误负例发现当前两个 direct tool API 的 schema enforcement 差异。”同一句必须披露：没有执行 LangGraph/LlamaIndex Agent loop、模型/provider、网络、remote tool、异步取消或真实副作用；adapter parity 不证明框架默认 ACL、幂等、生产安全或跨版本兼容。

若展示新增的 Agent-loop control，可另写：“在固定 `langchain==1.3.14`/`langgraph==1.2.10` 与 `llama-index-core==0.14.23` 上，真实执行 `create_agent()`/`FunctionAgent.run()` 的 model→tool→model，把 authorized/replay/cross-tenant/unknown-tool 四组调用接入同一 canonical runtime，并让独立 verifier 拒绝两组无可接受 receipt 的虚假成功文本；same-id replay 两框架 handler 均只执行一次。”必须紧邻披露 model 是 scripted in-process fixture；LangChain 使用 injected call ID，当前 LlamaIndex handler 未获 selection ID，而使用可信 fixture action hash，且 Workflow 每 case 捕获 73 次 Pydantic deprecation。不能写成真实模型任务成功、provider 接入、默认框架安全、remote side-effect exactly-once、durable resume、async/cancel、质量/性能或跨版本兼容。

若简历写“实现 MCP”，至少注明 protocol version、transport、initialize/capability 流程、实际 server/SDK、tools/resources/prompts 覆盖、timeout/cancel、认证和 conformance/smoke 环境。本仓库可准确拆成四条证据：一是“以官方 `mcp==1.29.0` memory client/server 执行 2025-11-25 初始化、发现和 schema/unknown-tool gate”；二是“以官方 stdio client/server 在独立 subprocess/OS pipe 上执行同一协议路径，并用最小 receipt 核对 handler 与 graceful EOF”；三是“以官方 Streamable HTTP client/session manager/low-level server 在独立 subprocess 与真实 loopback TCP/HTTP 上执行 stateful POST/GET SSE、DELETE 和 handler receipt”；四是“以自写固定子集分别覆盖 strict stdio framing 与 HTTP Origin/Bearer/session/cancel negative matrix”。后两种 official transport controls 仍没有借到 authored negative matrix 或 conformance；四条都没有完整 conformance、OAuth、TLS、远程、跨厂商 server 或业务授权。私有 readiness/shutdown token 也不能写成 MCP auth。不能缩写成“通用 MCP 平台”“生产认证”或“完成 Agent 互操作”。

若简历写“实现 A2A”，至少注明 spec/SDK version、binding、Agent Card discovery、实际 client/server、operation matrix、stream/cancel、schema/TCK、TLS/认证、签名与远程/跨语言环境。本仓库只能写“使用官方 a2a-sdk 1.1.2，在 IPv4 loopback JSON-RPC/HTTP 上执行 A2A 1.0 Agent Card、SendMessage/GetTask、版本/legacy 错误和冻结官方 Schema 校验，并以本地 verifier 复核 completed task”；不能写成“通过 A2A conformance”“跨厂商互通”或“安全 Agent 网络”。

### 可选项目：云 API 契约与 Responses typed events

如果岗位偏 API 平台、Agent infrastructure 或 LLMOps，可以把 Cloud API Contracts 独立展示：保留 OpenAI-compatible、Anthropic Messages 与 Gemini `generateContent` 的字段/terminal 差异，分开实现 strict JSON、SSE framing、retry/outcome policy、逐 attempt budget reconciliation，以及 Responses `response → output item → content part` typed lifecycle。

Anthropic 子集可如实写成：“为 Messages 构建 canonical→wire text adapter 与 text-block stream state machine，显式处理顶层 `system`、ordered content blocks、usage、stop reason 和 `message_stop`，并对 event/payload mismatch、inactive block、重复 terminal、截断 EOF 与非 text delta fail closed；另以 provider-neutral controls 验证 retry/outcome 三问和逐 attempt reserve/reconcile。”若展示固定预算数字，必须写明 80 uncertain + 66 settled = 146 micro-USD 来自 authored pricing/usage fixture、`httpx.MockTransport` 和本地 SQLite，不是 Anthropic 价格、usage、账单或真实请求。

该表述必须紧邻披露：仓库未执行 Anthropic SDK、账号、DNS/TLS、真实 HTTP/SSE、Claude model、tool/thinking/signature blocks、prompt caching、server cancellation 或 billing；text parser 会有损丢弃非 text blocks，offline state machine 也不证明 Claude 质量、安全、完整协议兼容或生产 SLO。只有另做固定 API/version 的受限 network smoke，才可增加“单请求 L4 协议证据”；仍不能把它写成生产接入。

Gemini 子集可如实写成：“为 legacy-but-supported `generateContent` 构建 text-only canonical→wire adapter 与单 candidate stream state machine，映射 `user/model`、`systemInstruction`、`usageMetadata` 和 `finishReason + EOF`，对 non-text part、多 candidate、非法 index/usage、重复或缺失 terminal fail closed；另按 2026-08-15 官方契约设计 Interactions resource/status/step/state/tool/retention 的迁移 gate。”“设计 Interactions gate”不能缩写成“实现 Interactions API”：仓库没有其 event fixture/parser，也没有执行 background、resume、function/thought steps。

同一句或紧邻位置必须披露：所有 `generateContent` 证据来自 authored fixture/offline state machine，未执行 Google GenAI SDK、账号、DNS/TLS、真实 HTTP/SSE、Gemini model、多模态、tool、thought/signature、file/cache 或 billing；text parser 只读 `candidates[0]` 的 text parts，会丢失 prompt feedback、安全、其他 candidates/parts、response id 与 usage 明细。80 uncertain + 66 settled = 146 micro-USD 是 provider-neutral MockTransport/SQLite fixture，不是 Google 定价、usage、发票或成本优化。不能写成“接入最新 Gemini”“支持全量多模态”“验证长上下文/缓存收益”或“生产级 Google Cloud 网关”。

本仓库当前可如实写成：

> 为 OpenAI Responses 实现 SDK-shaped typed-event 离线 replay，对固定 3,208-byte/15-event/2-item authored fixture 重建 text 与 function call，校验 response/item/content lifecycle、delta→done→terminal output 和 12+9=21 usage；以输入/event projection/receipt fingerprints 绑定工件，并用 16 个测试覆盖错序、refusal、incomplete/failed、未知字段、invalid UTF-8、duplicate/non-finite JSON 与截断。

同一句或紧邻位置必须披露：没有执行 OpenAI SDK、HTTP/SSE/WebSocket、真实 API、模型或 billing；authored `model`、response id 与 usage 不认证 provider；reviewed subset 不是完整 Responses API；连续 `sequence_number` 是本地 evidence 规则。不能把它改写成“接入最新 GPT”“验证线上 token 计费”“生产级 OpenAI 网关”或“保证工具调用安全”。

面试时应能解释三层边界：arbitrary byte chunk 不等于 SSE event，SSE event 不等于 typed Responses event，typed function call 也不等于已授权副作用。最低证据包包括原始 JSONL size/hash、event/type ledger、terminal receipt、parser revision、至少一个错序/截断失败 case，以及真实接入仍待验证的 auth、quota、rate limit、cancel、usage/billing reconciliation 清单。

## 项目三：单卡微调与推理

**问题**：在固定消费 GPU 上提高结构抽取或领域格式能力。

**工作**：Prompt/RAG 基线、严格 SFT/pairwise preference/raw judgment schema、group/exact 跨 split gate、A/B presentation/tie 与逐标注者判断保留、case/rater/order/agreement 门禁、train/combined artifact 绑定、held-out 审计与 train-only trainer 权限隔离、assistant-only SFT、scalar-head RM、LoRA/QLoRA/DPO、checkpoint、回归集、量化、vLLM 服务和压测。

**指标**：任务 F1/格式率、通用回归、峰值显存、训练时间、adapter 大小、success rate、client queue、dispatch/offered TTFT、TPOT、terminal latency、吞吐与每千成功任务成本；每项注明 all-attempt 或 success-conditional 分母。

**关键取舍**：解释为何选择微调而不是只加 Prompt，rank/target modules 如何消融，4-bit 对质量和性能有什么实际影响。

更完整的项目讲法是：“将 combined 数据审计与 train-only trainer 分权，用 readiness 绑定有序训练集、split/group/lexical/governance 决策；在 backward 前核对目标 tokenizer 的 assistant mask 与 configured collator 最终 labels；发布 adapter 时绑定 base revision、tokenizer/template 并做新基座重载；最后用同一 held-out artifact 比较 base、Prompt/RAG 与 adapter。”只有在目标 GPU 实测后，才补充 QLoRA 峰值、吞吐和 OOM 降级曲线；公式估算、CPU/Gloo 或固定 Qwen recorded control 不能代替 CUDA 结果。

SFT 标签管线可如实写成：“在固定 revision、加载前重哈希的 Qwen2.5-0.5B-Instruct 上，发现原生模板对多轮、并行 tool calls、tool preamble 三条 authored fixture 返回全零 assistant mask；审核 generation-aware 模板并在 Arrow 前预分词后，以真实 TRL 0.29.1 collator 核对 `[3,301]`、90 个监督 labels/813 个 `-100`，并执行 CPU FP32 no-grad forward。”同一句必须披露模板证据只覆盖固定 Qwen schema/fixture，forward loss `1.251716` 不是训练结果；不能写成“完成 SFT”“loss 已收敛”“数据合规”或“支持任意 provider 工具格式”。

仓库当前可以如实写成：“在固定 revision、加载前逐文件重哈希的 Qwen2.5-0.5B-Instruct 上，以 CPU FP32 执行 41-token prompt/3-token assistant-only supervision、24 层 `q_proj/v_proj` LoRA backward 与一次 AdamW step；验证 494,032,768 frozen-base 参数 fingerprint 不变、48 个 B tensors 非零，并将 1.09 MB adapter 重载到新基座得到 bit-exact logits。”必须同时披露这是单样本单步，loss 从约 0.003864 升到 0.584557；不能改写成“loss 下降”“质量提升”“完成 QLoRA”或“单卡 GPU 训练优化”。若简历要写后四项，需再补 held-out 曲线、目标 CUDA/bitsandbytes、峰值显存、吞吐和统一质量回归。

DPO 可另写成严格的机制证据：“在同一固定 checkpoint 上绑定两条 authored binary pair/readiness，以 TRL 0.29.1 + PEFT 执行一次 CPU FP32 DPO step；核对 `[4,28]` token/mask、96 个 finite gradients、冻结 parameter/state/config 指纹，loss `0.693147→0.333352`。”同一句必须披露它是同 batch 单步、reference replay drift=`0.547077` 且无 bitwise determinism，不得改成“基于人类反馈完成对齐”“提升模型偏好/安全”或“完成 QLoRA”。真正写质量提升需要独立 held-out 人类/任务评测、置信区间、长度/position/风格控制、通用与安全回归。

若岗位偏模型原理、推理框架或训练系统，可把 Transformers Basics 作为独立机制项目：“从 byte-BPE、dense/online/cache attention 和真实 `GenerationMixin.generate()` 停止控制出发，建立 immutable release-evidence→固定 Qwen CPU FP32 prefill/cache/generate→selected-weight INT4→activation-patching 正负对照的分层证据链；再以两进程 Gloo fixtures 分开验证 global capacity、token-to-owner all-to-all、reverse-split backward 和一步 SGD。”这句话必须同时说明 target Qwen 只覆盖一个 31-token prompt、一个矩阵与一个 France/Germany pair，MoE 是 authored tiny fixtures；CPU/Gloo、逻辑 payload bytes和单步 loss 不能写成完整 low-bit/MoE checkpoint、CUDA/NCCL、FlashAttention、质量或性能复现。

如果把多条 Qwen control 组织成一个作品集，建议展示“共享 immutable checkpoint、分离 execution scope”的证据地图，而不是合成一句全栈成功：config/GQA/KV 属于静态推导；7-file hash 与 CPU FP32 cache/generate 属于权重运行；selected `o_proj` INT4 只属于局部 artifact/dequantized-forward；activation patching 是单协议干预；RAG 原始 0/2 与 guarded 1-call reject/0-call abstain 属于失败捕获和发布策略；SFT final-label、LoRA、DPO 又是三条不同训练路径。只有每个节点都链接自己的 manifest/report/scope，才可以写成“建立分层证据链”；不能写成“完成 Qwen INT4/GPU 微调、RAG 质量提升、增量流式 vLLM 部署”。仓库没有目标 GPU、代表性 held-out 或线上 workload，Qwen 页定义的 L5 仍未取得。

Llama release evidence 应与上面的 Qwen runtime evidence 分开写。当前仓库可以如实表述：“固定 Meta Llama 3.2 text-only model-card commit `0e0b8c…f301`，核对 25,416-byte/SHA-256、六段 exact source fragments 与 strict manifest/projection fingerprints，并把 1B/3B、128k、GQA、shared embeddings、9T 与 cutoff 投影为 vendor-reported claims。”同一句或紧邻位置必须披露：默认离线 receipt 的 `upstream_verified=false`，没有加载 Llama config/tokenizer/weights，没有执行 forward/generate/128k/GPU，也没有证明参数量、有效上下文、许可适用、质量、性能或生产安全。不能把固定 model card、authored GQA fixture 和另一个 Qwen checkpoint 的 runtime control 拼成“已部署/微调/评测 Llama”。

DeepSeek 当前只能写成 L2 config/fail-closed 项目：“固定 DeepSeek-V3 revision `e815299…c9eb` 的 1,660-byte raw config/SHA-256 与 semantic snapshot，识别 MLA/MoE/FP8/YaRN/MTP markers；即使存在 128 query/KV heads，也拒绝标准 MHA/GQA cache 公式并输出 `estimate_refused=true`。”必须披露没有下载/执行 DeepSeek weights、tokenizer、`auto_map` remote code、MLA cache、MoE routing、FP8/MTP、R1 或 API。通用 NumPy/PyTorch/Gloo MoE controls 只能另写为 authored mechanism fixtures，不能合并成“复现 DeepSeekMoE”；条件式 44,040,192-parameter expert 公式也不是 state-dict inventory、激活参数或 GPU 性能证据。

面试时应能现场解释四种身份：model-card/config artifact 被固定不等于权重已执行，权重逐文件重哈希不等于发布者签名，单次 forward 不等于总体质量，recorded verifier 通过也不等于本轮重新运行约 1 GB 模型。作品集至少提交 tokenizer merge/round-trip、attention tolerance、generation token trace、selected-file manifest、cache/full error、patching 正负对照、MoE split/collective ledger 和一个故意失败 case；只给截图或最终 JSON 数字不够。

若岗位偏训练框架或 JAX，可把 MiniGPT 作为独立机制项目：“用 core JAX/Optax 实现纯函数 decoder、PyTree state、JIT update 与显式 PRNG；通过同解析权重对账 PyTorch/JAX 的 20 个 unique parameter gradients 和 plain-SGD step，再以 shared-mask 三步 AdamW 对账 clipping/moments/schedule，并由两个 spawn processes 验证 params/Optax/typed PRNG/permutation/cursor 的 bit-exact resume。”证据数字可写 632 参数 60-step overfit、gradient max diff `2.384185791015625e-07`、13,476-byte strict artifact 和 wrong-PRNG/wrong-cursor 两个约 `0.037` 参数漂移，但必须链接对应 report/test。必须注明这是 CPU tiny fixture；shared mask 不证明 native RNG equivalence，`ALLMJAX1` 不等于 Orbax/TensorStore 或分片 checkpoint，不能写成“完成 TPU/GPU 多设备训练”或“JAX 性能优于 PyTorch”。

服务部分可如实写成：“把固定 revision、加载前重哈希的 Qwen2.5-0.5B-Instruct 放入 Transformers CPU FP32 reference subprocess，经真实 IPv4 loopback HTTP/Bearer 完成 models、拒绝负例，以及 non-stream/SSE 各一次；对账 prompt 31 tokens、completion IDs `[17,151645]`、usage/stop、后端 fingerprint 和两次 `GenerationMixin.generate()` audit，并提供 closed-schema recorded verifier。”不能缩写成“部署 vLLM”“完成高并发 OpenAI API”“实现流式取消”或“达到生产 SLO”：该 SSE 在 generation 完成后才发块，没有 CUDA/TLS/IAM/多 worker/远程/性能/质量证据。

可另写一条严格分开的控制流证据：“以 authored async pseudo-token backend 和真实 Uvicorn loopback subprocess 验证 content-before-completion；client 收到首 delta 后断开，ASGI task/backend iterator 均观察 `CancelledError`，且后续 scripted token 未产生。”必须同时写明它没有运行 Qwen/Transformers generation thread、vLLM/CUDA、KV/GPU release 或 provider billing；不能把两条 control 拼成“Qwen 已支持增量解码与断连释放显存”。

若加入 tiny Transformers control，可准确写：“在随机 1,272 参数 GPT-2 CPU 上真实执行一次 forward 与 thread 内 `GenerationMixin.generate()`；以 streamer pause 固定首 token 竞争窗口，断连后由 event/`StoppingCriteria` 令 continuation 保持 `[7]` 并 join thread。”同一句必须披露 cooperative hook 和暂停均为自写控制机制，没有 tokenizer/公开或目标 checkpoint、正常 logits、未修改调用、CUDA/KV-release/性能证据。不能把它和 Qwen post-completion service 合并成“目标 Qwen 已实现可取消流式推理”。

量化证据要拆开写：CPU reference 完成 2–8 bit offset-binary dense packing、padding/非法码拒绝、strict little-endian 单矩阵 artifact、no-overwrite 和 exact reload，只能证明该 tensor 序列化契约与 raw/file bytes；没有多 tensor manifest、未量化参数、config/tokenizer 和签名就不能写成完整模型文件，没有目标 GPU fused kernel 和 workload 就不能写成显存节省或加速。简历中的“INT4”必须注明 code range、group/axis/scale、packing layout、未量化层、artifact/resident bytes 和目标硬件实测边界；unkeyed SHA-256 不能写成来源认证。

当前可再写一条目标权重但严格局部的证据：“在固定 revision、加载前重哈希的 Qwen2.5-0.5B-Instruct 上，捕获真实 31-token activation，将第一层 `[896,896]` `o_proj.weight` 的 802,816 个 FP32 参数按 row-group 128/码域 `[-7,7]` packed INT4；strict artifact 为 427,328 bytes，相对该矩阵 FP32 为 7.514752×，重载后 selected output exact，并执行仅替换该矩阵的完整模型 forward。”同一句必须披露它只覆盖全模型 0.1625% 参数、计算使用 dequantized FP32；last logits relative-L2/max-abs 为 0.085138/1.625518，argmax 17→17 只是单提示观察。不能缩写成“量化 Qwen”“显存下降 7.5×”“无损 INT4”或“推理加速”。

KV INT8 也要单列：per-token/per-KV-head CPU oracle 能写“物化 K/V code+scale、验证 causal GQA 与 incremental prefix、报告 K/V/probability/output error 和 payload bytes”，不能缩写成“KV cache 4×、注意力无损或推理加速”。至少注明 scale granularity、`Hkv`、head dim、block/alignment/workspace 是否计入，以及是否真的执行目标 runtime fused kernel 和长上下文质量集。

Continuous batching 作品也要区分 state-machine 与性能证据。CPU 离散 oracle 可写“实现 FCFS admission、sequence/token cap、chunked prefill、decode-first 和 `P+O-1` forward-work ledger，并验证 queue/first-token/completion trace”；不能写成“复现 vLLM 调度、提高吞吐或降低 TTFT”。简历若报告优化百分比，必须在固定 model/runtime/hardware/workload/arrival process 下保存真实 server trace、质量和失败分母；离散 step 与 token-slot utilization 不能冒充秒或 GPU utilization。

MoE 机制项目可写“从零实现 padding-aware top-k、per-expert capacity、deterministic overflow、post-drop combine，并对 exact count/weight 做回归”；如果只执行线性 expert CPU fixture，就不能写“复现 DeepSeek/Qwen MoE”“完成 expert parallel”或“降低推理 FLOPs”。最低证据还应列清 routing group、capacity 公式、tie/drop/reroute、assignment 与 token drop 双分母、aux/z-loss 版本，以及目标硬件的 all-to-all/grouped-GEMM trace。

若简历写 PPO/在线 RL，最低限度还要保存 rollout policy/revision、old log-prob、reward/value、terminated/truncated/EOS/padding 语义、GAE/return、advantage normalization、ratio/clip fraction、sampled KL 口径、value/entropy/KL loss 与 optimizer/checkpoint 状态。只有 authored NumPy GAE/clipping toy 时，应写“解析验证 objective 与 mask 契约”；两状态 PyTorch control 可写“贯通 categorical rollout、GAE 与 minibatch optimizer，并用可枚举期望回报做控制”，但仍不能写“完成语言模型 RLHF”“KL 受控”或“对齐效果提升”。

随机 tiny Transformer 的 integer-token PPO 可进一步写“贯通 causal backbone token rollout、冻结 reference、sampled log-ratio reward 与 snapshot-bound PPO，并穷举短 horizon 精确 task reward”；若没有 tokenizer、自然语言、learned RM、目标 checkpoint、真实偏好 held-out 与 GPU 记录，仍只能算机制控制，不能包装成目标模型 RLHF 项目。

本地 text PPO control 还可如实写“贯通 WordLevel/chat template、EOS/length/padding 变长 rollout、分离 actor/critic、冻结 reference 与精确两 token oracle；从 \(25/169\) 提升到 1.9 以上，目标 `good, EOS` 概率从 \(1/169\) 提升到 0.95 以上”。必须同时注明 reward 是作者构造、模型是随机 tiny GPT-2、有限时域 objective 在 generation cap 停止且默认不 bootstrap；这仍不是 learned-RM RLHF、目标 checkpoint 训练、自然语言质量、GPU 或生产稳定性证据。

若加入当前 learned-RM PPO 对照，可以写“训练并冻结 tiny Transformer scalar RM，在 generation allowlist 的 57 条完整 response support 上发现训练 chosen 仅排第 38；PPO 精确 proxy 从 2.739 升至 4.652，但 strict target success 从 \(1/64\) 降至 \(4.99\times10^{-4}\)，以自动化不变量复现受控 proxy exploitation”。还应披露 dense partial credit 从 \(15/64\) 升至 0.566，避免声称所有外部指标恶化。不要简写成“发现真实 reward hacking”：数据只有一个 authored pair，模型与 support 都是 tiny fixture，没有人类 held-out、目标 checkpoint、CUDA 或线上影响证据。

**最低证据包**：可复算的数据 audit manifest、许可审查责任人与结论、group-level 切分，以及写明 normalization/n/阈值/比较分母/人工复核的 near-duplicate 报告；再附 purpose/expiry/evidence 明确的 source registry、敏感候选 detector 的类别/误差/例外账本、不含 held-out 原文的 readiness、trainer 对篡改/陈旧凭证的拒绝测试、chat template 与 assistant-only mask、Prompt baseline、rank/target-module 消融、训练曲线与 checkpoint、held-out 回归、峰值显存和固定 workload。服务压测需固定 arrival process/长度分布/并发，保存 scheduled-offered/dispatch/first-token/terminal 与全部 outcome；若只在 semaphore 后计时，不能声称测得用户排队或 offered-load SLO。scheduled `offered_at` 也不证明负载机准时执行，需报告 generator lag/CPU；有限 seeded schedule 不等于生产流量分布。若写 RM，至少附 pair 与 split binding、完整 prompt+response tokenization/truncation、scalar-head pooling 位置、train/held-out loss 和 strict pair accuracy、zero-margin tie、margin/score 分布、head/backbone 冻结策略，以及 length/style/position 等 counterfactual slice；还应证明 trainer 只拿 train+held-out-free readiness，并在模型加载前拒绝缺失、篡改或陈旧绑定。作者构造的数值或 tiny Transformer overfit 不能写成人类 preference、目标 RM、广泛 OOD 鲁棒性或 reward-hacking 证据，无密钥 readiness 也不能写成签名认证。若写 DPO，再附原始 presentation/tie/disagreement、有序 binary train/combined binding、prompt 与跨 A/B candidate 的 lexical 比较分母、prompt/两侧 candidate governance、目标 tokenizer 的 prompt-prefix/两侧 token/截断报告、prompt/completion mask、policy/reference revision、beta、chosen/rejected log-prob 与人类 held-out preference；偏好评测还应附逐标注者 raw judgment、case/rater/order 完整性 gate、raw agreement/Fleiss’ κ 的 numerator/denominator、逐 case position estimand 与 case-cluster interval。顺序覆盖或 blind flag 只是协议声明，不能包装成随机化、真实盲化或因果位置偏差证据；tiny authored-pair overfit 和 authored judgment fixture 只能写控制流/统计口径回归。即使 preference readiness 执行了 source/license/sensitive/near-duplicate gate，registry allow 也不能写成“法律许可”，扫描未命中不能写成“无 PII/secret”，exact/lexical gate 通过不能写成“无泄漏”，无密钥 hash 不能写成“签名/防篡改”；QLoRA 不能简写成“全流程 4-bit”，因为通常只是冻结底座低位存储，而计算、adapter、梯度、optimizer 和激活仍有更高精度状态。

## 项目四：可审计合成数据供应链

**问题**：生成候选进入训练前，来源图、verifier gate、重复分母和目标采样暴露不可追溯，报告本身也可能与输入或 policy 漂移。

**工作**：strict JSONL contract、external/internal parent graph、cycle/round diagnostics、required verifier missing/fail、generator/verifier revision overlap、byte-exact/NFC-whitespace identity、candidate/eligible/eligible-unique 三分母、target mixture expectation、input/policy-bound artifact 与 full local recomputation。

**可写指标**：candidate/eligible/eligible-unique count、missing/failed/unresolved/cycle/nonmonotonic count、各 round acceptance、human-review authored flag count、synthetic target fraction、expected repetition，以及 artifact/input byte identity。若还没有真实训练 observed ledger，不要报告“实际 synthetic 暴露”。

本仓库当前可以如实写成：

> 实现 CPU/offline 合成数据审计 reference control，以 strict loader 拒绝 duplicate key、non-finite、unknown field 与 invalid UTF-8；对 external/internal lineage、cycle、generation-round monotonicity、required verifier 和 exact duplicate 分账；将 1,457-byte records、341-byte mixture 与 caller policy 绑定到 `about-llm.synthetic-data-audit.v2`，从可信输入完整复算固定 4 candidates / 2 eligible / 1 eligible unique 与 25% target synthetic / 5.0 expected repetition，并以 40 个测试覆盖 input drift、cooperative rehash 和 no-overwrite。

同一句或紧邻位置必须披露：`eligible` 只表示声明的 required verifier 存在且 pass；revision overlap 只是字符串 finding；target repetition 不是 observed training exposure；无密钥 SHA-256 不认证发布者；file `fsync` 不证明 directory durability。这个 control 没有调用 teacher/student/verifier model，没有训练、人工 calibration、语义 near-duplicate、许可/consent/PII/secret、无泄漏、collapse 或 downstream benefit 证据。

**最低证据包**：

- 固定 records/mixture 原始 bytes、size/SHA-256 与 strict schema；
- caller-supplied required verifier、known parents 与 fingerprint profile；
- candidate / eligible / eligible unique 三种分母和逐 round 账本；
- missing、failed、unresolved、nonmonotonic、cycle、revision-overlap 的独立 findings；
- audit→verify 命令、canonical report 和可信 policy head 的保存位置；
- duplicate key、input byte drift、policy drift、cooperative rehash、output collision 五类反例；
- 真实流水线的 target/observed token ledger 设计与 publication policy；
- 尚未执行的 teacher/verifier、法律隐私、语义质量与训练收益清单。

面试时要能解释为什么“同时改结果并重算 self-hash”仍会相对原输入失败，以及为什么攻击者若连 caller 的 inputs/policy 一起替换，无密钥 full recomputation 仍不能提供 provenance authentication。也要能区分 file flush、file `fsync`、directory `fsync`、atomic rename 和 immutable object publication。

## 项目五：模型评测平台

**问题**：Prompt、模型、索引与工具升级缺少统一门禁。

**工作**：严格 JSONL schema、ordered case/result/answer/metric/system run manifest、绑定全部 bootstrap/gate 配置与 run identity 的 comparison artifact、case/slice、raw judgment、case/rater/order 完整性门禁、agreement/κ、逐 case position diagnostic、runner、配对 bootstrap、judge 校准、报告和 CI gate。

**指标**：人工一致性、judge precision、回归检出率、评测耗时/成本、发布阻断与线上事故关联。

**最低证据包**：不可变 case id 之外，还要用 manifest 绑定每个 case 的 input/gold/slice/metadata、ordered baseline/candidate result、recorded output identity、metric implementation revision、scorer 与 system revision；comparison artifact 再绑定 bootstrap seed/sample/confidence、质量/安全/延迟/slice 阈值、统计结果和全部失败原因。展示 duplicate key、同 ID 换 gold、结果/manifest/threshold 篡改和 metric revision mismatch 被拒绝，并区分只检查工件内部自洽的 `verify-comparison` 与重开 answers/results/manifests、重新评分和重建 comparison 的 `verify-evidence`。若展示 HMAC release ledger，还要把 chain authentication、引用 artifact byte rehash 和 ledger 外 trusted-head 截断检测分开报告；公开 fixture key、caller timestamp 或无外部 head 的合法链不能写成生产 key custody、可信时间或不可回滚历史。再附逐标注者原始判断与 assignment/presentation metadata、专家校准子集、agreement 的明确分母、paired difference 与 case-level 置信区间。unsigned fingerprint 和自报 system id 不能写成真实模型来源认证或不可篡改历史；若没有真实人类实验或发布历史，就只能展示 authored fixture/离线 gate 行为，不能声称标注一致性、因果位置偏差或线上事故下降。

本仓库可如实写成：“固定 Qwen2.5-0.5B-Instruct immutable revision，在 CPU FP32 greedy 下真实生成 7 条 authored cases；保存 raw/token/terminal identity，并以 reviewed suite/report 重算 literal exact 4/7、NFKC+casefold normalized exact 5/7、token F1 6/7，定位英文算术、大小写复制和 JSON metric mismatch。”必须同句或紧邻披露：suite 未外部预注册、未独立抽样/留出、非代表性、无统计功效；没有 latency、系统对照、judge、人评或发布 gate。禁止把 token F1 的 `6/7=85.7%` 改写成“Qwen 准确率 85.7%”或“模型质量达到 85.7%”。

结构化评测还可如实写：“实现 strict JSON Schema v2 与 canonical JSON value v1 两个独立 scorer；duplicate object key、`NaN/Infinity`、`$id`/external ref fail closed，五条 fixture 精确展示 schema-valid wrong value 与 F1=1 的 reversed array 仍被 value gate 拒绝，并把 metric revision 写入 run manifest。”同时披露：`format` 仍是 annotation，value policy 区分 integer/float、保留 array order；authored fixture 的 `latency_seconds=0.0` 不是性能测量，两种 scorer 都不证明业务语义、权限、真实模型质量或生产安全。

## 推荐的证据组织：从结果追到原始 artifact

作品集 README 给结论，报告给分层指标，原始 artifact 给逐 case 记录，代码和 lockfile 给重算路径。面试演示时随机挑一个失败 case，能够从最终分数一路追到输入、检索证据、Prompt、模型输出和判分理由，比只展示漂亮 dashboard 更可信。

个人或敏感数据不要为了“可复现”直接提交；发布脱敏/合成样例、schema、生成脚本与访问说明，并明确它与内部数据的差异。

## 常见扣分项

- 声称使用“大模型”却说不清 model id、revision 和 Prompt。
- 只报最好结果，不报基线、样本量、方差和失败案例。
- 把框架名当核心贡献。
- 在测试集上反复调参。
- 把本地 demo 描述成高并发生产系统。
- 写 99% 准确率却不能定义标签和类别分布。
- 没有处理密钥、权限、日志、重试和删除。
- 把 pass@k 写成一次生成成功率，或不说明 \(k\)、采样数与测试覆盖。
- 只写模型名称，不固定 revision、tokenizer/chat template、Prompt 和 generation config。
- 用 fingerprint、trace id 或“可观测”宣称请求一定可重放。
- 把 schema-valid 当业务正确，把摘要当真实状态，把 delimiter 当安全边界。

## 面试前自查

对简历每个数字都能回答：数据从哪来？样本量？基线？公式？聚合单位？硬件？模型版本？配置与 artifact 在哪？置信区间？失败分布？证据属于离线、目标硬件还是线上？若重做会改什么？若不能，删掉或补证据。
