# 简历项目与作品集

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

**工作**：结构切分、BM25+dense、RRF、rerank、引用验证、tenant/principal ACL、目标 tokenizer/chat-template context packing、packing→raw-output→answer trace binding、蓝绿索引和 tenant-aware cache。

**可写指标**：Recall@5、nDCG@10、引用 precision/recall、无答案准确、ACL 负例、p95、每成功答案成本。

**简历句式**：

> 面向 X 类文档和 Y 条评测查询，构建 BM25+dense+reranker 的多租户 RAG；相对 BM25 基线将 Recall@5 从 A 提升到 B，同时通过 N 条跨租户 canary 测试；在输入 P50/P95 为 M/N token 时将端到端 p95 控制在 T。

所有字母必须替换为可复现数字。不要只写“提升检索准确率”。

**最低证据包**：版本化 corpus/query/qrels、BM25 baseline、候选检索结果、指标脚本、ACL 负例、引用 claim-source 对、长度分布和原始 latency 样本。先用 exact-span answer/abstain 基线贯通检索、packing 与离线 gate，保存 span offset/content hash 并证明 qrels 不进入生成；简历只能写“授权原文逐字支持”，不能写成语义正确率或 LLM 忠实度。Context packing 要绑定 tokenizer/revision、chat/system/user-template hash、完整 prompt token IDs、输出预留和每个 selected/dropped reason；generation trace 还要 exact join query/security context、逐 chunk version/content hash、canonical context、raw output 与 parsed answer fingerprint，并展示篡改失败 case。仅有 UTF-8 bytes 不能写成模型 token 窗口，自定义 fixture token IDs 也不能写成目标模型 token；unsigned hash 不能写成真实调用/生产 provenance，成功 tokenize 也不证明 tokenizer/权重匹配、生成忠实或 context 上限有效。若只在本机 CPU 跑过，延迟只注明该环境，不能外推生产 p95。

## 项目二：可恢复工具 Agent

**问题**：Agent 创建工单/发送消息，不能重复执行或越权。

**工作**：typed planner decision、独立 completion verifier、可信 subject/resource policy、cache 重新授权、proposal/execution 双 identity、审批 grant、幂等 ledger、step/token/cost/time 预算、循环/重复错误停止、严格结果快照、故障恢复、注入测试和 trace。

**指标**：带 verifier 分母的任务成功率、平均步骤、参数合法率、policy-denied handler 数、未审批副作用 attempt 数、重复 applied effect 数、unresolved pending case 数、恢复成功率和 p95。安全 guardrail 不与任务成功取平均。

**面试证据**：演示 approval pause 后把 checkpoint 严格 JSON 化，新 runtime 恢复 handler counter，重新授权并执行原 pending decision，且 planner token/cost 不重复；再演示进程在工具调用后崩溃，恢复后查询 ledger 而不重复执行。展示身份漂移、过期 grant、未恢复 counter、撤权后 cache/resume 被拒绝，以及同 call_id 换参数/主体/资源/tool/policy version 被拒绝。

**最低证据包**：确定性 fake tool、verifier pass/fail、repeated action/短周期/repeated error、approval pause/resume、checkpoint tamper/identity/counter/revocation、执行前/执行后响应丢失、handler 非 JSON 结果、ledger 快照、同键异参冲突、审批过期和各预算耗尽 case；trace 记录 environment/policy/verifier version，并区分 proposal、handler attempt 与 verified effect。`0/N` 次重复副作用必须附分母、effect verifier、测试次数和场景覆盖，不能解释成所有生产故障下的绝对保证。若 planner、token/cost 和 verifier 都是本地 supplied fixture，应明确写“控制流回归”，不能包装成真实模型任务成功、账单或开放语义评测。无密钥 checkpoint hash、文件与 ledger 分开写入也不能写成“防篡改、原子持久化”。

## 项目三：单卡微调与推理

**问题**：在固定消费 GPU 上提高结构抽取或领域格式能力。

**工作**：Prompt/RAG 基线、严格 SFT/pairwise preference/raw judgment schema、group/exact 跨 split gate、A/B presentation/tie 与逐标注者判断保留、case/rater/order/agreement 门禁、train/combined artifact 绑定、held-out 审计与 train-only trainer 权限隔离、assistant-only SFT、scalar-head RM、LoRA/QLoRA/DPO、checkpoint、回归集、量化、vLLM 服务和压测。

**指标**：任务 F1/格式率、通用回归、峰值显存、训练时间、adapter 大小、success rate、client queue、dispatch/offered TTFT、TPOT、terminal latency、吞吐与每千成功任务成本；每项注明 all-attempt 或 success-conditional 分母。

**关键取舍**：解释为何选择微调而不是只加 Prompt，rank/target modules 如何消融，4-bit 对质量和性能有什么实际影响。

量化证据要拆开写：CPU reference 完成 2–8 bit offset-binary dense packing、padding/非法码拒绝、strict little-endian 单矩阵 artifact、no-overwrite 和 exact reload，只能证明该 tensor 序列化契约与 raw/file bytes；没有多 tensor manifest、未量化参数、config/tokenizer 和签名就不能写成完整模型文件，没有目标 GPU fused kernel 和 workload 就不能写成显存节省或加速。简历中的“INT4”必须注明 code range、group/axis/scale、packing layout、未量化层、artifact/resident bytes 和目标硬件实测边界；unkeyed SHA-256 不能写成来源认证。

KV INT8 也要单列：per-token/per-KV-head CPU oracle 能写“物化 K/V code+scale、验证 causal GQA 与 incremental prefix、报告 K/V/probability/output error 和 payload bytes”，不能缩写成“KV cache 4×、注意力无损或推理加速”。至少注明 scale granularity、`Hkv`、head dim、block/alignment/workspace 是否计入，以及是否真的执行目标 runtime fused kernel 和长上下文质量集。

Continuous batching 作品也要区分 state-machine 与性能证据。CPU 离散 oracle 可写“实现 FCFS admission、sequence/token cap、chunked prefill、decode-first 和 `P+O-1` forward-work ledger，并验证 queue/first-token/completion trace”；不能写成“复现 vLLM 调度、提高吞吐或降低 TTFT”。简历若报告优化百分比，必须在固定 model/runtime/hardware/workload/arrival process 下保存真实 server trace、质量和失败分母；离散 step 与 token-slot utilization 不能冒充秒或 GPU utilization。

MoE 机制项目可写“从零实现 padding-aware top-k、per-expert capacity、deterministic overflow、post-drop combine，并对 exact count/weight 做回归”；如果只执行线性 expert CPU fixture，就不能写“复现 DeepSeek/Qwen MoE”“完成 expert parallel”或“降低推理 FLOPs”。最低证据还应列清 routing group、capacity 公式、tie/drop/reroute、assignment 与 token drop 双分母、aux/z-loss 版本，以及目标硬件的 all-to-all/grouped-GEMM trace。

若简历写 PPO/在线 RL，最低限度还要保存 rollout policy/revision、old log-prob、reward/value、terminated/truncated/EOS/padding 语义、GAE/return、advantage normalization、ratio/clip fraction、sampled KL 口径、value/entropy/KL loss 与 optimizer/checkpoint 状态。只有 authored NumPy GAE/clipping toy 时，应写“解析验证 objective 与 mask 契约”；两状态 PyTorch control 可写“贯通 categorical rollout、GAE 与 minibatch optimizer，并用可枚举期望回报做控制”，但仍不能写“完成语言模型 RLHF”“KL 受控”或“对齐效果提升”。

随机 tiny Transformer 的 integer-token PPO 可进一步写“贯通 causal backbone token rollout、冻结 reference、sampled log-ratio reward 与 snapshot-bound PPO，并穷举短 horizon 精确 task reward”；若没有 tokenizer、自然语言、learned RM、目标 checkpoint、真实偏好 held-out 与 GPU 记录，仍只能算机制控制，不能包装成目标模型 RLHF 项目。

本地 text PPO control 还可如实写“贯通 WordLevel/chat template、EOS/length/padding 变长 rollout、分离 actor/critic、冻结 reference 与精确两 token oracle；从 \(25/169\) 提升到 1.9 以上，目标 `good, EOS` 概率从 \(1/169\) 提升到 0.95 以上”。必须同时注明 reward 是作者构造、模型是随机 tiny GPT-2、有限时域 objective 在 generation cap 停止且默认不 bootstrap；这仍不是 learned-RM RLHF、目标 checkpoint 训练、自然语言质量、GPU 或生产稳定性证据。

若加入当前 learned-RM PPO 对照，可以写“训练并冻结 tiny Transformer scalar RM，在 generation allowlist 的 57 条完整 response support 上发现训练 chosen 仅排第 38；PPO 精确 proxy 从 2.739 升至 4.652，但 strict target success 从 \(1/64\) 降至 \(4.99\times10^{-4}\)，以自动化不变量复现受控 proxy exploitation”。还应披露 dense partial credit 从 \(15/64\) 升至 0.566，避免声称所有外部指标恶化。不要简写成“发现真实 reward hacking”：数据只有一个 authored pair，模型与 support 都是 tiny fixture，没有人类 held-out、目标 checkpoint、CUDA 或线上影响证据。

**最低证据包**：可复算的数据 audit manifest、许可审查责任人与结论、group-level 切分，以及写明 normalization/n/阈值/比较分母/人工复核的 near-duplicate 报告；再附 purpose/expiry/evidence 明确的 source registry、敏感候选 detector 的类别/误差/例外账本、不含 held-out 原文的 readiness、trainer 对篡改/陈旧凭证的拒绝测试、chat template 与 assistant-only mask、Prompt baseline、rank/target-module 消融、训练曲线与 checkpoint、held-out 回归、峰值显存和固定 workload。服务压测需固定 arrival process/长度分布/并发，保存 scheduled-offered/dispatch/first-token/terminal 与全部 outcome；若只在 semaphore 后计时，不能声称测得用户排队或 offered-load SLO。scheduled `offered_at` 也不证明负载机准时执行，需报告 generator lag/CPU；有限 seeded schedule 不等于生产流量分布。若写 RM，至少附 pair 与 split binding、完整 prompt+response tokenization/truncation、scalar-head pooling 位置、train/held-out loss 和 strict pair accuracy、zero-margin tie、margin/score 分布、head/backbone 冻结策略，以及 length/style/position 等 counterfactual slice；还应证明 trainer 只拿 train+held-out-free readiness，并在模型加载前拒绝缺失、篡改或陈旧绑定。作者构造的数值或 tiny Transformer overfit 不能写成人类 preference、目标 RM、广泛 OOD 鲁棒性或 reward-hacking 证据，无密钥 readiness 也不能写成签名认证。若写 DPO，再附原始 presentation/tie/disagreement、有序 binary train/combined binding、prompt 与跨 A/B candidate 的 lexical 比较分母、prompt/两侧 candidate governance、目标 tokenizer 的 prompt-prefix/两侧 token/截断报告、prompt/completion mask、policy/reference revision、beta、chosen/rejected log-prob 与人类 held-out preference；偏好评测还应附逐标注者 raw judgment、case/rater/order 完整性 gate、raw agreement/Fleiss’ κ 的 numerator/denominator、逐 case position estimand 与 case-cluster interval。顺序覆盖或 blind flag 只是协议声明，不能包装成随机化、真实盲化或因果位置偏差证据；tiny authored-pair overfit 和 authored judgment fixture 只能写控制流/统计口径回归。即使 preference readiness 执行了 source/license/sensitive/near-duplicate gate，registry allow 也不能写成“法律许可”，扫描未命中不能写成“无 PII/secret”，exact/lexical gate 通过不能写成“无泄漏”，无密钥 hash 不能写成“签名/防篡改”；QLoRA 不能简写成“全流程 4-bit”，因为通常只是冻结底座低位存储，而计算、adapter、梯度、optimizer 和激活仍有更高精度状态。

## 项目四：模型评测平台

**问题**：Prompt、模型、索引与工具升级缺少统一门禁。

**工作**：严格 JSONL schema、ordered case/result/answer/metric/system run manifest、绑定全部 bootstrap/gate 配置与 run identity 的 comparison artifact、case/slice、raw judgment、case/rater/order 完整性门禁、agreement/κ、逐 case position diagnostic、runner、配对 bootstrap、judge 校准、报告和 CI gate。

**指标**：人工一致性、judge precision、回归检出率、评测耗时/成本、发布阻断与线上事故关联。

**最低证据包**：不可变 case id 之外，还要用 manifest 绑定每个 case 的 input/gold/slice/metadata、ordered baseline/candidate result、recorded output identity、metric implementation revision、scorer 与 system revision；comparison artifact 再绑定 bootstrap seed/sample/confidence、质量/安全/延迟/slice 阈值、统计结果和全部失败原因。展示 duplicate key、同 ID 换 gold、结果/manifest/threshold 篡改和 metric revision mismatch 被拒绝。再附逐标注者原始判断与 assignment/presentation metadata、专家校准子集、agreement 的明确分母、paired difference 与 case-level 置信区间。unsigned fingerprint 和自报 system id 不能写成真实模型来源认证或不可篡改历史；若没有真实人类实验或发布历史，就只能展示 authored fixture/离线 gate 行为，不能声称标注一致性、因果位置偏差或线上事故下降。

## 推荐的第四层：从结果追到原始 artifact

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
