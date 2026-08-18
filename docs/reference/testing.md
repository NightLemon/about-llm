# 教材测试与证据策略

本仓库的测试首先服务于教材可信度，其次才是代码回归保护。
测试通过只说明给定输入、实现路径与 oracle 一致，不自动证明理论、事实来源、模型质量、生产性能或系统绝对安全。

## 从教学 claim 开始

新增或修改测试前，先写清五件事：

1. 教材提出的最小可证伪 claim 是什么？
2. Oracle 来自手推公式、标准、独立实现、gold fixture，还是被测代码自身？
3. 测试实际证明了什么？
4. 哪些假设和外推没有被验证？
5. 最小正例、边界例与反例分别是什么？

如果 expected value 是从被测实现复制或运行后抄回，测试只能锁定当前行为，不能充当正确性 oracle。

## 三个彼此独立的维度

### 证据性质

| Marker | 回答的问题 |
|---|---|
| `formula` | 数学定义或算法不变量是否与教材一致？ |
| `contract` | 输入、输出、artifact 或协议边界是否明确且 fail closed？ |
| `security` | 授权、审批、预算或发布安全边界是否拒绝危险路径？ |
| `smoke` | 读者入口、CLI、Notebook 或 package 是否至少能够执行？ |

一个测试可以同时属于多类。分类不代表结论强度；`security` 测试通过尤其不能推出“系统不存在未知漏洞”。
Collection 门禁要求每个测试至少声明一种证据性质；否则新增用例会直接失败，而不是悄悄成为无法解释的回归测试。

### 运行属性

| Marker | 含义 |
|---|---|
| `integration` | 跨 framework、进程、socket、数据库或其他组件边界 |
| `slow` | 单 case 在基线机器上通常超过一秒；速度不决定重要性 |
| `network` | 访问外部网络或可能产生费用，必须显式 opt-in |
| `gpu` | 需要 CUDA GPU，必须显式 opt-in |

### CI 调度

`extended` 是调度决定，不是证据类型。它必须同时带 `integration` 或 `slow`，以说明延后的真实原因。

- Pull request：运行所有非 `extended`、非 `network`、非 `gpu` 测试，以及完整文档、三个 Notebook 和跨平台 wheel smoke。
- Main push：在 PR 层之外补跑 `extended` transport/子进程矩阵。
- Weekly/manual：重复两层离线门禁，用于发现依赖和环境漂移。
- 真实网络、付费 API、目标 GPU 与生产 benchmark：仍属于外部验证，不能由默认 CI 冒充。

当前只有 3 个小型 Notebook，完整执行约 25 秒，仍直接保护读者学习入口，因此暂不降级。

## 测试增长规则

本仓库冻结的是无理由净增长，而不是机械冻结用例数：

- 新增关键教学 claim，应增加可信 oracle。
- 新增重复的 JSON、tamper 或 fingerprint 场景，优先参数化、合并或替换已有测试。
- 高风险安全边界可以净新增，但要写明威胁、失败模式与不能证明的范围。
- 教材或公共实现改变时，除当前章节外，还要运行所有受影响 claim 的回归测试。

覆盖率不能替代这套判断。未执行代码可能重要，执行过的代码也可能只有错误 oracle。

## 首批 claim 审查台账

审查日期：2026-08-18。这里记录 oracle 的来源和边界，不记录容易过时的“全绿”宣传数字。

| Claim ID | 教材 claim | Oracle 与测试 | 当前判断 | 不能推出 |
|---|---|---|---|---|
| `ATTN-001` | scaled dot-product attention 对 score 做缩放、softmax，再加权 V | `test_attention_numpy.py` 新增两 key 解析 fixture：score 为 `[0, ln(3)]`，概率为 `[1/4, 3/4]`，输出为 `5` | 手算 expected value，不调用第二份 attention 实现 | CUDA kernel、FlashAttention、性能或大模型质量 |
| `ATTN-002` | causal、GQA、RoPE 与 blockwise online softmax 满足文中不变量 | norm/相对位置、future mask、显式 KV repetition、dense/blockwise 对账与失败路径 | 多种性质相互补充；dense/blockwise 对账仍共享同一模块，不能单独视为完全独立实现 | padding/packing 全组合、GPU backend、HBM 流量 |
| `GEN-001` | 仓库采样策略按 repetition→temperature→top-k→top-p→inverse CDF 执行 | `test_sampling.py` 用精确概率、tie、阈值 crossing、平移不变性和 CDF 边界验证 | Oracle 对应仓库明确声明的 authored policy | Transformers/vLLM/provider 默认顺序或生成质量 |
| `EVAL-001` | Brier、equal-width ECE 与 tie-aware risk–coverage 使用声明的分母 | `test_calibration_metrics.py` 的手算四样本与 tie fixture | 解析 expected value 与实现分离 | calibration construct validity、总体误差或业务阈值 |
| `EVAL-002` | paired percentile bootstrap 对 paired differences 重采样 | `test_evaluation_statistics.py` 新增 constant-difference 退化分布与 zero-difference strict-improvement 语义；另有 seeded 非退化 fixture | 精确边界补足了原先主要依赖“同 seed 可重复”的弱点 | percentile coverage、case 独立性、cluster 选择或贝叶斯后验概率 |
| `TRAIN-001` | masked cross-entropy 只按监督 token 求 mean，零监督与越界 target 必须暴露 | PyTorch forward 在 loss 前拒绝；JAX 低层 primitive 返回 non-finite sentinel，checked train step 在 compiled update 前拒绝 | 修正了原测试把全 ignored batch 的有限 `0` 锁成正确行为的问题 | 数据语义正确、任意 Trainer 的 mask 或训练收敛 |
| `TRAIN-002` | token-mean accumulation 必须按整个 update window 的有效 token count 加权 | `test_gradient_accumulation.py` 用 `Fraction` 手算 `[1,3]` token 反例，并与独立 PyTorch backward control 分层 | 精确 oracle、错误基线与 framework control 证据互补 | 随机层、optimizer、DDP/FSDP、AMP、目标模型或质量 |
| `ALIGN-001` | Bradley–Terry/DPO、GAE/PPO 与 categorical policy gradient 使用正文给定符号和 mask | log-two/大 margin 精确值、两步 GAE 手算、正负 advantage clip、full-KL 反例、binary policy-gradient 手算与 finite difference | 核心目标的符号、边界和反例可解释；仍是 authored finite controls | 完整 RLHF/GRPO、reward 正确、训练稳定或目标模型改善 |
| `DATA-001` | trainer 只能消费已审计 combined artifact 中按原顺序绑定的 train subset | SFT/preference tests 分开篡改 content、order、membership、split、binary label 与 readiness binding | 训练输入和 held-out 隔离契约明确，错误路径 fail closed | 数据来源真实、许可充分、标签正确或没有语义泄漏 |
| `DATA-002` | exact/group/near-duplicate 是不同泄漏信号，不能互相冒充 | exact content/group 跨 split fixture；role-tagged user view 的 character 3-gram 手算 `14/16`；preference 比较 prompt 与四种 candidate cross-surface | numerator、denominator、normalization profile、serialization view 和 scope 都进入证据；公共 role framing 会影响短文本 | 阈值适合真实域、semantic equivalence、MinHash/ANN 召回或零泄漏 |
| `RAG-001` | tenant/principal ACL 必须先于 BM25 的集合统计和逐文档打分 | 同一可见语料加入跨租户与 ACL-blocked 文档，结果 ID 与 score 必须逐项不变；context/packer/reranker 再次授权 | 修正了旧测试只检查 result ID、未发现 global IDF/平均长度侧信道的问题 | timing/cache/共享基础设施侧信道、IAM 正确或生产隔离 |
| `RAG-002` | 空授权证据应在生成前拒答；有证据输出仍需引用与答案动作 gate | generator spy=0、missing/unknown/uncited 负例、public projection 不泄漏 raw output、ACL 与 supported label 正交 | 控制流、分母和本地 publication policy 可解释 | citation entailment、来源真实、模型质量或 provider 未计费 |
| `AGENT-001` | 模型 proposal 不拥有执行权；default deny、capability、server-resolved tenant 与逐次重新授权先于 handler | handler spy 在缺 capability、跨 tenant、indeterminate policy、撤权后的 cached replay 中均保持 0；allow 恢复后只读 cached value | 直接观察 handler 次数和 policy decision，不从最终文本猜权限是否生效 | resolver/policy 本身无副作用、集中 IAM 正确、未知漏洞或生产身份真实 |
| `AGENT-002` | side-effect approval 必须绑定 proposal、subject、task、resource、tool 与 policy 身份，handler claim 后才消耗预算 | 漂移反例、固定 canonical fingerprint vector、handler spy、pending ledger 与同 call-id conflict；失败/非法结果后重放仍不再进入 handler | 固定 digest 不依赖 runtime 输出，漂移矩阵逐字段挑战 identity；SQLite 只保护本地 claim | grant 签名/一次性消费、远端 effect 原子性、数据库防篡改或 exactly-once |
| `AGENT-003` | Agent trace gate 必须区分 proposal、policy、approval、handler attempt 与 verified effect，非布尔观测不得靠 truthiness 放行 | 手算 numerator/denominator 和安全失败组合；新增 direct Python API 的 `approved=1` 反例 | 修正了 JSONL loader 会拒绝、但 typed API 曾把整数 1 当作已审批并通过 gate 的不一致 | supplied trace 真实、recorder 防篡改、effect observer 正确或开放任务质量 |
| `AGENT-004` | local transactional outbox 是 at-least-once，不是远端 exactly-once | SQLite 原子回滚、并发 lease、stale ack 与 success-before-ack crash；provider spy 得到 2 requests、同 key、1 simulated effect | 状态、timeline、调用数与 effect 数分别断言，没有把 delivered receipt 当作唯一 oracle | 真实 provider honor idempotency key、broker/网络/断电、多区域恢复或外部 effect 唯一 |
| `CLOUD-001` | 每个云调用 attempt 必须发送前 reserve，之后 settle/cancel/mark uncertain；未知 usage 不能记作 0 | micro-USD 整数手算、并发 Barrier、MockTransport request spy、SQLite reopen/event timeline；固定 500→200 为 80 uncertain + 66 settled | 预算拒绝发生在下一次 transport 前，超额 usage 先入账再报 breach | 真实 provider 定价、错误计费语义、usage/账单真实性、跨机全局 quota 或成本优化 |
| `ARTIFACT-001` | frozen manifest/report/ledger 的 identity 输入必须在构造时快照，不能保留 caller-owned list/dict alias | 构造后修改原 `ordered_case_ids`、release records 与 release-evidence `to_dict()` 嵌套值；内部 tuple/mapping identity 必须不变 | 修正了三处 `frozen=True` 外壳内仍可被外部容器改写、导致 fingerprint 或 ledger 内容漂移的问题 | 存储不可变性、进程间 TOCTOU、来源认证、签名或可信发布者 |
| `ARTIFACT-002` | self-fingerprint 只绑定 canonical content；只有带密钥链和外部 trusted head 才能分别讨论认证与前缀回滚 | 未重哈希 tamper、协同重哈希 semantic drift、HMAC wrong key、artifact byte rehash、valid-prefix truncation with/without trusted head | 无密钥 artifact 允许合法地形成新 identity；测试明确不把新 hash 的自洽误报成 provenance | key custody、真实时间、公钥不可否认性、目录原子发布或 verify 后不变 |
| `PROTOCOL-001` | authored parser、official SDK memory、SDK stdio/HTTP 与 A2A loopback 是不同证据层，不能互借结论 | lifecycle/schema/handler spies、真实 subprocess/pipe/loopback counters、strict receipt 与独立 scope verifier | 相同 strict-JSON 反例保留在独立 loader 上是边界回归，不因为 payload 相同就合并成只测一个实现 | 完整 conformance、远程跨厂商互操作、OAuth/TLS、业务授权、真实 provider 或生产 supervisor |

本轮 1797 个 collected tests 已全部声明证据性质；其中 17 个真实 transport/子进程 case 属于 `extended`，其余离线公式、契约、安全和入口证据留在 PR 层。这个数字只记录 2026-08-18 的治理结果，不是覆盖率目标；后续仍以 claim 与 oracle 质量决定增删。

## 本地命令

~~~powershell
# Pull-request 测试层
python -m pytest -m "not extended and not gpu and not network"

# Main/定时专项层
python -m pytest -m "extended and not gpu and not network"

# 防止 marker 选择漏测的最终对账
python -m pytest
~~~

首次分类不要删除测试。先观察耗时、失败价值和重复范围，再决定合并、降级或保留。
