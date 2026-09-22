# 前沿系统总览：推理、长上下文与 MoE

<!-- learning-contract -->
<div class="learning-contract" markdown="1">

**学习导航**

- **适合读者**：已经理解 Transformer，希望系统进入 reasoning、long context 和 MoE 的工程师。
- **先修**：[Transformer](../core/transformer.md)、[生成](../core/generation.md)与基础评测。
- **首次阅读**：先比较三类扩展，再只选择一条专题路线。
- **完成信号**：能说明每条路线增加了什么资源、解决什么问题、需要什么证据。
- **卡住时**：把“模型更大/想得更久/输入更多”分别写成三个系统预算。

</div>

**专题导航**：[推理系统](reasoning-systems.md) · [长上下文](long-context-systems.md) · [MoE 系统](moe-systems.md) · [端侧小模型与本地智能](on-device-small-models.md) · [证据台账](../evidence/frontier-controls.md)
{ .doc-nav }

假设一个合同审查系统答错了“提前解约需要提前多少天通知”。团队提出三个改法：

1. 对同一问题生成四份答案，再选最好的一份；
2. 把整份合同都放进上下文；
3. 换成总参数更多、每个 token 只激活部分专家的 MoE 模型。

这三种改法解决的不是同一个瓶颈。第一种增加每道题在推理阶段使用的计算，第二种增加模型一次能读取的
信息，第三种增加模型总容量并改变每层前向计算的组织方式。

| 路线 | 它扩大什么 | 随之增加的系统负担 |
|---|---|---|
| Reasoning / test-time compute | 每题的采样、搜索、验证或工具调用 | 输出 token、调用次数、延迟和选择错误 |
| Long context | 一次调用可访问的信息范围 | Attention、KV Cache、位置可靠性、干扰和成本 |
| MoE | 模型总参数容量，同时让每个 token 只经过部分专家 | 权重存储、路由失衡和跨设备通信 |

先定位失败原因。如果正确条款根本没有进入输入，应优先修检索或上下文；如果条款已经在输入中，但任务需要
比较多个条件，可以增加推理预算；如果要更换模型容量与计算结构，才进入 MoE 选择。三条路线都可能有帮助，
也都可能让结果更差。

## 用一笔请求看清三种预算 { #request-ledger }

下面的数字只用于建立账本，不代表任何模型的默认配置：

```text
input:
  total_tokens: 24,000
  answer_position: around 18,000
model:
  routed_experts_per_moe_layer: 64
  experts_selected_per_token: 2
generation:
  candidates: 4
  max_output_tokens_each: 1,000
selection:
  verifier_calls: 1
```

**长上下文账本**关心 24,000 个输入 token 能否被协议接受、运行时能否完成 prefill，以及模型能否在
18,000 附近找到并整合条款。输入成功并不等于答案证据被有效使用。

**MoE 账本**发生在模型每个 routed layer 内。每个 token 都会单独选择专家；top-2 表示产生两个专家任务，
不是整条请求只使用两个专家。系统要记录各专家收到和实际执行的 token 数，以及丢弃、改派和跨设备通信。

**Reasoning 账本**关心四条候选及一次选择。四条候选最多使用 4,000 个输出 token，另外还有 verifier 的
调用与 token。若候选中至少一条正确，`oracle@4` 记为成功；只有 verifier 最后选中了正确候选，
`selected@4` 才成功。前者衡量候选集合，后者才是系统交给用户的结果。

四个候选可以共享同一次 prefill，也可能由四次独立调用生成；前缀缓存、批处理和供应商计费规则都会改变
真实成本。因此，不能把上面的数字直接相加后当作 GPU 计算量或账单。

## 用一个问题判断该学哪条

### 输出需要更多计算步骤

例如数学、代码、规划或多工具任务。先进入[推理系统](reasoning-systems.md)，学习 sampling、self-consistency、best-of-N、verifier 与预算曲线。

### 答案依赖更多输入证据

例如整份合同、长代码库、多文档冲突或长会话 memory。先进入[长上下文系统](long-context-systems.md)，区分协议长度、runtime 完成和有效任务长度。

### 模型容量想增大但不希望每 token 激活全部参数

进入 [MoE 系统](moe-systems.md)，跟踪路由器怎样为每个 token 选择专家、专家容量怎样限制实际执行，
再理解跨设备专家并行，以及总参数与每次激活参数的区别。

如果问题是实时事实、权限或副作用，可能更需要 RAG、tools 和 system policy，而不是三条路线中的任何一种。
如果约束来自设备内存、能耗、离线可用性或数据不能外发，先读[端侧小模型与本地智能](on-device-small-models.md)：它把
本地模型、数据 policy、router、verifier 与受控升级放进同一条请求路径。

## 它们在一次请求的哪里发生 { #combined-request }

合同先作为长输入进入模型；每经过一个 MoE 层，当前 token 都会路由到专家；模型随后生成多条候选，
最后由 verifier 选择。三种技术可以出现在同一请求中，但仍各自需要独立证据。

~~~mermaid
flowchart LR
    A["长输入与证据位置"] --> B["Attention / MoE 路由"]
    B --> C["候选生成与工具"]
    C --> D["Verifier 选择"]
    D --> E["业务终态"]
~~~

| 路线 | 必须记录的主变量 | 最容易误判 |
|---|---|---|
| Reasoning | 候选、oracle/selected、model/verifier/tool calls、输出 token、wall time、失败终态 | 输出更长或 proxy 更高不等于任务更成功。 |
| Long context | Tokenized length、证据位置、distractors、integration、KV、TTFT、截断/OOM | 请求被接受不等于可靠使用全部输入。 |
| MoE | Total/active parameters、tokens/expert、capacity、drop/reroute、collective、memory 与 tail latency | 激活参数少不等于加载更小或服务更快。 |

## 共同证据要求 { #evidence-strength }

先固定 single-sample、short-context/RAG 或 dense 等基线；每次只改变候选数、context length、capacity、
quantization 或 runtime 中的一个主变量。所有 timeout、OOM、truncation、invalid output、dropped token、
tool error 与 verifier miss 都进入分母。

| 证据 | 能回答 | 不能直接推出 |
|---|---|---|
| 论文 / 技术报告 | 作者在固定设置下主张什么 | 当前 checkpoint 或服务已经采用 |
| Config / 源码 | 存在哪些结构和实现入口 | 权重匹配、实际执行或性能 |
| 机制 control | 局部公式、状态与反例是否一致 | 目标模型、GPU 或生产表现 |
| 目标 runtime | 指定版本实际经过哪些 cache/router/search 路径 | 代表性任务与 SLO |
| 目标 workload | 在给定硬件、流量、预算下是否值得发布 | 其他分布、版本或组织环境 |

精确复算值和 CPU/Gloo 边界查[前沿证据台账](../evidence/frontier-controls.md)。

## 三个专题出口

| 当前问题 | 进入 | 离开该页时应得到 |
|---|---|---|
| 输出需要更多计算、候选或 verifier | [推理系统](reasoning-systems.md) | 一条固定总预算的 quality/cost 曲线，并分开 oracle 与 selected |
| 答案依赖更多输入证据 | [长上下文系统](long-context-systems.md) | 一张 position × distractor × integration 评测矩阵 |
| 想扩大总容量但不激活全部参数 | [MoE 系统](moe-systems.md) | 一份 routing/capacity/collective 账本 |
| 约束来自设备、能耗、离线或数据不外发 | [端侧小模型与本地智能](on-device-small-models.md) | 本地 model/router/verifier 与升级边界 |

应用/Agent 工程优先 reasoning→long context；部署工程优先 long context→MoE；训练工程优先 MoE→reasoning；
端侧工程先进入本地智能。这里的顺序只是入口，不把其他专题变成隐性前置。

## 判断是否选错方向

- 正确条款没有进入输入：先修 RAG、ACL 或 context construction。
- 证据已在输入但需要组合、搜索或执行验证：再增加 reasoning/test-time compute。
- 模型容量是瓶颈且运行平台能承担权重与通信：再评估 MoE。
- 需要更新事实、来源引用或租户隔离：长 context 不能替代 RAG。
- 只看到长输出、标称窗口、active 参数或理论 Big-O：证据还不足以声称能力或性能提升。

完成本页后，只选择一个专题继续，并写下 baseline、主变量、完整分母和最可能推翻结论的实验。
