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

**专题导航**：[推理系统](reasoning-systems.md) · [长上下文](long-context-systems.md) · [MoE 系统](moe-systems.md) · [证据台账](../evidence/frontier-controls.md)
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

## 它们在一次请求的哪里发生 { #combined-request }

这份合同先作为长输入进入模型。模型每经过一个 MoE 层，当前 token 都要路由到专家；完成前向计算后，
系统生成四条候选，最后再由 verifier 选择。它们在同一请求里相遇，却仍然需要三份独立证据。

~~~mermaid
flowchart TB
    A["24k-token 合同"] --> B["Attention / 共享层"]
    B --> C["每个 MoE 层的路由"]
    C --> D["选中专家并合并结果"]
    D --> E["四条候选答案"]
    E --> F["Verifier 选出最终答案"]
~~~

排查时分别记录：

- 长上下文：输入长度、答案所在位置、缓存、首 token 延迟和任务正确性；
- MoE：每个专家收到与实际执行的 token 数、丢弃或改派数量、跨设备通信和负载；
- 推理过程：每条候选、集合中是否存在正确答案、最终选择、工具调用、token 用量和任务结果。

最后答案正确，不能反推每一层都正确；最终失败，也不能只凭输出判断是哪一层造成。

## 看到什么证据，能下多强结论 { #evidence-strength }

| 证据 | 例子 | 能回答什么 |
|---|---|---|
| 论文/技术报告 | 方法与受控实验 | 作者在固定设置下主张什么 |
| 模型配置与源码 | 专家数量、位置字段、生成选项 | 代码具备哪些实现入口 |
| 可运行的机制样例 | 本仓库提供的路由、采样与缓存参考实现 | 局部公式和状态变化是否一致 |
| 目标运行环境 | 真实缓存、路由和候选记录 | 指定版本实际走了哪条路径 |
| 目标负载评测 | 任务、硬件、流量和成本 | 这套方案是否值得发布 |

不能把一篇论文、一个 config 字段和一个通用 toy 拼成目标模型已验证。

## 共同的实验方法

三条路线都用同一实验纪律。

### 1. 写资源预算

~~~text
input tokens
output / sampled tokens
model and verifier calls
tool calls
wall-clock deadline
GPU memory / parallelism
cost
~~~

### 2. 固定 baseline

- Reasoning：single sample/direct answer。
- Long context：short context 或 RAG baseline。
- MoE：dense 或不同 expert/capacity 配置。

### 3. 每次只改变一个主变量

候选数、context length、expert capacity、quantization 和 runtime 不要同时变化。

### 4. 保留完整分母

请求超时、显存不足、输入截断、输出无效、token 被丢弃、工具报错和 verifier 选择失败都要进入分母。

### 5. 解释边界

三层实验回答三种问题：机制样例检查局部公式，目标模型运行检查一个具体配置，生产压测检查真实容量。
后一级的结论都需要新的输入和环境，不能从前一级直接推出。

## 三种常见错误归因

### “输出更长，所以 reasoning 更强”

长 rationale 可能包含重复、错误或事后解释。应比较固定 token/call budget 下的 verified task success。

### “支持 1M tokens，所以能可靠使用 1M tokens”

Protocol acceptance 只说明请求可能被接受。还要测 runtime completion 和不同位置/任务的 effective context。

### “总参数巨大、激活参数小，所以服务更快”

MoE 仍需存储/分片总权重，并承担 routing、expert imbalance、all-to-all 和 kernel overhead。

## 一个三路线对照项目

选择一个有可执行 verifier 的小任务集，例如受限数学或代码：

1. **Reasoning 条件**：固定 context，比较 single sample 与 best-of-N。
2. **Long-context 条件**：固定生成预算，改变 evidence 位置和 distractors。
3. **MoE 条件**：只用本仓库提供的路由样例，改变 capacity 和 token distribution。

三组结果分别回答下面的问题，因此分开报告：

- 候选变多后，正确答案是否更常出现在候选集合中，verifier 又能否把它选出来？
- 输入更长后，答案证据是否仍能被找到和整合？
- Capacity 改变后，哪些 tokens 被接受、丢弃或 reroute？

精确复算脚本和记录结果见[前沿证据台账](../evidence/frontier-controls.md)。

## 学习顺序建议

### 应用/Agent 工程师

1. [推理系统](reasoning-systems.md)
2. [长上下文](long-context-systems.md)
3. MoE 只读 total/active 与部署边界

### 推理部署工程师

1. [长上下文](long-context-systems.md)
2. [MoE 系统](moe-systems.md)
3. 推理系统中的 offered budget 与多候选成本

### 训练/算法工程师

1. [MoE 系统](moe-systems.md)
2. [推理系统](reasoning-systems.md)
3. [长上下文](long-context-systems.md)

## 常见错误

- 用“reasoning model”品牌名代替具体训练和 test-time protocol。
- 用长输出、judge 自信或 reward score代替 verified success。
- 把 context window、KV capacity 和 effective context 写成同一个数字。
- 把 Long context 当作 RAG 的替代，不做权限和来源管理。
- 把 total、active、resident parameters 混为一谈。
- 用单进程 routing toy 声称 expert-parallel runtime 已验证。
- 只报最好配置，不报告 tokens、calls、latency、OOM 和 dropped 分母。

## 面试时怎样回答

面对“介绍大模型前沿扩展”，先用一句话分开三条路线：

- Reasoning 增加每题的 test-time computation。
- Long context 增加一次调用可读取的信息范围。
- MoE 增加总容量但每 token 只激活部分 experts。

再为每条路线给出一个核心风险和一个验证方法。不要列模型名称代替机制。

## 自测

1. Self-consistency 增加的是模型参数、输入信息还是 test-time compute？
2. 请求被 API 接受，为什么不等于有效 context 得到验证？
3. MoE active parameters 少，为什么仍可能需要多 GPU？
4. 三条路线组合在同一系统时，应分别记录哪些 trace？
5. 哪种证据才能支持“这个改动值得在目标负载发布”？

## 继续学习

- [推理系统](reasoning-systems.md)：sampling、search、verifier 与 tools。
- [长上下文系统](long-context-systems.md)：position、attention、KV、RAG 与评测。
- [MoE 系统](moe-systems.md)：router、capacity、通信与部署。
- [前沿证据台账](../evidence/frontier-controls.md)：数学小实验、collective 验证程序和适用范围。
