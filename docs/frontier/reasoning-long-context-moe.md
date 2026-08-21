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

Reasoning、long context 和 Mixture of Experts（MoE）常被放在“前沿模型”同一页，但它们扩大的是三种不同资源：

| 路线 | 扩大什么 | 主要系统代价 |
|---|---|---|
| Reasoning / test-time compute | 每个问题的采样、搜索、验证或工具调用 | tokens、calls、延迟、verifier 风险 |
| Long context | 一次调用可访问的信息范围 | attention/KV、位置可靠性、干扰和成本 |
| MoE | 模型总参数容量，保持部分激活计算 | 权重存储、routing、负载与通信 |

三者都可能提升某些任务，也都可能让结果更差。更多 tokens 会污染轨迹，更长 context 会增加干扰，更多 experts 会产生路由不均和通信瓶颈。

## 用一个问题判断该学哪条

### 输出需要更多计算步骤

例如数学、代码、规划或多工具任务。先进入[推理系统](reasoning-systems.md)，学习 sampling、self-consistency、best-of-N、verifier 与预算曲线。

### 答案依赖更多输入证据

例如整份合同、长代码库、多文档冲突或长会话 memory。先进入[长上下文系统](long-context-systems.md)，区分协议长度、runtime 完成和有效任务长度。

### 模型容量想增大但不希望每 token 激活全部参数

进入[MoE 系统](moe-systems.md)，学习 router、top-k、capacity、expert parallel 和 total/active parameters。

如果问题是实时事实、权限或副作用，可能更需要 RAG、tools 和 system policy，而不是三条路线中的任何一种。

## 三条路线怎样组合

一个推理 API 可能运行 MoE 模型、读取长 context，再为同一题采样多条候选。这不表示三种机制可以共用证据。

~~~mermaid
flowchart LR
    A["Long input"] --> B["Model forward"]
    C["MoE routing inside model"] --> B
    B --> D["Candidate trajectories"]
    D --> E["Verifier / search"]
    E --> F["Final answer"]
~~~

每层都有自己的观察：

- Long context：输入 token、位置切片、cache、TTFT 和任务正确性。
- MoE：expert selection、accepted/dropped tokens、all-to-all 和负载。
- Reasoning：candidate、oracle@k、selection、tool calls、tokens 和最终成功。

最后答案正确，不能反推每一层都正确；最终失败，也不能只凭输出判断是哪一层造成。

## 统一的证据阶梯

| 层级 | 例子 | 能回答什么 |
|---|---|---|
| 论文/技术报告 | 方法与受控实验 | 作者在固定设置下主张什么 |
| Config/code markers | expert 数、position fields、生成选项 | 静态实现候选 |
| 机制样例 | 本仓库提供的 routing、sampling 与 cache 参考实现 | 局部数学与状态机 |
| Target runtime | 真实 cache、routing、candidate trace | 指定实现实际做了什么 |
| Workload evaluation | 任务、硬件、负载和成本 | 候选系统是否值得发布 |

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

Timeout、OOM、truncation、invalid output、dropped token、tool error 和 verifier failure 都进入统计。

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

- 更多 candidate 是否提高 oracle，verifier 能否选对？
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
