# DeepSeek 家族

## 学习目标与证据边界

读完本章应能区分 DeepSeek 的 MoE 训练/服务问题、Multi-head Latent Attention（MLA）的 KV 压缩思路，以及推理模型的后训练与 test-time compute；还能判断一个蒸馏 checkpoint 是否真的使用 DeepSeek-V3 架构。

**先修知识**：MHA/GQA、KV Cache、MoE routing、SFT/偏好优化、强化学习、量化与服务基准。

“DeepSeek”同时指研究路线、开放 checkpoint 和云产品。具体 checkpoint 是否包含 MLA、MoE、Multi-Token Prediction 或某种后训练机制，必须看其技术报告、config、代码与 model card，不能因品牌相同就默认。

## 先分开三条技术线

1. **训练与架构效率**：稀疏专家、路由、负载均衡和并行通信；
2. **推理内存效率**：MLA 等潜在表示压缩怎样改变 KV Cache；
3. **推理行为后训练**：可验证奖励、强化学习、SFT/冷启动数据、蒸馏和 test-time compute。

三条线可能出现在同一技术报告中，但解决的问题不同。MoE 不自动带来推理能力，MLA 不等于量化，强化学习也不改变 checkpoint 的基础 attention 结构。

## DeepSeekMoE 的系统视角

MoE 层把 token 路由到少数专家。若总专家参数为 \(P_{total}\)，每 token 只激活其中 \(P_{active}\)，前向 FLOPs 可能接近较小 dense 模型；但单卡加载通常仍要容纳或访问大量总权重。

公平报告至少包括：

- 总参数、激活参数与非专家共享参数；
- experts 数、top-k 与是否有 shared experts；
- 每 token FLOPs 和实际 tokens/s；
- router 分布、expert utilization 与 dropped/overflow token；
- expert parallel 拓扑和 all-to-all 时间；
- 权重驻留、通信 buffer 与峰值显存。

路由不均会让少数专家成为 straggler；小 batch 下每个专家收到的 token 太少，矩阵乘效率也可能很差。辅助负载均衡损失或其他路由策略会影响训练信号，不能只看最终 perplexity。

仓库的通用 NumPy MoE fixture 可用于练习 top-k、per-expert capacity、assignment overflow、gate combine 与线性 expert dispatch，但其 capacity 公式、score-priority 和 `E*sum(f_e*p_e)` 只是明确的教学契约。它没有实现 DeepSeekMoE 的细粒度/共享专家、特定无辅助损失策略、通信布局或 checkpoint config，不能把 fixture 结果标成 DeepSeek-V2/V3/R1 架构复现。

## MLA：为什么能改变 KV Cache

标准 MHA 在每层、每 token 缓存各 KV head 的 key 与 value：

\[
M_{MHA}\propto 2\,L\,H_{kv}\,d_h\,T\,bytes(dtype)
\]

MLA 的核心学习视角是：先把与 K/V 相关的表示压到更低维 latent，再缓存该 latent 与必要的位置相关分量，在注意力计算时通过投影恢复所需表示。于是缓存维度不再简单等于 `num_kv_heads × head_dim`。

这带来三个重要结论：

1. 不能把标准 GQA/MHA 公式直接用于 MLA checkpoint；
2. cache 更小不代表所有计算更少，恢复投影和 kernel 实现仍有成本；
3. 理论字节减少只有被 runtime 的专用 kernel 利用时才转化为吞吐或并发收益。

精确容量必须按对应技术报告/config/runtime 的实际 cache layout 计算。本仓库通用 `estimate_kv_cache_bytes` 明确只适用于理想化 dense K/V，不适用于 MLA。

可以用以下本地反例检查这条边界：

```powershell
python projects/transformers-basics/inspect_config.py `
  projects/transformers-basics/configs/mla-moe.example.json --tokens 4096
```

输出必须给出 `estimate_refused: true`。该文件的 `AuthoredMLAMoECausalLM`、层数、head 与 expert 数完全是本仓库自编 fixture，不是 DeepSeek-V2/V3/R1 配置快照；它只证明检查器看到一组已知 MLA marker 后不会误套标准 GQA/MHA 公式。反过来，没有这些 marker 也不能证明某个未知自定义架构一定适用标准公式。

## V2/V3 报告怎样阅读

公开 V2/V3 工作把 MoE、MLA 和训练系统优化放在一起讨论。学习时应建立“论文主张—实现配置—可复现实验”三列表，而不是把技术报告中的全部机制套到每个衍生模型。

例如，技术报告可能描述特定训练精度、负载均衡策略、并行通信或 Multi-Token Prediction；下载的蒸馏模型、API 模型或第三方量化文件不一定保留同样结构和训练过程。检查：

```text
architectures / model_type
num_hidden_layers / hidden_size / attention fields
MoE expert and routing fields
MLA latent and RoPE-related fields
tokenizer / chat template / generation config
base architecture declared by model card
weight hash / quantization method / license
```

这三类 JSON 不能各看各的：tokenizer、model config 与 generation config 的 BOS/EOS/PAD 可能不完全相同。EOS superset 可能有意加入 turn-end token，disjoint 或越界 ID 则至少需要人工复核。仓库 `inspect_generation_protocol.py` 只做显式快照对账；它不证明蒸馏 Qwen/Llama checkpoint、DeepSeek-V3 或云 API 最终采用相同停止协议。

## R1 与推理后训练

DeepSeek-R1 公开工作讨论通过强化学习、可验证奖励、冷启动/SFT 数据与蒸馏提升推理行为。这里要区分：

- **结果可验证任务**：数学、代码等可用规则、编译器或测试给奖励；
- **开放任务**：写作、事实综合与价值判断没有单一可靠 verifier；
- **训练时 RL**：更新参数分布；
- **推理时采样/搜索**：不更新参数，用更多计算产生候选；
- **蒸馏**：让另一个基座学习 teacher 产生的数据/行为。

R1 Distill 类 checkpoint 可能建立在 Qwen 或 Llama 架构之上。它们学习了推理数据/行为，不因此变成 DeepSeek-V3 的 MLA/MoE 架构；显存公式、LoRA target 和 runtime 支持应按实际 base config 判断。

可见 reasoning 文本可能冗余、错误、事后合理化或含敏感内容，不是内部机制的完美解释。生产系统可保留简短可审计 rationale、验证器结果和工具证据，不应把未经校验的长轨迹当作事实来源。

## Test-time compute 的公平评测

生成更长轨迹、增加候选或使用 verifier 有时提升可验证任务质量，但收益不保证单调。比较至少固定或报告：

| 轴 | 必须记录 |
|---|---|
| 输出预算 | max tokens、实际 tokens、停止原因 |
| 采样 | temperature/top-p、seed、候选数 |
| 验证 | verifier 版本、工具调用、选择规则 |
| 性能 | TTFT、E2E、tokens/s、并发 |
| 成本 | 每请求与每成功任务成本 |
| 质量 | final answer、pass@1/pass@k、失败 taxonomy |

不能把给模型 8 次机会的 pass@8 与另一个模型单次成功率直接比较，也不能忽略被截断的 reasoning 输出。

## API 与开放权重

DeepSeek 云 API 可能提供 OpenAI-compatible 请求形状，但 provider 端模型、量化、路由、上下文、缓存、内容政策和模型更新与开放 checkpoint 不同。兼容只说明部分字段相似，不保证 reasoning 字段、tool calling、stream usage、错误或限额相同。

API 实验记录 provider、base URL、model id、checked_at、请求字段、usage、finish reason、request id 与重试；本地实验记录 checkpoint hash、tokenizer/template、runtime、量化和硬件。两类结果不能共用一个含糊的“DeepSeek 分数”。

## 单卡与服务实践

大型 MoE checkpoint 即使每 token 激活参数较少，也可能因总权重无法在单张消费 GPU 上运行。选择小尺寸 dense、蒸馏 checkpoint 或可靠量化时，先确认它们的真实 base architecture 与许可。

单卡路线：

1. 固定 revision 并运行 config inspection；
2. 确认 dense/MoE、MHA/GQA/MLA 和 tokenizer/template；
3. 计算权重与实际 cache layout 的容量；
4. 建立 Transformers 正确性基线；
5. 检查目标 runtime 是否有该架构/量化的原生 kernel；
6. 在相同 token budget 下比较普通/推理模式；
7. 分开报告质量、显存、TTFT、TPOT 与每成功任务成本。

若使用蒸馏 Qwen/Llama checkpoint 做 LoRA，target modules 与 cache 公式服从它的 base architecture，不服从 teacher 品牌。

## 可运行实验

选择一个可在本机运行的 DeepSeek 蒸馏或小尺寸开放 checkpoint：

1. 导出 config，证明其 base architecture、attention 与是否 MoE；
2. 对同一组数学/代码/事实题设置短、中、长三档输出预算；
3. 保存 final answer、实际 token、截断、wall time 和 verifier 结果；
4. 比较 greedy、多个候选 + verifier，以及非推理基线；
5. 检查 pass@k 的收益是否抵得上成本；
6. 用错误 taxonomy 区分推理错、知识错、格式错、截断和 verifier 错。

这比展示几个“长思考过程”更能说明系统价值。

## 常见错误

- 把 DeepSeek 品牌下所有 checkpoint 都写成 MLA + MoE；
- 把激活参数当成加载显存或实际吞吐；
- 用标准 KV 公式估 MLA cache；
- 把蒸馏行为当作 teacher 架构复制；
- 比较推理模型时不控制输出 token、候选数和 verifier；
- 认为可见 reasoning 文本天然真实、安全或可作为审计结论；
- 把 OpenAI-compatible 当作云 API 与开放权重完全等价。

## 面试追问

1. MoE 为什么可能被通信和负载不均而不是 FLOPs 限制？
2. 总参数、激活参数和每 token FLOPs 分别影响什么？
3. MLA 怎样改变 KV Cache 表示，为什么需要专用容量公式？
4. R1 Distill checkpoint 为什么不必使用 DeepSeek-V3 架构？
5. test-time compute 怎样做同预算比较？
6. verifier 有哪些 reward hacking 和分布外风险？
7. API 模型与开放权重怎样建立可复现的对比协议？

## 一手资料

- DeepSeek-AI，[DeepSeek-V3 official repository](https://github.com/deepseek-ai/DeepSeek-V3)，技术报告、模型卡与运行入口。
- DeepSeek-AI，[DeepSeek-R1 official repository](https://github.com/deepseek-ai/DeepSeek-R1)，推理后训练与蒸馏 checkpoint 说明。
- DeepSeek-AI，[DeepSeek-V2](https://arxiv.org/abs/2405.04434)，DeepSeekMoE 与 MLA 公开描述。
- 目标 checkpoint 的 config、tokenizer、model card 和 runtime 支持矩阵；具体部署的最高优先级证据。
