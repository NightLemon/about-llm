# DeepSeek 家族

## 分开理解三条线

DeepSeek 的公开研究与权重涉及基础/代码/数学、MoE 架构和推理模型。学习时分开：

1. **训练/架构效率**：稀疏专家、路由和通信；
2. **推理内存效率**：公开工作中的 Multi-head Latent Attention 等表示压缩思路；
3. **推理行为后训练**：可验证奖励、强化学习、蒸馏和 test-time compute。

具体 checkpoint 是否包含某机制必须看其技术报告与 config，不能因品牌相同就默认。

## MoE

MoE 总参数多，但每 token 路由到部分专家。性能取决于激活参数、router、capacity、负载均衡和 all-to-all。单卡推理仍可能需要容纳大量总权重；“计算像小模型”不等于“内存像小模型”。

公平报告总参数、激活参数、每 token FLOPs、显存、通信和实际 tokens/s。

## MLA 的学习视角

标准 MHA KV Cache 随 K/V 头、长度、层数和 head dim 增长。潜在表示压缩的目标是减少需要缓存/读取的维度，再在注意力计算中恢复所需表示。理解重点是内存—计算—表达能力权衡，而不是只记缩写。

## 推理模型

生成更长的中间轨迹或增加采样/验证可提升可验证任务，但收益不保证单调。评测同时固定：

- 最大输出 token 和真实使用 token；
- 候选数、验证器和工具；
- wall time 与费用；
- 最终正确率和失败类别；
- 是否泄漏 benchmark/答案格式。

可见 reasoning 文本不是内部机制的完美解释，也可能包含错误或敏感内容。

## API 与开放权重

DeepSeek 云 API 可能提供 OpenAI-compatible 形状，但模型、路由、上下文和产品策略与开放 checkpoint 不同。固定 provider/model 和检查日期，保留 usage 与 finish reason。

## 面试追问

1. MoE 为什么可能被通信而不是 FLOPs 限制？
2. 激活参数与总参数分别影响什么？
3. KV 压缩怎样改变 decode 瓶颈？
4. test-time compute 如何做同成本比较？
5. 蒸馏推理轨迹有哪些错误放大风险？
