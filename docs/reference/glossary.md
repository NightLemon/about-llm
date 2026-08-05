# 术语表

> 英文缩写按字母检索；定义强调在 LLM 语境中的含义。

| 术语 | 含义 |
|---|---|
| Abstain | 证据不足或风险过高时明确不作答/转交，而非猜测。 |
| Activation | 神经网络某层对当前输入产生的中间数值；也指激活函数。 |
| Agent | 模型在状态—动作—观察循环中调用工具完成任务的系统。 |
| Alignment | 让系统行为更符合人类意图、价值、安全和权限边界的一组方法。 |
| Attention | 根据 query-key 匹配对 value 加权聚合的信息混合机制。 |
| Autoregressive | 按此前 token 条件逐个预测后续 token。 |
| Backpropagation | 用链式法则在计算图上高效求参数梯度。 |
| Batch | 一次并行处理的一组样本；LLM 还需区分 micro/global batch。 |
| Beam search | 保留多个累计得分最高部分序列的解码算法。 |
| Benchmark | 固定数据与协议的比较测试；可能饱和或受训练污染。 |
| BF16 | 8 位指数、7 位尾数的 16-bit 浮点格式，范围接近 FP32。 |
| BM25 | 基于词频、逆文档频率和长度归一化的经典稀疏检索算法。 |
| BPE | 反复合并高频相邻单元的子词 tokenizer 算法。 |
| Calibration | 预测置信度与真实正确频率匹配的程度。 |
| Causal mask | 屏蔽未来 token，保证自回归训练不泄露答案。 |
| Checkpoint | 可恢复训练/推理状态的参数及相关元数据快照。 |
| Chunk | RAG 中用于索引和检索的文档片段。 |
| Context window | 一次模型调用可处理的最大 token 范围；有效利用能力另需评测。 |
| Cross-attention | query 来自一序列、key/value 来自另一序列的注意力。 |
| Cross-entropy | 训练分类/语言模型常用的负对数似然损失。 |
| Data contamination | 评测内容或其近似版本进入训练数据。 |
| Decode | 推理中逐 token 生成的阶段；区别于一次性处理输入的 prefill。 |
| Dense model | 每个 token 大体激活相同全部层/参数的模型；与稀疏 MoE 相对。 |
| Distillation | 用教师模型的输出或分布训练学生模型。 |
| DPO | 直接从偏好对优化策略相对参考模型概率的训练方法。 |
| Embedding | 离散项目或输入映射得到的连续向量表示。 |
| EOS/BOS | 序列结束/开始特殊 token。 |
| Epoch | 完整遍历训练集一次；大规模预训练更常用 consumed tokens。 |
| Exposure bias | 训练看真实历史、推理看自身生成历史造成的分布差异。 |
| Fine-tuning | 在预训练模型上用目标数据继续更新参数。 |
| FlashAttention | 通过 IO-aware 分块计算精确注意力的高效算法族。 |
| FP16/FP32/FP8 | 不同位宽和范围的浮点格式。 |
| FSDP | 将参数、梯度、优化器状态跨数据并行设备分片的训练方法。 |
| Function/tool calling | 模型按 schema 提出外部函数/工具调用参数的机制。 |
| GQA | 多个 query 头共享一组 K/V 头，在质量与 KV 内存间折中。 |
| Gradient | 损失对参数的偏导数，指示局部最速上升方向。 |
| Gradient accumulation | 多个 micro-batch 累积梯度后再更新一次。 |
| Hallucination | 输出缺乏依据或与事实/给定证据矛盾；边界需按任务定义。 |
| In-context learning | 不更新权重，仅凭当前上下文中的说明/示例适应任务。 |
| Instruction tuning | 用指令—回答示例监督微调模型。 |
| KV Cache | 自回归推理缓存旧 token 各层 key/value 的内存。 |
| Latency | 完成请求所需时间；需区分 TTFT、TPOT 和端到端。 |
| Logit | softmax 前每个候选 token 的未归一化分数。 |
| LoRA | 用低秩矩阵参数化权重增量的 PEFT 方法。 |
| LLM-as-judge | 使用另一个模型按 rubric 评价输出；需做人类校准。 |
| Masked LM | 遮住部分 token 并利用双向上下文恢复它们的训练范式。 |
| MHA/MQA | 每个 query 头独立 K/V，或所有 query 头共享一组 K/V。 |
| Mixed precision | 不同算子/状态用不同数值精度以兼顾速度、内存和稳定性。 |
| MoE | Mixture of Experts，每个 token 只路由到部分专家的稀疏模型。 |
| NLL | Negative Log-Likelihood，正确观测的负对数概率。 |
| nDCG | 考虑相关等级与排名位置的检索排序指标。 |
| Parameter | 训练学习并持久保存的权重数值。 |
| PEFT | Parameter-Efficient Fine-Tuning，只更新少量新增/选定参数。 |
| Perplexity | 平均 NLL 的指数；不可跨 tokenizer 直接比较。 |
| PII | 可识别个人身份的信息。 |
| PagedAttention | 用分页方式管理 KV Cache、降低碎片的服务内存技术。 |
| Positional encoding | 将 token 顺序/距离信息注入模型的方法。 |
| Prefill | 对整个输入 prompt 并行前向并建立 KV Cache 的推理阶段。 |
| Prompt injection | 不可信输入诱导模型改变指令或执行越权动作的攻击。 |
| Prompt | 发送给模型的指令、上下文、示例和输出约束整体。 |
| Quantization | 用较低位宽表示权重、激活或 KV Cache。 |
| RAG | 生成前从外部知识源检索证据的系统模式。 |
| Red teaming | 从攻击者视角系统寻找滥用和安全失败。 |
| Reward model | 预测人类/AI 偏好分数的模型。 |
| RLHF/RLAIF | 用人类/AI 反馈构造奖励并优化模型行为。 |
| RMSNorm/LayerNorm | 稳定隐藏状态尺度的归一化方法。 |
| RoPE | 旋转 Q/K 分量以编码相对位置信息的位置方法。 |
| SFT | Supervised Fine-Tuning，使用理想输入输出对的监督微调。 |
| Softmax | 将任意 logits 归一化成总和为 1 的概率分布。 |
| Speculative decoding | 小模型草拟、大模型批量验证以减少串行解码步骤。 |
| SSM | State Space Model，以状态递推处理序列的模型族。 |
| Temperature | 缩放 logits、调节采样分布尖锐程度的参数。 |
| Tensor parallelism | 把单个层/矩阵运算切到多个设备。 |
| Token | tokenizer 词表中的离散单位，可能是字节、字符或子词片段。 |
| Top-k / Top-p | 仅从最高 k 个或累计概率达到 p 的最小集合采样。 |
| Transformer | 以注意力、MLP、残差和归一化堆叠为核心的序列架构。 |
| TTFT / TPOT | 首 token 延迟 / 每个输出 token 的平均时间。 |
| Vector database | 为向量近邻查询及元数据过滤优化的存储/检索系统。 |
| Weight decay | 抑制权重过大的正则化更新；AdamW 将其与梯度解耦。 |
| ZeRO | 分阶段分片优化器、梯度和参数的分布式训练方法。 |

若同一术语在某个框架中定义不同，以对应章节和官方实现为准。
